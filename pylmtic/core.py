import requests
import difflib
from typing import List, Type
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import re
from urllib.parse import urlunparse, urlparse

# --- Modellinformationen vom Ollama-Server ---
class OllamaModelInfo(BaseModel):
    id: str
    object: str
    owned_by: str


class OllamaModelList(BaseModel):
    data: List[OllamaModelInfo]
    object: str


def find_closest_model(models: List[OllamaModelInfo], model_name: str) -> OllamaModelInfo:
    """
    Wählt das Modell, das exakt oder am nächsten dem angegebenen model_name entspricht.
    """
    # exakte Übereinstimmung
    for m in models:
        if m.id == model_name:
            return m

    # semantische Nähe via difflib
    model_ids = [m.id for m in models]
    closest_matches = difflib.get_close_matches(model_name, model_ids, n=1, cutoff=0.0)
    if closest_matches:
        match_id = closest_matches[0]
        for m in models:
            if m.id == match_id:
                return m

    # fallback: erstes Modell
    return models[0]

class LMEndpoint(BaseModel):
    name: str
    protocol: str = Field(default="http", description="Protokoll: http oder https")
    host: str = Field(default="localhost", description="Hostname oder IP-Adresse")
    port: int = Field(default=1234, description="Portnummer (1-65535)")
    api_path: str = Field(default="/v1", description="API-Pfad, beginnt mit /")

    @field_validator("protocol")
    @classmethod
    def validate_protocol(cls, v):
        if v not in ("http", "https"):
            raise ValueError("protocol muss 'http' oder 'https' sein")
        return v

    @field_validator("host")
    @classmethod
    def validate_host(cls, v):
        hostname_regex = r"^(([a-zA-Z0-9\-]+\.)*[a-zA-Z0-9\-]+|localhost)$"
        ipv4_regex = r"^(?:\d{1,3}\.){3}\d{1,3}$"
        if not re.match(hostname_regex, v) and not re.match(ipv4_regex, v):
            raise ValueError("host muss ein gültiger Hostname oder IPv4-Adresse sein")
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError("port muss zwischen 1 und 65535 liegen")
        return v

    @field_validator("api_path")
    @classmethod
    def validate_api_path(cls, v):
        if not v.startswith("/"):
            raise ValueError("api_path muss mit '/' beginnen")
        return v

    def get_url(self) -> str:
        """
        Gibt die vollständige URL des Endpoints zurück, z.B. http://localhost:1234/v1
        """
        # urlunparse(tuple): (scheme, netloc, path, params, query, fragment)
        netloc = f"{self.host}:{self.port}"
        return urlunparse((self.protocol, netloc, self.api_path, "", "", ""))


class PyLMTic:
    """
    Wrapper für lokale LLM-Anfragen via Ollama/LMStudio, integriert pydantic_ai Agenten.
    """

    def __init__(self, model_name: str = "qwen", host_url: str | None = None, endpoints: List[LMEndpoint] | None = None):
        # Standardendpoints, falls keine übergeben werden
        self.endpoints = endpoints or [
            LMEndpoint(name="Ollama", host="localhost", port=11434, api_path="/v1"),
            LMEndpoint(name="LMStudio", host="localhost", port=1234, api_path="/v1")
        ]

        self.model_name = model_name
        self.models: List[OllamaModelInfo] = []
        self.selected_model: OllamaModelInfo | None = None
        self.model: OpenAIChatModel | None = None
        self.agent: Agent | None = None
        self.host_url: str | None = None

        # Wenn host_url angegeben ist → URL parsen und als Endpoint speichern
        if host_url:
            parsed = urlparse(host_url)
            protocol = parsed.scheme or "http"
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if protocol == "https" else 80)
            api_path = parsed.path if parsed.path else "/v1"

            self.endpoints = [LMEndpoint(name="Custom Endpoint", protocol=protocol, host=host, port=port, api_path=api_path)]
        else:
            # Standardendpoints (LMStudio & Ollama)
            self.endpoints = endpoints or [
                LMEndpoint(name="LMStudio", protocol="http", host="localhost", port=1234, api_path="/v1"),
                LMEndpoint(name="Ollama", protocol="http", host="localhost", port=11434, api_path="/v1"),
            ]

        # Endpoints durchtesten
        self._initialize()

    def _initialize(self):
        for endpoint in self.endpoints:
            test_url = f"{endpoint.get_url()}/models"
            try:
                response = requests.get(test_url, timeout=1)
                response.raise_for_status()
                models_json = response.json()
                self.models = OllamaModelList(**models_json).data

                if not self.models:
                    print(f"[WARN] Endpoint {endpoint.name} liefert keine Modelle, nächster Endpoint...")
                    continue  # nächsten Endpoint testen

                self.host_url = endpoint.get_url()
                print(f"[INFO] Erfolgreich verbunden zu {endpoint.name} ({self.host_url})")
                break  # erfolgreicher Endpoint gefunden
            except Exception as e:
                print(f"[WARN] Verbindung zu {endpoint.name} fehlgeschlagen: {e}")
        else:
            raise ConnectionError("Kein funktionierender Endpoint gefunden!")

        # nächstes Modell auswählen
        self.selected_model = find_closest_model(self.models, self.model_name)

        # OpenAIChatModel mit OllamaProvider erstellen
        self.model = OpenAIChatModel(
            model_name=self.selected_model.id,
            provider=OllamaProvider(base_url=self.host_url)
        )

        # Agent vorbereiten (Output-Type wird bei run_prompt übergeben)
        self.agent = Agent(self.model, output_type=None)


    def run_prompt(self, prompt: str, output_type: Type[BaseModel]) -> list[BaseModel]:
        """
        Führt einen Prompt über den Agenten aus.
        - `prompt`: Textanfrage an das Modell
        - `output_type`: Pydantic BaseModel für die Ausgabe (erforderlich)
        """
        if not issubclass(output_type, BaseModel):
            raise TypeError("output_type muss ein Pydantic BaseModel sein")

        # Agent mit Output-Type erstellen
        agent = Agent(self.model, output_type=List[output_type])
        result = agent.run_sync(prompt)

        # Ausgabe als Liste von Dictionaries zurückgeben
        return result.output