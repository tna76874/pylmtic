import requests
import difflib
from typing import List, Type
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


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


# --- PyLMTic Hauptklasse ---
class PyLMTic:
    """
    Wrapper für lokale LLM-Anfragen via Ollama, integriert pydantic_ai Agenten.
    """

    def __init__(self, model_name: str = "qwen", host_url: str = "http://localhost:1234/v1"):
        self.host_url = host_url
        self.model_name = model_name
        self.models: List[OllamaModelInfo] = []
        self.selected_model: OllamaModelInfo | None = None
        self.model: OpenAIChatModel | None = None
        self.agent: Agent | None = None

        self._initialize()

    def _initialize(self):
        # Modelle vom Host abrufen
        response = requests.get(f"{self.host_url}/models")
        response.raise_for_status()
        models_json = response.json()
        self.models = OllamaModelList(**models_json).data

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
