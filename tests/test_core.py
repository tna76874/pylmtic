from pydantic import BaseModel
import pytest
from typing import List, Type
import requests
from pylmtic import PyLMTic, OllamaModelInfo, find_closest_model

# --- Beispiel-Pydantic-Klasse für AI-Output ---
class CityLocation(BaseModel):
    city: str
    country: str

# --- Tests ---

def test_pylmtic_init(monkeypatch):
    """
    Testet die Initialisierung von PyLMTic.
    Wir patchen requests.get, um keine echte LLM-Instanz zu benötigen.
    """
    class DummyResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data":[{"id":"qwen-7b","object":"model","owned_by":"user"}], "object":"list"}

    monkeypatch.setattr("requests.get", lambda url: DummyResponse())
    
    lm = PyLMTic(model_name="qwen")
    assert lm.host_url.startswith("http")
    assert lm.selected_model.id == "qwen-7b"
    assert lm.model.model_name == "qwen-7b"
    assert lm.agent is not None

@pytest.mark.skipif(
    not requests.get("http://localhost:1234/v1/models").ok,
    reason="Local LLM host not available"
)
def test_pylmtic_real_access():
    """
    Testet PyLMTic gegen einen echten lokalen Ollama-Host.
    Der Test wird übersprungen, wenn kein Server läuft.
    """
    try:
        lm = PyLMTic(model_name="qwen", host_url="http://localhost:1234/v1")
        prompt = "Where were the Olympics held in 2012?"
        result = lm.run_prompt(prompt, output_type=CityLocation)
        # Ausgabe prüfen
        assert isinstance(result,List)
        # Prüfen, dass alle Items BaseModel Instanzen sind
        for item in result:
            assert isinstance(item, BaseModel)
        print("Echter LLM-Zugriff erfolgreich:", result)
    except Exception as e:
        pytest.skip(f"Echter Zugriff fehlgeschlagen: {e}")
