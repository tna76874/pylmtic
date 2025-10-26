from pydantic import BaseModel, Field
import pytest
from typing import List
from pylmtic import PyLMtic, OllamaModelInfo, find_closest_model, LMEndpoint


# --- Beispiel-Pydantic-Klasse für AI-Output ---
class CityLocation(BaseModel):
    city: str
    country: str


# --- Dummy-Response für requests.get ---
class DummyResponse:
    def raise_for_status(self):
        pass
    def json(self):
        return {"data":[{"id":"qwen-7b","object":"model","owned_by":"user"},
                         {"id":"gpt-test","object":"model","owned_by":"user"}],
                "object":"list"}


# --- Tests ---

def test_find_closest_model_exact_match():
    models = [OllamaModelInfo(id="a", object="model", owned_by="u"),
              OllamaModelInfo(id="b", object="model", owned_by="u")]
    result = find_closest_model(models, "b")
    assert result.id == "b"


def test_find_closest_model_fallback(monkeypatch):
    models = [OllamaModelInfo(id="a", object="model", owned_by="u")]
    result = find_closest_model(models, "nonexistent")
    assert result.id == "a"


def test_lm_endpoint_get_url():
    ep = LMEndpoint(name="Test", protocol="https", host="localhost", port=8080, api_path="/v2")
    url = ep.get_url()
    assert url == "https://localhost:8080/v2"


def test_pylmtic_init(monkeypatch):
    """
    Testet die Initialisierung von PyLMtic.
    Wir patchen requests.get, um keine echte LLM-Instanz zu benötigen.
    """
    monkeypatch.setattr("requests.get", lambda url, timeout=None: DummyResponse())
    
    lm = PyLMtic(model_name="qwen")
    assert lm.host_url.startswith("http")
    assert lm.selected_model.id == "qwen-7b"
    assert lm.model.model_name == "qwen-7b"
    assert lm.agent is not None


def test_run_prompt_structure(monkeypatch):
    """
    Testet die run_prompt Methode mit Dummy-Agent.
    """
    class DummyModel(BaseModel):
        city: str = Field(..., description="Name of the city")
        country: str = Field(..., description="Name of the country where the city is located")
        year: int = Field(..., description="Year of the data collection or observation")

    class DummyAgent:
        def __init__(self, model, output_type=None):
            self.model = model
        def run_sync(self, prompt):
            class Result:
                output = [DummyModel(**{"city": "London", "country": "UK", "year": "2024"})]
            return Result()
    
    monkeypatch.setattr("pylmtic.core.Agent", DummyAgent)
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: DummyResponse())
    
    lm = PyLMtic(model_name="qwen")
    result = lm.run_prompt("Test", output_type=DummyModel)
    assert isinstance(result, list)
    assert result[0].city == "London"
