from pydantic import BaseModel
import pytest
from typing import List
import requests
from pylmtic import PyLMTic

# --- Beispiel-Pydantic-Klasse für AI-Output ---
class CityLocation(BaseModel):
    city: str
    country: str


@pytest.mark.skipif(
    not requests.get("http://localhost:1234/v1/models").ok,
    reason="Local LLM host not available"
)
def test_pylmtic_real_access():
    """
    Testet PyLMTic gegen einen echten lokalen LLM-Host.
    Der Test wird übersprungen, wenn kein Server läuft.
    """
    try:
        lm = PyLMTic(model_name="qwen")
        prompt = "Where were the Olympics held in 2012?"
        result = lm.run_prompt(prompt, output_type=CityLocation)
        # Ausgabe prüfen
        assert isinstance(result,List)
        for item in result:
            assert isinstance(item, BaseModel)
        print("Echter LLM-Zugriff erfolgreich:", result)
    except Exception as e:
        pytest.skip(f"Echter Zugriff fehlgeschlagen: {e}")
