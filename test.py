#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests for usage
"""
from pylmtic import PyLMTic
from pydantic import BaseModel


# --- Beispiel-Pydantic-Klasse für AI-Output ---
class CityLocation(BaseModel):
    city: str
    country: str

lm = PyLMTic(model_name="qwen", host_url="http://localhost:1234/v1")
prompt = "Where were the Olympics held in 2012 and in 2018?"
result = lm.run_prompt(prompt, output_type=CityLocation)

print(result)