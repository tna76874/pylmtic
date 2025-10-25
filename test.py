#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests for usage
"""
import pandas as pd
from pylmtic import PyLMTic
from pydantic import BaseModel


# --- Beispiel-Pydantic-Klasse für AI-Output ---
class CityLocation(BaseModel):
    city: str
    country: str
    year: int

lm = PyLMTic(model_name="qwen", host_url="http://localhost:1234/v1")
prompt = "Where were the Olympics held in 2012 and in 2018?"
result = lm.run_prompt(prompt, output_type=CityLocation)

print("Here we go: there are two answers, so we will get two pydantic objects")
print(result)

print("\nWe can convert them to a pandas dataframe")
df = pd.DataFrame([r.model_dump() for r in result])
print(df)

print("\nAnd filtering for events older than 2015")

print(df[df['year']>= 2015])