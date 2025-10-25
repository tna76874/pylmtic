#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests for usage
"""
import pandas as pd
from pylmtic import PyLMTic
from pydantic import BaseModel, Field
import nest_asyncio
nest_asyncio.apply()


# --- Beispiel-Pydantic-Klasse für AI-Output ---
class CityLocation(BaseModel):
    city: str = Field(..., description="Name of the city")
    country: str = Field(..., description="Name of the country where the city is located")
    year: int = Field(..., description="Year of the data collection or observation")

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

class BullShitDetector(BaseModel):
    bullshit_level: float = Field(
        ...,
        ge=0,
        le=1,
        description="The level of bullshit of this item, between 0 and 1"
    )
    
lm = PyLMTic(model_name="qwen", host_url="http://localhost:1234/v1")

print("Or just detect the level of meaningless:")
print(lm.run_prompt("A Duck between moonlight ist greater", output_type=BullShitDetector))
print(lm.run_prompt("Water is wet", output_type=BullShitDetector))