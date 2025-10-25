#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 16:16:43 2025

@author: lukas
"""
from pylmtic import PyLMTic
from pydantic import BaseModel


# --- Beispiel-Pydantic-Klasse für AI-Output ---
class CityLocation(BaseModel):
    city: str
    country: str


lm = PyLMTic(model_name="qwen", host_url="http://localhost:1234/v1")
prompt = "Where were the Olympics held in 2012?"
result = lm.run_prompt(prompt, output_type=CityLocation)