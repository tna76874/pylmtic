# PyLMtic

`pylmtic` is a Python wrapper for local LLMs (LM Studio or Ollama) that integrates `pydantic_ai` agents for structured AI output. It allows you to query models and automatically parse results into Pydantic models for structured data handling.

## Installation

```bash
pip install --upgrade git+https://github.com/tna76874/pylmtic@main
```

------

## Quick Start

You can run the example directly from the `test.py` file included in the repository:

```bash
python test.py
```

This script demonstrates:

- Initializing `PyLMtic` with a model or custom endpoint
- Defining Pydantic models for structured AI output
- Running prompts and converting results to Pydantic objects
- Converting results to a `pandas.DataFrame` and filtering
- Running a nonsense detection example

------

## Usage Notes

- `PyLMtic` automatically selects the closest available model if the exact name is not found.
- Structured output requires Pydantic `BaseModel` types.
- You can filter and manipulate AI outputs using `pandas`.
- Supports multiple endpoints (`LMStudio` and `Ollama`) and auto-fallback.
- You can provide a custom endpoint via `host_url`.

```python
from pylmtic import PyLMtic

# Example with custom endpoint
lm = PyLMtic(model_name="qwen", host_url="http://localhost:1234/v1")
```

