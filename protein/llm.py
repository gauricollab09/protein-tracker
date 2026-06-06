"""One adapter, two providers, two jobs.

    parse_meal(text)  -> Meal      # v2 uses this: items only, NEVER numbers
    ask_total(text)   -> float     # v0 uses this: the naive "just ask the model"

Provider is resolved from config (Claude if a key is set, else local Ollama).
Both calls force structured JSON output and parse defensively.
"""
from __future__ import annotations

import json

from . import config
from .schema import Meal, MEAL_JSON_SCHEMA

PARSE_SYSTEM = (
    "You extract food items from a meal description into structured data. Rules:\n"
    "- Identify each food. Keep meaningful descriptors that change nutrition "
    "(e.g. 'greek yogurt' not 'yogurt', 'whole wheat bread' not 'bread'). "
    "Drop only brands and cooking words (grilled, scrambled).\n"
    "- Copy the quantity and unit EXACTLY as written. Do not convert. "
    "Allowed units: g, oz, ml, cup, tbsp, tsp, slice, scoop, can, bar, bowl, "
    "katori, serving, or 'count' for whole items (eggs, bananas).\n"
    "- 'a'/'an'/'one' = quantity 1. If no amount is given, quantity 1 and a "
    "sensible serving unit.\n"
    "- Split compound dishes into their main protein-bearing ingredients.\n"
    "- NEVER estimate protein or calories. You only identify items and amounts."
)

V0_SYSTEM = (
    "You are a nutrition calculator. Given a meal description, return the TOTAL "
    'protein in grams as JSON: {"total_protein_g": <number>}. Return only JSON.'
)


# --------------------------------------------------------------------------- #
# v2 — parse only
# --------------------------------------------------------------------------- #
def parse_meal(text: str) -> Meal:
    raw = _structured_json(
        system=PARSE_SYSTEM,
        user=f"Meal: {text}",
        json_schema=MEAL_JSON_SCHEMA,
    )
    try:
        return Meal.model_validate(raw)
    except Exception:
        return Meal()  # bad parse -> empty meal, logged by the caller


# --------------------------------------------------------------------------- #
# v0 — naive, ask the model for the number
# --------------------------------------------------------------------------- #
def ask_total(text: str) -> float:
    raw = _structured_json(
        system=V0_SYSTEM,
        user=f"Meal: {text}",
        json_schema={
            "type": "object",
            "properties": {"total_protein_g": {"type": "number"}},
            "required": ["total_protein_g"],
        },
    )
    try:
        return float(raw["total_protein_g"])
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# provider plumbing
# --------------------------------------------------------------------------- #
def _structured_json(system: str, user: str, json_schema: dict) -> dict:
    p = config.provider()
    if p == "claude":
        return _claude_json(system, user, json_schema)
    if p == "gemini":
        return _gemini_json(system, user, json_schema)
    if p == "groq":
        return _groq_json(system, user, json_schema)
    return _ollama_json(system, user, json_schema)


def _ollama_json(system: str, user: str, json_schema: dict) -> dict:
    import ollama

    resp = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        format=json_schema,  # Ollama supports JSON-schema-constrained output
        options={"temperature": 0},
    )
    return json.loads(resp["message"]["content"])


def _groq_json(system: str, user: str, json_schema: dict) -> dict:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": f"{system}\n\nRespond with valid JSON matching this schema: {json.dumps(json_schema)}"},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def _resolve_refs(schema: dict) -> dict:
    """Inline $defs so Gemini can handle Pydantic-generated schemas."""
    import copy
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return resolve(defs[ref_name])
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(i) for i in obj]
        return obj

    return resolve(schema)


def _gemini_json(system: str, user: str, json_schema: dict) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=_resolve_refs(json_schema),
            temperature=0,
        ),
    )
    return json.loads(resp.text)


def _claude_json(system: str, user: str, json_schema: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    tool = {
        "name": "emit",
        "description": "Return the structured result.",
        "input_schema": json_schema,
    }
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        temperature=0,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": "emit"},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    return {}
