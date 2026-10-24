"""One adapter, two providers, two jobs.

    parse_meal(text)  -> Meal      # v2 uses this: items only, NEVER numbers
    ask_total(text)   -> float     # v0 uses this: the naive "just ask the model"

Provider is resolved from config (Claude if a key is set, else local Ollama).
Both calls force structured JSON output and parse defensively.
"""
from __future__ import annotations

import json
import logging
import time

from . import config
from .errors import LLMError
from .schema import Meal, MEAL_JSON_SCHEMA

log = logging.getLogger("protein.llm")
_sleep = time.sleep  # module attr so tests can neutralize backoff

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
    raw = _structured_json(PARSE_SYSTEM, f"Meal: {text}", MEAL_JSON_SCHEMA)
    try:
        meal = Meal.model_validate(raw)
    except Exception as exc:
        raise LLMError(f"model returned data that does not fit the Meal schema: {exc}") from exc
    log.info("parsed %d item(s) from %r", len(meal.items), text)
    return meal


# --------------------------------------------------------------------------- #
# v0 — naive, ask the model for the number
# --------------------------------------------------------------------------- #
def ask_total(text: str) -> float:
    raw = _structured_json(
        V0_SYSTEM,
        f"Meal: {text}",
        {
            "type": "object",
            "properties": {"total_protein_g": {"type": "number"}},
            "required": ["total_protein_g"],
        },
    )
    try:
        return float(raw["total_protein_g"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError(f"model did not return a numeric total: {raw!r}") from exc


# --------------------------------------------------------------------------- #
# provider plumbing
# --------------------------------------------------------------------------- #
def _structured_json(system: str, user: str, json_schema: dict) -> dict:
    prov = config.provider()
    last_exc: Exception | None = None
    for attempt in range(1, config.LLM_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            raw = _call_provider(system, user, json_schema)
            log.info("%s call ok in %.0f ms (attempt %d)", prov, 1000 * (time.perf_counter() - t0), attempt)
            return raw
        except LLMError:
            raise  # schema garbage from a lower layer: not transient, don't retry
        except Exception as exc:  # network / provider errors are transient
            last_exc = exc
            log.warning("%s call failed (attempt %d/%d): %s", prov, attempt, config.LLM_RETRIES, exc)
            if attempt < config.LLM_RETRIES:
                _sleep(0.5 * attempt)
    raise LLMError(f"{prov} call failed after {config.LLM_RETRIES} attempt(s): {last_exc}") from last_exc


def _call_provider(system: str, user: str, json_schema: dict) -> dict:
    if config.provider() == "claude":
        return _claude_json(system, user, json_schema)
    return _ollama_json(system, user, json_schema)


def _ollama_json(system: str, user: str, json_schema: dict) -> dict:
    import ollama

    client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.LLM_TIMEOUT_S)
    resp = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format=json_schema,
        options={"temperature": 0},
    )
    return json.loads(resp["message"]["content"])


def _claude_json(system: str, user: str, json_schema: dict) -> dict:
    import anthropic

    client = anthropic.Anthropic(
        api_key=config.ANTHROPIC_API_KEY, timeout=config.LLM_TIMEOUT_S, max_retries=0
    )
    tool = {"name": "emit", "description": "Return the structured result.", "input_schema": json_schema}
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
    raise LLMError("Claude response contained no tool_use block")
