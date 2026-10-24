import pytest

from protein import config, llm
from protein.errors import LLMError


def test_retry_then_success(monkeypatch):
    calls = {"n": 0}

    def flaky(system, user, json_schema):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("boom")
        return {"items": [{"name": "egg", "quantity": 2, "unit": "count"}]}

    monkeypatch.setattr(llm, "_call_provider", flaky)
    monkeypatch.setattr(config, "LLM_RETRIES", 2)
    meal = llm.parse_meal("2 eggs")
    assert calls["n"] == 2
    assert meal.items[0].name == "egg"


def test_exhausted_retries_raise(monkeypatch):
    def always_fails(system, user, json_schema):
        raise ConnectionError("down")

    monkeypatch.setattr(llm, "_call_provider", always_fails)
    monkeypatch.setattr(config, "LLM_RETRIES", 2)
    with pytest.raises(LLMError):
        llm.parse_meal("2 eggs")


def test_garbage_schema_raises(monkeypatch):
    monkeypatch.setattr(llm, "_call_provider", lambda *a: {"items": "not-a-list"})
    with pytest.raises(LLMError):
        llm.parse_meal("2 eggs")


def test_ask_total_garbage_raises(monkeypatch):
    monkeypatch.setattr(llm, "_call_provider", lambda *a: {"total_protein_g": "many"})
    with pytest.raises(LLMError):
        llm.ask_total("2 eggs")


def test_no_sleep_needed(monkeypatch):
    """Retry backoff must be patchable so tests stay fast."""
    assert hasattr(llm, "_sleep")
