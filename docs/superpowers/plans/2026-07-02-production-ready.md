# Production-Ready Protein Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the working v0/v2 protein-tracker demo into a deployable, tested, observable FastAPI service (Docker, both Claude-cloud and Ollama-local providers) plus an interview-prep document.

**Architecture:** Keep the existing pipeline semantics untouched (v0 naive / v2 parse→ground→compute, fuzzy matcher default). Wrap it in: typed errors + retries instead of silent failures, stdlib logging, thread-safe caches, a thin FastAPI layer with exception mapping, and Docker/CI/packaging around it.

**Tech Stack:** Python 3.12, uv, Pydantic v2, FastAPI + uvicorn, pytest, ruff, GitHub Actions, Docker + compose. Providers: Anthropic SDK (Claude) and ollama client.

## Global Constraints

- Python `>=3.12`; venv is uv-managed (`.venv`), never Anaconda/native (user rule).
- Do NOT change eval semantics, testset files, verified DB values, or matcher defaults (`fuzzy`, threshold 80).
- Library code (`protein/`) never `print()`s — logging only. Scripts/CLI may print.
- A returned `0.0` must mean "the model said 0", never "a call failed". Failures raise.
- CI must pass with no network and no LLM installed: mock `httpx.get` and the parser.
- All commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Working dir: `/Users/gauritulsulkar/AI_proj/protein-tracker`. Run everything via `uv run …`.

---

### Task 0: Commit the outstanding matcher bake-off work

**Files:** none created — commits existing modified/untracked files
(`protein/matchers.py`, `protein/foods.py`, `protein/config.py`, `eval_match.py`,
`data/match_testset.jsonl`, `match_results.txt`, `eval_results.txt`, `pyproject.toml`, `uv.lock`).

- [ ] **Step 1: Verify the tree state and commit**

```bash
git add -A
git commit -m "Add 3-way food-name matcher bake-off (substring/bm25/fuzzy)

fuzzy (rapidfuzz WRatio, threshold 80) wins the measured bake-off 90% vs
bm25 71% vs substring 52% on 21 messy-name cases and becomes the default
matcher via PROTEIN_MATCHER.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: clean `git status` afterwards.

---

### Task 1: Typed errors + input validation

**Files:**
- Create: `protein/errors.py`
- Modify: `protein/schema.py` (add `validate_meal_text`)
- Create: `tests/__init__.py` (empty), `tests/test_validation.py`
- Modify: `pyproject.toml` (add `[project.optional-dependencies] dev` with pytest/ruff; needed to run tests at all)

**Interfaces:**
- Produces: `protein.errors.ProteinError`, `InputError(ProteinError)`, `LLMError(ProteinError)`; `protein.schema.validate_meal_text(text: str) -> str` (returns stripped text or raises `InputError`); `protein.schema.MAX_MEAL_CHARS = 2000`.

- [ ] **Step 1: Add dev deps so pytest exists**

Append to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
]
```

Run: `uv pip install -e ".[dev]"` — expected: installs pytest + ruff.

- [ ] **Step 2: Write the failing test**

`tests/test_validation.py`:

```python
import pytest

from protein.errors import InputError
from protein.schema import MAX_MEAL_CHARS, validate_meal_text


def test_valid_text_is_stripped():
    assert validate_meal_text("  2 eggs  ") == "2 eggs"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_text_rejected(bad):
    with pytest.raises(InputError):
        validate_meal_text(bad)


def test_oversized_text_rejected():
    with pytest.raises(InputError):
        validate_meal_text("x" * (MAX_MEAL_CHARS + 1))


def test_max_length_boundary_ok():
    assert validate_meal_text("x" * MAX_MEAL_CHARS) == "x" * MAX_MEAL_CHARS
```

- [ ] **Step 3: Run to verify it fails** — `uv run pytest tests/test_validation.py -q` → FAIL (ModuleNotFoundError: protein.errors).

- [ ] **Step 4: Implement**

`protein/errors.py`:

```python
"""Typed failures. A 0.0 result must mean "the model said 0" — anything that
breaks raises one of these instead of being swallowed."""
from __future__ import annotations


class ProteinError(Exception):
    """Base for all protein-tracker errors."""


class InputError(ProteinError):
    """The caller's meal text is unusable (empty, too long)."""


class LLMError(ProteinError):
    """The LLM provider call failed after retries, or returned garbage."""
```

Add to `protein/schema.py` (after the imports):

```python
from .errors import InputError

MAX_MEAL_CHARS = 2000


def validate_meal_text(text: str) -> str:
    """Boundary check for every public entry point (v0, v2, API)."""
    if text is None or not text.strip():
        raise InputError("meal text is empty")
    if len(text) > MAX_MEAL_CHARS:
        raise InputError(f"meal text too long ({len(text)} > {MAX_MEAL_CHARS} chars)")
    return text.strip()
```

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/test_validation.py -q` → 4 passed.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: typed errors + meal-text validation ..."`

---

### Task 2: Harden llm.py — retries, timeouts, no silent failures, logging

**Files:**
- Modify: `protein/config.py` (add `OLLAMA_HOST`, `LLM_TIMEOUT_S`, `LLM_RETRIES`)
- Modify: `protein/llm.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Consumes: `LLMError` from Task 1.
- Produces: `parse_meal(text) -> Meal` and `ask_total(text) -> float` now RAISE `LLMError` on provider failure/garbage (no more silent `Meal()` / `0.0`). Internal seam for tests: `llm._call_provider(system, user, json_schema) -> dict` (single un-retried provider call; `_structured_json` wraps it with retry).
- Config produces: `config.OLLAMA_HOST` (default `http://localhost:11434`), `config.LLM_TIMEOUT_S` (default 60.0), `config.LLM_RETRIES` (default 2).

- [ ] **Step 1: Write the failing tests**

`tests/test_llm.py`:

```python
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
```

Note: `llm._sleep = time.sleep` module attribute; retry loop calls `llm._sleep(...)`. Autouse fixture in `tests/conftest.py` patches it to a no-op:

`tests/conftest.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    from protein import llm

    monkeypatch.setattr(llm, "_sleep", lambda s: None)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_llm.py -q` → FAIL (`_call_provider` missing).

- [ ] **Step 3: Implement**

Add to `protein/config.py` under the provider section:

```python
# Where the Ollama server lives (in Docker: http://host.docker.internal:11434
# for the host's Ollama, or http://ollama:11434 with the compose 'local' profile).
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# One provider call may take at most this long; we retry transient failures.
LLM_TIMEOUT_S = float(os.getenv("PROTEIN_LLM_TIMEOUT_S", "60"))
LLM_RETRIES = int(os.getenv("PROTEIN_LLM_RETRIES", "2"))
```

Rewrite the plumbing part of `protein/llm.py` (keep PARSE_SYSTEM/V0_SYSTEM as-is):

```python
from __future__ import annotations

import json
import logging
import time

from . import config
from .errors import LLMError
from .schema import Meal, MEAL_JSON_SCHEMA

log = logging.getLogger("protein.llm")
_sleep = time.sleep  # module attr so tests can neutralize backoff


def parse_meal(text: str) -> Meal:
    raw = _structured_json(PARSE_SYSTEM, f"Meal: {text}", MEAL_JSON_SCHEMA)
    try:
        meal = Meal.model_validate(raw)
    except Exception as exc:
        raise LLMError(f"model returned data that does not fit the Meal schema: {exc}") from exc
    log.info("parsed %d item(s) from %r", len(meal.items), text)
    return meal


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
```

(`max_retries=0` because we own retry policy; docstring at top of file stays.)

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_llm.py tests/test_validation.py -q` → all pass.

- [ ] **Step 5: Commit.**

---

### Task 3: Harden foods.py — thread-safe caches, atomic writes, logging + unit tests for grounding

**Files:**
- Modify: `protein/foods.py`, `protein/config.py` (make DATA overridable)
- Create: `tests/test_foods.py`, `tests/test_matchers.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: same public API (`to_grams`, `lookup`, `ground_item`, `_load_verified`) — now safe under concurrent calls. `config.DATA` honors `PROTEIN_DATA_DIR` env (needed by Docker in Task 6).

- [ ] **Step 1: Make DATA overridable in `protein/config.py`:**

```python
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("PROTEIN_DATA_DIR", ROOT / "data"))
```

- [ ] **Step 2: Write the failing tests**

`tests/test_foods.py`:

```python
import json

import httpx
import pytest

from protein import config, foods
from protein.schema import FoodItem


@pytest.fixture(autouse=True)
def _isolated_off_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OFF_CACHE", tmp_path / "off_cache.json")
    monkeypatch.setattr(foods, "_cache", None)


# ---- to_grams ----
@pytest.mark.parametrize(
    "item,expected",
    [
        (FoodItem(name="chicken breast", quantity=150, unit="g"), 150.0),
        (FoodItem(name="salmon", quantity=2, unit="oz"), 56.7),
        (FoodItem(name="milk", quantity=250, unit="ml"), 250.0),
        (FoodItem(name="rice", quantity=1, unit="cup"), 200.0),
        (FoodItem(name="peanut butter", quantity=2, unit="tbsp"), 30.0),
        (FoodItem(name="rajma", quantity=1, unit="katori"), 150.0),
        (FoodItem(name="tuna", quantity=1, unit="can"), 140.0),
        (FoodItem(name="bread", quantity=2, unit="slice"), 56.0),
        (FoodItem(name="whey protein", quantity=1, unit="scoop"), 30.0),
        (FoodItem(name="egg", quantity=2, unit="count"), 100.0),
        (FoodItem(name="banana", quantity=1, unit=""), 118.0),
        (FoodItem(name="mystery", quantity=3, unit="count"), 300.0),
        (FoodItem(name="steak", quantity=200, unit="furlongs"), 200.0),
    ],
)
def test_to_grams(item, expected):
    assert foods.to_grams(item) == pytest.approx(expected, abs=0.01)


# ---- verified lookup via matcher ----
def test_verified_hit_fuzzy(monkeypatch):
    monkeypatch.setattr(config, "MATCHER", "fuzzy")
    val, source = foods.lookup("grilled chicken breast")
    assert source == "verified" and val == pytest.approx(31.0)


def test_out_of_db_falls_through(monkeypatch):
    monkeypatch.setattr(config, "MATCHER", "fuzzy")
    monkeypatch.setattr(foods, "_off_lookup", lambda name: None)
    val, source = foods.lookup("paneer tikka")
    assert val is None and source == "none"


# ---- OFF fallback (mocked httpx) + cache ----
def _fake_off_response(*a, **kw):
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"products": [{"product_name": "x", "nutriments": {"proteins_100g": 19.5}}]}

    return R()


def test_off_fallback_and_cache(monkeypatch):
    monkeypatch.setattr(config, "MATCHER", "fuzzy")
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return _fake_off_response()

    monkeypatch.setattr(httpx, "get", counting)
    val, source = foods.lookup("weird packaged thing")
    assert (val, source) == (19.5, "off")
    foods.lookup("weird packaged thing")  # second call: served from cache
    assert calls["n"] == 1
    assert json.loads(config.OFF_CACHE.read_text())["weird packaged thing"] == 19.5


def test_ground_item_no_hit(monkeypatch):
    monkeypatch.setattr(config, "MATCHER", "fuzzy")
    monkeypatch.setattr(foods, "_off_lookup", lambda name: None)
    g = foods.ground_item(FoodItem(name="paneer tikka", quantity=1, unit="serving"))
    assert g.no_hit and g.protein_g == 0.0 and g.source == "none"
```

`tests/test_matchers.py`:

```python
import pytest

from protein import matchers

KEYS = ["chicken breast", "greek yogurt", "yogurt", "peanut butter", "cheddar"]


@pytest.mark.parametrize(
    "fn_name,name,expected",
    [
        ("substring", "grilled chicken breast", "chicken breast"),
        ("substring", "chicken", None),                      # brittle: not a substring hit
        ("bm25", "breast of chicken", "chicken breast"),      # word order
        ("bm25", "chedder cheese", None),                     # typo: token miss
        ("fuzzy", "chicken", "chicken breast"),               # partial
        ("fuzzy", "yoghurt", "yogurt"),                       # spelling
        ("fuzzy", "chedder cheese", "cheddar"),               # typo
        ("fuzzy", "quinoa salad", None),                      # out of db -> no hit
    ],
)
def test_matchers(fn_name, name, expected):
    fn = matchers.MATCHERS[fn_name]
    got = fn(name, KEYS, threshold=80.0) if fn_name == "fuzzy" else fn(name, KEYS)
    assert got == expected


def test_registry_fallback():
    assert matchers.get("nonsense") is matchers.substring_match
```

- [ ] **Step 3: Run — expect mostly green already** (`uv run pytest tests/ -q`). Any failure = real bug; fix before proceeding. (`test_off_fallback_and_cache` may pass already; the concurrency changes below must not break it.)

- [ ] **Step 4: Add locks + atomic writes + logging to `protein/foods.py`**

```python
import logging
import os
import tempfile
import threading

log = logging.getLogger("protein.foods")
_verified_lock = threading.Lock()
_cache_lock = threading.Lock()


def _load_verified() -> dict[str, float]:
    global _verified
    with _verified_lock:
        if _verified is None:
            raw = json.loads((config.DATA / "verified_foods.json").read_text())
            _verified = {k.lower(): float(v) for k, v in raw.items() if not k.startswith("_")}
            log.info("loaded %d verified foods", len(_verified))
        return _verified


def _load_cache() -> dict[str, float | None]:
    global _cache
    if _cache is None:
        if config.OFF_CACHE.exists():
            _cache = json.loads(config.OFF_CACHE.read_text())
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    config.OFF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=config.OFF_CACHE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_cache, f, indent=2, sort_keys=True)
        os.replace(tmp, config.OFF_CACHE)  # atomic: readers never see a torn file
    except BaseException:
        os.unlink(tmp)
        raise


def lookup(name: str) -> tuple[float | None, str]:
    val = _verified_lookup(name)
    if val is not None:
        log.debug("%r grounded from verified DB: %.1f g/100g", name, val)
        return val, "verified"
    key = name.strip().lower()
    with _cache_lock:
        cache = _load_cache()
        if key in cache:
            cached = cache[key]
            log.debug("%r served from OFF cache", name)
            return (cached, "off") if cached is not None else (None, "none")
    value = _off_lookup(key)  # network call outside the lock
    with _cache_lock:
        _load_cache()[key] = value
        _save_cache()
    if value is None:
        log.info("%r not found anywhere -> no_hit", name)
    return (value, "off") if value is not None else (None, "none")
```

(Keep `_verified_lookup`, `to_grams`, `ground_item`, `_off_lookup` bodies as they are.)

- [ ] **Step 5: Run all tests** — `uv run pytest tests/ -q` → all pass.

- [ ] **Step 6: Commit.**

---

### Task 4: Wire validation + errors into v0/v2 and cover the pipeline

**Files:**
- Modify: `protein/v2.py`, `protein/v0.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `validate_meal_text`, `LLMError`, `InputError`.
- Produces: `v2.analyze(text)` raises `InputError` on bad input, `LLMError` propagates. `MealResult` gains `provider: str` field (used by API in Task 5).

- [ ] **Step 1: Failing tests**

`tests/test_pipeline.py`:

```python
import pytest

from protein import v2
from protein.errors import InputError
from protein.schema import FoodItem, Meal


def test_analyze_totals_and_no_hits(monkeypatch):
    meal = Meal(
        items=[
            FoodItem(name="chicken breast", quantity=150, unit="g"),   # 46.5 g
            FoodItem(name="egg", quantity=2, unit="count"),            # 12.6 g
            FoodItem(name="paneer tikka", quantity=1, unit="serving"), # no hit
        ]
    )
    monkeypatch.setattr(v2.llm, "parse_meal", lambda text: meal)
    monkeypatch.setattr(v2.foods, "_off_lookup", lambda name: None)
    res = v2.analyze("150g chicken breast, 2 eggs, paneer tikka")
    assert res.total_protein_g == pytest.approx(59.1, abs=0.1)
    assert res.no_hits == ["paneer tikka"]
    assert res.provider in ("claude", "ollama")


def test_analyze_rejects_empty():
    with pytest.raises(InputError):
        v2.analyze("   ")
```

- [ ] **Step 2: Run to verify fail** (provider attr missing / no InputError).

- [ ] **Step 3: Implement** — in `v2.py`: add `provider: str = ""` field to `MealResult`; `analyze` starts with `meal_text = validate_meal_text(meal_text)` and sets `result.provider = config.provider()`. In `v0.py`: `total_protein` starts with `meal_text = validate_meal_text(meal_text)`. Imports: `from .schema import validate_meal_text`.

- [ ] **Step 4: Run all tests green.**

- [ ] **Step 5: Commit.**

---

### Task 5: FastAPI service

**Files:**
- Create: `protein/api.py`
- Create: `tests/test_api.py`
- Modify: `pyproject.toml` (deps: `fastapi>=0.115`, `uvicorn>=0.30`; scripts: `protein-api = "protein.api:main"`; dev extra gains `httpx` is already a main dep — TestClient needs it, fine)

**Interfaces:**
- Consumes: `v2.analyze`, `v0.total_protein`, `foods._load_verified`, errors.
- Produces: ASGI app `protein.api:app`; endpoints `GET /health`, `POST /analyze`, `POST /v0/estimate`; `main()` runs uvicorn on `0.0.0.0:8000` (port via `PORT` env).

- [ ] **Step 1: Failing tests**

`tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

from protein import api, v2
from protein.errors import LLMError
from protein.schema import FoodItem, Meal

client = TestClient(app=api.app, raise_server_exceptions=False)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["verified_foods"] > 0
    assert body["provider"] in ("claude", "ollama")


def test_analyze_ok(monkeypatch):
    meal = Meal(items=[FoodItem(name="chicken breast", quantity=150, unit="g")])
    monkeypatch.setattr(v2.llm, "parse_meal", lambda text: meal)
    r = client.post("/analyze", json={"text": "150g grilled chicken breast"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_protein_g"] == pytest.approx(46.5)
    assert body["items"][0]["source"] == "verified"
    assert body["meets_goal"] is False


def test_analyze_empty_is_422():
    r = client.post("/analyze", json={"text": "   "})
    assert r.status_code == 422


def test_analyze_llm_down_is_503(monkeypatch):
    def boom(text):
        raise LLMError("provider down")

    monkeypatch.setattr(v2.llm, "parse_meal", boom)
    r = client.post("/analyze", json={"text": "2 eggs"})
    assert r.status_code == 503
    assert "provider down" in r.json()["detail"]


def test_v0_estimate(monkeypatch):
    from protein import v0

    monkeypatch.setattr(v0.llm, "ask_total", lambda text: 42.0)
    r = client.post("/v0/estimate", json={"text": "2 eggs"})
    assert r.status_code == 200
    assert r.json()["total_protein_g"] == 42.0
```

- [ ] **Step 2: Add deps** (`fastapi`, `uvicorn` to `[project].dependencies`), `uv pip install -e ".[dev]"`, run tests → FAIL (no `protein.api`).

- [ ] **Step 3: Implement `protein/api.py`:**

```python
"""HTTP surface. Thin: validation/errors live in the library, this maps them.

    uv run protein-api            # or: uvicorn protein.api:app
    POST /analyze      {"text": "150g chicken and 2 eggs"}
    POST /v0/estimate  {"text": "..."}   (naive baseline, for the live A/B)
    GET  /health
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config, foods, v0, v2
from .errors import InputError, LLMError

log = logging.getLogger("protein.api")

app = FastAPI(
    title="Protein Tracker",
    version="1.0.0",
    description="The LLM is a parser, not a nutrition oracle: items are parsed "
    "by an LLM, protein numbers come from a verified DB, totals are computed.",
)


class AnalyzeRequest(BaseModel):
    text: str = Field(examples=["150g grilled chicken breast and 2 eggs"])


class ItemOut(BaseModel):
    name: str
    grams: float
    protein_per_100g: float | None
    protein_g: float
    source: str
    no_hit: bool


class AnalyzeResponse(BaseModel):
    text: str
    provider: str
    items: list[ItemOut]
    total_protein_g: float
    goal_g: float
    meets_goal: bool
    no_hits: list[str] = Field(description="foods not found in any source; flagged, never guessed")


class EstimateResponse(BaseModel):
    text: str
    provider: str
    total_protein_g: float


@app.exception_handler(InputError)
async def _input_error(request: Request, exc: InputError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def _llm_error(request: Request, exc: LLMError):
    return JSONResponse(status_code=503, content={"detail": f"LLM provider failed: {exc}"})


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s -> %d in %.0f ms", request.method, request.url.path,
             response.status_code, 1000 * (time.perf_counter() - t0))
    return response


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": config.provider(),
        "matcher": config.MATCHER,
        "verified_foods": len(foods._load_verified()),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    res = v2.analyze(req.text)
    return AnalyzeResponse(
        text=res.text,
        provider=res.provider,
        items=[ItemOut(**vars(i)) for i in res.items],
        total_protein_g=res.total_protein_g,
        goal_g=config.PROTEIN_GOAL_G,
        meets_goal=res.meets_goal,
        no_hits=res.no_hits,
    )


@app.post("/v0/estimate", response_model=EstimateResponse)
def estimate(req: AnalyzeRequest) -> EstimateResponse:
    return EstimateResponse(
        text=req.text, provider=config.provider(), total_protein_g=v0.total_protein(req.text)
    )


def main() -> None:
    import os

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
```

- [ ] **Step 4: Tests green** — `uv run pytest tests/ -q`.

- [ ] **Step 5: Smoke it** — `uv run protein-api &`, `curl -s localhost:8000/health`, kill. Expected: `{"status":"ok",...}`.

- [ ] **Step 6: Commit.**

---

### Task 6: Docker (both providers) + compose

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.yml`
- Modify: `.env.example` (document Docker env)

**Interfaces:**
- Consumes: `PROTEIN_DATA_DIR` from Task 3, `protein.api:app`, `OLLAMA_HOST` from Task 2.

- [ ] **Step 1: `Dockerfile`:**

```dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROTEIN_DATA_DIR=/app/data

RUN useradd --create-home appuser
WORKDIR /app

COPY pyproject.toml README.md ./
COPY protein ./protein
RUN pip install --no-cache-dir .

COPY data ./data
RUN chown -R appuser:appuser /app/data

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=4).raise_for_status()"

CMD ["uvicorn", "protein.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: `.dockerignore`:**

```
.venv
.git
__pycache__
*.pyc
.env
data/off_cache.json
docs
tests
*.txt
.github
```

- [ ] **Step 3: `docker-compose.yml`:**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      # cloud mode: set ANTHROPIC_API_KEY in your shell or .env
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
      # local mode (default): the Mac's Ollama; with --profile local use http://ollama:11434
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
      PROTEIN_LLM: ${PROTEIN_LLM:-auto}
      OLLAMA_MODEL: ${OLLAMA_MODEL:-llama3.2}
    extra_hosts:
      - "host.docker.internal:host-gateway"

  ollama:
    profiles: ["local"]
    image: ollama/ollama:latest
    volumes:
      - ollama-models:/root/.ollama
    ports:
      - "11434:11434"

volumes:
  ollama-models:
```

- [ ] **Step 4: Build & smoke** — `docker build -t protein-tracker .` then `docker run --rm -p 8000:8000 protein-tracker` + `curl localhost:8000/health`. Expected 200. (If Docker daemon isn't running locally, note it and rely on the CI build check.)

- [ ] **Step 5: Update `.env.example`** — add `OLLAMA_HOST=`, `PROTEIN_LLM_TIMEOUT_S=60`, `PROTEIN_MATCHER=fuzzy`, `PROTEIN_FUZZY_THRESHOLD=80` with comments.

- [ ] **Step 6: Commit.**

---

### Task 7: Eval entry points + ruff + CI

**Files:**
- Create: `protein/eval_pipeline.py` (moved from `eval.py`), `protein/eval_matchers.py` (moved from `eval_match.py`)
- Modify: `eval.py`, `eval_match.py` → 3-line shims (`from protein.eval_pipeline import main; main()`) so `uv run eval.py` still works
- Modify: `pyproject.toml`: scripts `protein-eval = "protein.eval_pipeline:main"`, `protein-eval-match = "protein.eval_matchers:main"`; `[tool.ruff]` (line-length 100, target py312), `[tool.pytest.ini_options]` (`testpaths = ["tests"]`); explicit `[build-system]` (setuptools) + `[tool.setuptools] packages = ["protein"]`; bump version to `1.0.0`.
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `protein.eval_pipeline.main()`, `protein.eval_matchers.main()` console scripts.

- [ ] **Step 1: Move eval bodies into the package** (`git mv` not needed — create new modules with identical content, imports unchanged since they already `from protein import …`; replace root files with shims:)

```python
"""Shim kept so `uv run eval.py` still works; real code: protein/eval_pipeline.py"""
from protein.eval_pipeline import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `.github/workflows/ci.yml`:**

```yaml
name: ci
on:
  push: {branches: [main]}
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv venv --python 3.12 && uv pip install -e ".[dev]"
      - run: uv run ruff check .
      - run: uv run pytest -q

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t protein-tracker .
```

- [ ] **Step 3: Run `uv run ruff check .` and fix every finding** (expected: unused imports, maybe line lengths). Re-run until clean. Run `uv run pytest -q` — green. Verify `uv run protein-eval-match` produces the bake-off table (no LLM needed).

- [ ] **Step 4: Commit.**

---

### Task 8: README rewrite + INTERVIEW.md

**Files:**
- Modify: `README.md` — keep the story sections; add: **API** section (uvicorn + curl examples for /analyze, /v0/estimate, /health), **Docker** section (3 modes: host-Ollama default, cloud via ANTHROPIC_API_KEY, fully-local via `docker compose --profile local up` + `docker compose exec ollama ollama pull llama3.2`), **Tests & CI** section (`uv run pytest`, ruff, badge-free), **Limitations** section (no auth/rate limiting/persistence; portion estimation is the dominant residual error; verified DB is small by design).
- Create: `INTERVIEW.md` with these sections (write full prose, use the measured numbers from `eval_results.txt` / `match_results.txt`):
  1. **The 60-second story** — journey: vibe-coded v0 → noticed confidently-wrong numbers → key insight "the LLM is a parser, not a nutrition oracle" → v2 parse→ground→compute → eval harness proves it (naive MAE ~14g/46% → grounded 1.8–3.1g/89%).
  2. **Architecture in one breath** + the request-flow diagram (text).
  3. **Decisions I can defend** — table: curated-DB-first over OFF (measured noise: salmon-flavored snack 7.8g), fuzzy matcher chosen by bake-off (90% vs 71% vs 52%), abstention over guessing (no_hit), retries+typed errors ("0 means 0"), both-provider design (cost/privacy tradeoff), component evals over end-to-end only (the compensating-errors story: fixing the chicken match RAISED end-to-end MAE by unmasking a portion error — why you eval components).
  4. **Production-readiness work** — what changed and why (errors/retries/timeouts, logging, thread-safety+atomic writes, FastAPI+error mapping, Docker both-provider, tests-without-network, CI).
  5. **Likely interview questions, strong answers** — ~10 Q&As (why not fine-tune; why not RAG/embeddings; how do you know it works; what breaks first at scale; how would you add auth/multi-user; hallucination handling; why MAE; what would you do with 2 more weeks).
  6. **Resume bullets** — 3-4, quantified.
  7. **Numbers to memorize** — small table.

- [ ] **Step 1: Write both docs.**
- [ ] **Step 2: `uv run pytest -q` still green; `uv run ruff check .` clean.**
- [ ] **Step 3: Commit.**

---

### Task 9: Final verification + push

- [ ] **Step 1: Full local gate** — `uv run ruff check . && uv run pytest -q` → clean/green.
- [ ] **Step 2: Live smoke (Ollama)** — `uv run python -m protein.v2 "150g grilled chicken breast and 2 eggs"` → itemized output, total ≈ 59 g. And `uv run protein-api` + one real `curl /analyze` if Ollama is up.
- [ ] **Step 3: Docker build if daemon available** — `docker build -t protein-tracker .`.
- [ ] **Step 4: Push** — `git push origin main`. Confirm CI triggers on GitHub (`gh run list --limit 1`).
- [ ] **Step 5: Report the final state to the user.**
