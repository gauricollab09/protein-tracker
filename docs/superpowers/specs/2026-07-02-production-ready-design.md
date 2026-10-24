# Production-Ready Protein Tracker — Design

Date: 2026-07-02
Status: approved (user), scope chosen: **deployable service** with **both local
(Ollama) and cloud (Claude) providers** supported in Docker via env config.

## Goal

Take the working demo (v0 naive vs v2 grounded pipeline + eval
harnesses) and make it a deployable, tested, observable service without
changing its measured behavior (eval numbers, matcher bake-off results,
v0/v2 semantics stay as-is).

## 1. Core library hardening (`protein/`)

- **Errors are visible, never silent.** New `protein/errors.py` with
  `LLMError` (provider call failed) and `InputError` (bad meal text).
  `llm.py` stops returning `Meal()` / `0.0` on exceptions; it retries
  transient failures (2 attempts, short backoff) and then raises `LLMError`.
  A returned 0 must mean "the model said 0", never "something broke".
- **Timeouts** on both providers (Claude SDK timeout param; Ollama client
  with a configured host + timeout). New env knobs: `OLLAMA_HOST`,
  `PROTEIN_LLM_TIMEOUT_S`.
- **Input validation** at the public boundary (`v2.analyze`, `v0.total_protein`,
  API): empty/whitespace text and text > 2000 chars → `InputError`.
- **Logging**: stdlib `logging` under the `protein` logger namespace. Library
  code never prints. Key events: provider call + latency, retry, parse result
  size, grounding source per item, no-hits, OFF fallback + cache hits.
- **Concurrency safety** in `foods.py`: module-level caches (`_verified`,
  OFF `_cache`) guarded by locks; OFF cache file written atomically
  (temp file + `os.replace`).
- **Matcher default stays `fuzzy`** (measured bake-off winner, 90%).

## 2. API service (`protein/api.py`, FastAPI)

- `POST /analyze` — body `{"text": "..."}` → grounded result: items with
  grams / per-100g / protein / source / no_hit, total, `meets_goal`,
  no-hit names, provider used. Pydantic response models → self-documenting
  OpenAPI at `/docs`.
- `POST /v0/estimate` — the naive baseline over HTTP (keeps the A/B runnable
  live).
- `GET /health` — verified DB loads + configured provider; no LLM call.
- Error mapping: `InputError` → 422, `LLMError` → 503 with message,
  unexpected → 500. Request logging middleware (method, path, status, ms).
- **Known gaps documented, not built** (YAGNI for a demo service): no auth,
  no rate limiting, no persistence. Named in README under Limitations.

## 3. Docker — both providers, env-selected

- `Dockerfile`: `python:3.12-slim`, uv-installed locked deps, non-root user,
  `HEALTHCHECK` hitting `/health`, uvicorn on 8000.
- `docker-compose.yml`:
  - `api` service (default): reads `.env`; cloud mode via `ANTHROPIC_API_KEY`,
    local mode via `OLLAMA_HOST=http://host.docker.internal:11434` (host Mac's
    Ollama).
  - `--profile local`: adds an `ollama` container + model volume for a fully
    self-contained stack (`docker compose --profile local up`).
- `.dockerignore`.

## 4. Tests + CI

- `tests/` with pytest; **no network and no LLM in CI** (OFF mocked via
  httpx mock/monkeypatch; parser mocked for pipeline tests):
  - `to_grams` unit table (g/oz/ml, volume, container, count-as-unit, unknown).
  - all three matchers incl. no-hit cases.
  - verified lookup + OFF fallback + cache behavior.
  - `v2.analyze` end-to-end with mocked `parse_meal` (totals, no-hit flow).
  - error paths: `LLMError` propagation, input validation.
  - API via `TestClient`: 200 shape, 422, 503, `/health`.
- `ruff` (lint) configured in pyproject.
- GitHub Actions `ci.yml`: ruff + pytest on push/PR + `docker build` check.

## 5. Packaging, config, docs

- pyproject: entry points `protein-api` (uvicorn runner) and `protein-eval`;
  dev extra `[dev]` with pytest, ruff, fastapi test deps; add `fastapi`,
  `uvicorn` to deps.
- `.env.example` updated (`OLLAMA_HOST`, `PROTEIN_LLM_TIMEOUT_S`,
  `PROTEIN_MATCHER`, `PROTEIN_FUZZY_THRESHOLD`).
- README: API usage (curl), Docker (both provider modes + local profile),
  test/CI badges-level instructions, Limitations section.

## Sequencing

1. Commit the currently-uncommitted matcher bake-off work as its own commit.
2. Core hardening (errors, retries, logging, locks, validation) + tests.
3. API + tests.
4. Docker + compose.
5. CI, packaging, README. Push to GitHub.

## Out of scope

Auth, rate limiting, persistence/meal history, embeddings matcher (#3),
changing eval numbers or one-pager content.
