# Protein Tracker — built to be defended

A personal study project: an AI nutrition logger built to be defended.
The one idea: **the LLM is a parser, not a nutrition oracle.** It turns messy
free text into structured food items; the protein numbers come from a verified
database; totals are computed in Python; an eval harness proves the grounded
version beats the naive one.

Everything runs in an **isolated environment** — `uv`-managed Python 3.12 in a
project-local `.venv`, or a single Docker image. No keys required: with none
set, the parser falls back to local **Ollama (llama3.2)**.

## The two versions

| | v0 — naive | v2 — grounded |
|---|---|---|
| approach | one LLM call → "total protein grams" | LLM parses items → verified DB grounds → Python computes |
| trusts the model for | the number (the thing it's bad at) | only item identification |
| unknown food | guesses a plausible number | abstains — `no_hit`, flagged for manual confirm |

## Measured results (`uv run protein-eval`, provider: ollama/llama3.2)

```
                              MAE (g)   within ±5g
v0  LLM returns the number      12.3        44%
v2  parse → ground → compute     3.1        89%     (2.4 g on answered items)
```

- **Grounding cut MAE ~4×** (12.3 g → 3.1 g); on the items it actually answers,
  v2 lands within **2.4 g** and is inside ±5 g **94%** of the time.
- v2 **abstained on 1/18** items (a homemade rajma not in the DB → flagged, not
  guessed). It never invents a number for food it can't identify.
- **Dominant residual error: compound dishes** (87% of answered error, n=3) —
  this is *portion/composition* estimation, **not** food identity. Single foods,
  countables, and volumes are essentially exact.

### Why the end-to-end MAE went *up* when I improved the matcher

An earlier configuration used a substring matcher and reported v2 **MAE 1.8 g**.
When I ran a matcher bake-off (below) and switched to the fuzzy matcher — a clear
component win, 90% vs 52% — the end-to-end MAE **rose to 3.1 g**. That looks like
a regression. It isn't.

The substring matcher missed bare `"chicken"` (it only matched `"chicken breast"`
as a substring), so the pipeline fell through to a noisy Open Food Facts value
(~8.4 g/100 g for a chicken-flavored product, measured live against the OFF API
during development) instead of the real 31 g/100 g. That
wrong-low number happened to **cancel out** an over-portioned estimate on a
*chicken caesar salad* (a compound dish the model over-sizes). Two errors quietly
adding to zero. The fuzzy matcher does the right thing — `"chicken" → "chicken
breast" → 31 g/100 g` — which removed the accidental cancellation and **exposed
the pre-existing portion-estimation error** that was always there.

The lesson: **end-to-end metrics can hide compensating errors.** If I had chased
the single number I would have "fixed" the pipeline by keeping a broken matcher.
You evaluate components separately, or you optimize a lie. Fuzzy is the correct
choice; 3.1 g is the honest number.

Numbers are measured from this repo and written to `eval_results.txt` /
`match_results.txt`. They vary slightly between runs (local LLM).

## Run it (native)

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"   # first time
ollama pull llama3.2                                   # if not already present

uv run python -m protein.v0 "2 eggs and a chicken breast"   # naive baseline
uv run python -m protein.v2 "2 eggs and a chicken breast"   # grounded, itemized
uv run protein-eval         # the before/after table   -> eval_results.txt
uv run protein-eval-match   # the matcher bake-off      -> match_results.txt
```

To use Claude instead of Ollama, put `ANTHROPIC_API_KEY=...` in `.env`
(this is API billing, separate from a claude.ai subscription). See `.env.example`.

## API

The same library is exposed over HTTP with FastAPI. Validation and errors live in
the library; the API is a thin layer that maps them (`InputError → 422`,
`LLMError → 503`).

```bash
uv run protein-api                    # console script
# or: uv run uvicorn protein.api:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/** — a human-friendly demo page (type a meal,
see the itemized grounding, compare against the naive guess). Swagger for
developers stays at `/docs`.

Three endpoints:

```bash
# grounded pipeline — itemized, with no_hit flags
curl -s localhost:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"text": "150g grilled chicken breast and 2 eggs"}'

# naive baseline — one number, for the live A/B
curl -s localhost:8000/v0/estimate \
  -H 'content-type: application/json' \
  -d '{"text": "150g grilled chicken breast and 2 eggs"}'

# health — provider, matcher, DB size
curl -s localhost:8000/health
```

`/analyze` returns each parsed item with its grounded `protein_per_100g`, the
`source` (verified / off / none), the computed `protein_g`, a `total_protein_g`,
whether it `meets_goal`, and a `no_hits` list of foods that were flagged rather
than guessed.

## Docker — one image, either provider

The image builds once and runs against whichever LLM you point it at.

```bash
# 1. Host Ollama (default). Uses the Ollama already running on your Mac
#    via host.docker.internal. Nothing leaves your machine.
docker compose up --build

# 2. Cloud Claude. Set the key; the same image switches providers by env.
ANTHROPIC_API_KEY=sk-ant-... docker compose up --build

# 3. Fully local, self-contained. Runs Ollama in a sibling container —
#    point OLLAMA_HOST at it so the api actually talks to it (no host
#    dependency once this is set).
OLLAMA_HOST=http://ollama:11434 docker compose --profile local up --build
docker compose exec ollama ollama pull llama3.2   # one-time model pull
```

The service listens on `localhost:8000` in every mode; the container has a
`/health` HEALTHCHECK and runs as a non-root user.

## Tests & CI

```bash
uv run pytest -q        # 50 tests, no network required
uv run ruff check .     # lint
```

All 50 tests run **offline** — the LLM and Open Food Facts are stubbed, so the
suite is deterministic and free. GitHub Actions (`.github/workflows/ci.yml`) runs
two jobs on every push and PR: **test** (ruff + pytest under `uv`) and **docker**
(a full image build).

## Grounding source — a defended decision

The plan named Open Food Facts. On inspection it's a **packaged-food** DB:
generic raw ingredients return noisy branded matches (a salmon-*flavored* snack
at 7.8 g/100 g — measured live against the OFF API during development) or null
protein, and the search endpoint rate-limits. So
grounding is **curated-first**: a small USDA-sourced `data/verified_foods.json`
(31 staples), with **OFF as a live fallback** (cached to `data/off_cache.json`),
and `no_hit` when neither has it. The OFF noise is itself measured evidence for
the round-table questions — and the small DB is deliberate: I'd rather abstain on
a food than serve a number I can't stand behind.

## Limitations (by design, and known)

- **No auth, rate limiting, or persistence.** This is a single-user demo of a
  measured pipeline, not a multi-tenant service. Each request is stateless.
- **Portion estimation is the dominant residual error**, not food identity —
  compound dishes ("a chicken caesar salad") are over/under-sized by the model.
  That's where the next accuracy work goes, and the eval already isolates it.
- **The verified DB is small (31 foods) on purpose.** Coverage grows by curation;
  anything outside it either hits the OFF fallback or abstains, never guesses.
- **Local LLM output varies run-to-run.** Numbers are re-measured, not asserted.

## Layout

```
protein/
  config.py         knobs: provider selection, goal, matcher, OFF settings
  schema.py         Pydantic FoodItem / Meal — the LLM↔pipeline contract; input validation
  errors.py         typed failures — a returned 0 means "the model said 0", never a swallowed error
  llm.py            one adapter: parse_meal() [v2] + ask_total() [v0]; Claude or Ollama; retries + timeouts
  foods.py          grounding: verified DB → OFF fallback → no_hit; thread-safe + atomic cache writes
  matchers.py       substring / bm25 / fuzzy food-name matchers
  v0.py / v2.py     naive baseline / grounded pipeline
  api.py            FastAPI service: /analyze, /v0/estimate, /health
  eval_pipeline.py  v0 vs v2 → MAE before/after       -> eval_results.txt
  eval_matchers.py  matcher bake-off                  -> match_results.txt
data/
  testset.jsonl        18 labeled meals, tagged by failure mode
  match_testset.jsonl  21 messy food names, for the matcher bake-off
  verified_foods.json  31 curated USDA protein-per-100g values
Dockerfile / docker-compose.yml   one image, either provider
```
