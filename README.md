# Protein Tracker — built to be defended

An AI nutrition logger for the **2-Day Production AI Thinking Intensive**.
The one idea: **the LLM is a parser, not a nutrition oracle.** It turns messy
free text into structured food items; the protein numbers come from a verified
database; an eval harness proves the grounded version beats the naive one.

Everything runs in an **isolated environment** — `uv`-managed Python 3.12 in a
project-local `.venv`. No keys required: with none set, the parser falls back to
local **Ollama (llama3.2)**.

## The two versions

| | v0 — naive | v2 — grounded |
|---|---|---|
| approach | one LLM call → "total protein grams" | LLM parses items → verified DB grounds → Python computes |
| trusts the model for | the number (the thing it's bad at) | only item identification |

## Measured results (`uv run eval.py`, provider: ollama/llama3.2)

```
                              MAE (g)   within ±5g
v0  LLM returns the number      11.3        56%
v2  parse → ground → compute     1.8        89%     (1.1 g on answered items)
```

- **Grounding cut MAE ~6×** (11.3 g → 1.8 g).
- v2 **abstained on 1/18** items (food not in DB → flagged, not guessed).
- **Dominant residual error: compound dishes** (72% of answered error) —
  portion/composition estimation, *not* food identity.

Numbers are measured from this repo's run and written to `eval_results.txt`.
They will vary slightly between runs (local LLM).

## Run it

```bash
uv venv --python 3.12 && uv pip install -e .   # first time
ollama pull llama3.2                            # if not already present

uv run python -m protein.v0 "2 eggs and a chicken breast"   # naive
uv run python -m protein.v2 "2 eggs and a chicken breast"   # grounded, itemized
uv run eval.py                                              # the before/after table
```

To use Claude instead of Ollama, put `ANTHROPIC_API_KEY=...` in `.env`
(this is API billing, separate from a claude.ai subscription).

## Grounding source — a defended decision

The plan named Open Food Facts. On inspection it's a **packaged-food** DB:
generic raw ingredients return noisy branded matches (a salmon-*flavoured* snack
at 7.8 g/100 g) or null protein, and the search endpoint rate-limits. So
grounding is **curated-first**: a small USDA-sourced `data/verified_foods.json`
of staples, with **OFF as a live fallback** (cached to `data/off_cache.json`),
and `no_hit` when neither has it. The OFF noise is itself measured evidence for
the round-table questions.

## Layout

```
protein/
  config.py   knobs: provider selection, goal=70g, OFF settings
  schema.py   Pydantic FoodItem / Meal — the LLM↔pipeline contract
  llm.py      one adapter: parse_meal() [v2] + ask_total() [v0]; Claude or Ollama
  foods.py    grounding: verified DB → OFF fallback → no_hit; unit→grams in code
  v0.py       naive baseline
  v2.py       grounded pipeline
data/
  testset.jsonl        ~18 labeled meals, tagged by failure mode
  verified_foods.json  curated USDA protein-per-100g
eval.py       runs v0 & v2 → MAE before/after → eval_results.txt
```
