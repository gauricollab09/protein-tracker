"""Matcher bake-off: substring vs BM25 vs fuzzy on messy food names.

    uv run eval_match.py

Scores each matcher by how often it resolves a (often messy) food name to the
correct verified-DB key — including correctly returning nothing for foods that
aren't in the DB. The point is to CHOOSE the matcher by measured hit-rate, not
by assuming BM25 (or anything) is better. Writes match_results.txt.
"""
from __future__ import annotations

import json

from protein import config, matchers
from protein.foods import _load_verified

MATCHER_ORDER = ["substring", "bm25", "fuzzy"]


def load() -> list[dict]:
    path = config.DATA / "match_testset.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_matcher(name: str, mname: str, keys: list[str]) -> str | None:
    return matchers.get(mname)(name, keys)


def main() -> None:
    rows = load()
    keys = list(_load_verified())
    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    emit(f"Food-name matcher bake-off — {len(rows)} names vs {len(keys)} DB keys")
    emit("=" * 78)
    emit(f"{'name':<22}{'expected':<18}{'substring':<13}{'bm25':<13}{'fuzzy':<13}")
    emit("-" * 78)

    correct = {m: 0 for m in MATCHER_ORDER}
    for r in rows:
        name, expected = r["name"], r["expected"]
        preds = {m: run_matcher(name, m, keys) for m in MATCHER_ORDER}

        def cell(m: str) -> str:
            p = preds[m]
            ok = (p == expected)
            if ok:
                correct[m] += 1
            mark = "✓" if ok else "✗"
            return f"{mark} {p if p is not None else '—'}"

        exp_disp = expected if expected is not None else "(no-hit)"
        emit(f"{name:<22}{exp_disp:<18}{cell('substring'):<13}{cell('bm25'):<13}{cell('fuzzy'):<13}")

    emit("=" * 78)
    emit("")
    n = len(rows)
    emit("ACCURACY (correct key, incl. correct no-hits)")
    for m in MATCHER_ORDER:
        bar = "█" * round(20 * correct[m] / n)
        emit(f"  {m:<10}{correct[m]:>3}/{n}   {100*correct[m]/n:>5.0f}%   {bar}")
    emit("")
    winner = max(MATCHER_ORDER, key=lambda m: correct[m])
    emit(f"  winner: {winner}  ({100*correct[winner]/n:.0f}%)  ← wired in via config.MATCHER")
    emit("")
    emit("  Read it for the round table: substring is the brittle baseline; BM25")
    emit("  fixes word-order/partial but is blind to spelling; fuzzy also catches")
    emit("  typos & spelling. What ALL three miss (e.g. 'chx breast') is where a")
    emit("  semantic/embedding matcher would be the next step.")

    (config.ROOT / "match_results.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
