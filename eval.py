"""The hero artifact: run v0 and v2 over a labeled set, print the before/after.

    uv run eval.py

Outputs Mean Absolute Error (g) and % within +/-5g for each version, a per-
failure-mode breakdown that names the dominant residual error, and the no-hit
rate. Writes the same to eval_results.txt. Numbers are measured, never invented.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from protein import config, v0, v2


def load_testset() -> list[dict]:
    path = config.DATA / "testset.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def mae(errors: list[float]) -> float:
    return sum(errors) / len(errors) if errors else 0.0


def pct_within(errors: list[float], band: float) -> float:
    if not errors:
        return 0.0
    return 100.0 * sum(1 for e in errors if e <= band) / len(errors)


def main() -> None:
    rows = load_testset()
    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    emit(f"Protein Tracker — eval ({len(rows)} entries, provider: {config.provider()})")
    emit("=" * 72)
    emit(f"{'meal':<44}{'truth':>7}{'v0':>7}{'v2':>7}")
    emit("-" * 72)

    v0_err, v2_err = [], []                 # all items (abstention counted as error)
    v2_err_answered = []                     # items v2 actually grounded (no abstention)
    answered_err_by_mode: dict[str, list[float]] = defaultdict(list)
    no_hits = 0

    for r in rows:
        text, truth = r["text"], float(r["truth_g"])
        print(f"  ... running: {text!r}", file=sys.stderr)
        g0 = v0.total_protein(text)
        res2 = v2.analyze(text)
        g2 = res2.total_protein_g
        abstained = bool(res2.no_hits)
        if abstained:
            no_hits += 1

        e0, e2 = abs(g0 - truth), abs(g2 - truth)
        v0_err.append(e0)
        v2_err.append(e2)
        if not abstained:
            v2_err_answered.append(e2)
            answered_err_by_mode[r["mode"]].append(e2)

        flag = "  ⚑ no-hit" if abstained else ""
        label = text if len(text) <= 42 else text[:41] + "…"
        emit(f"{label:<44}{truth:>7.1f}{g0:>7.1f}{g2:>7.1f}{flag}")

    emit("=" * 72)
    emit("")
    emit("BEFORE / AFTER  (all items)")
    emit(f"  {'version':<10}{'approach':<28}{'MAE (g)':>10}{'within ±5g':>13}")
    emit(f"  {'v0':<10}{'LLM returns the number':<28}{mae(v0_err):>10.1f}{pct_within(v0_err, config.WITHIN_G):>12.0f}%")
    emit(f"  {'v2':<10}{'parse → ground → compute':<28}{mae(v2_err):>10.1f}{pct_within(v2_err, config.WITHIN_G):>12.0f}%")
    emit("")
    emit(f"  v2 abstained on {no_hits}/{len(rows)} item(s) (food not in DB → flagged for manual")
    emit("  confirm, not guessed). On the items it actually answered:")
    emit(f"  {'v2 answered':<10}{'(abstentions excluded)':<28}{mae(v2_err_answered):>10.1f}{pct_within(v2_err_answered, config.WITHIN_G):>12.0f}%")
    emit("")

    # dominant residual error mode among ANSWERED items
    total_resid = sum(sum(v) for v in answered_err_by_mode.values()) or 1.0
    emit("v2 residual error by failure mode (answered items, share of abs error):")
    ranked = sorted(answered_err_by_mode.items(), key=lambda kv: sum(kv[1]), reverse=True)
    for mode, errs in ranked:
        share = 100.0 * sum(errs) / total_resid
        emit(f"  {mode:<14}{sum(errs):>7.1f} g   {share:>5.0f}%   (n={len(errs)})")
    dominant_mode, dominant_errs = ranked[0]
    dominant_share = 100.0 * sum(dominant_errs) / total_resid
    emit("")
    emit(f"  dominant residual (answered): {dominant_mode}  ({dominant_share:.0f}% of answered error)")
    emit(f"  no-hit / abstention rate: {no_hits}/{len(rows)} = {100*no_hits/len(rows):.0f}%")
    emit("")
    emit(f"  THE LINE: \"naive MAE {mae(v0_err):.1f} g  →  grounded MAE {mae(v2_err):.1f} g"
         f"  ({mae(v2_err_answered):.1f} g on answered items)\"")

    (config.ROOT / "eval_results.txt").write_text("\n".join(out) + "\n")
    print("\n[written to eval_results.txt]", file=sys.stderr)


if __name__ == "__main__":
    main()
