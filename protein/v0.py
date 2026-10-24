"""v0 — naive baseline. One LLM call returns the protein number.

Fast to build, confidently wrong. It exists to lose the A/B.
"""
from __future__ import annotations

from . import llm
from .schema import validate_meal_text


def total_protein(meal_text: str) -> float:
    """Ask the model directly for total grams of protein."""
    meal_text = validate_meal_text(meal_text)
    return llm.ask_total(meal_text)


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "2 eggs and a chicken breast"
    print(f"[v0 naive]  {text!r}")
    print(f"  total protein: {total_protein(text):.1f} g")
