"""v2 — grounded pipeline. parse (LLM) -> ground (DB) -> compute (Python).

The model only identifies items. Protein numbers come from Open Food Facts.
The sum is deterministic Python. `no_hit` items are flagged, not guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, foods, llm
from .foods import GroundResult


@dataclass
class MealResult:
    text: str
    items: list[GroundResult] = field(default_factory=list)
    total_protein_g: float = 0.0
    no_hits: list[str] = field(default_factory=list)

    @property
    def meets_goal(self) -> bool:
        return self.total_protein_g >= config.PROTEIN_GOAL_G


def analyze(meal_text: str) -> MealResult:
    meal = llm.parse_meal(meal_text)
    result = MealResult(text=meal_text)
    for item in meal.items:
        g = foods.ground_item(item)
        result.items.append(g)
        if g.no_hit:
            result.no_hits.append(g.name)
        else:
            result.total_protein_g += g.protein_g
    result.total_protein_g = round(result.total_protein_g, 2)
    return result


def total_protein(meal_text: str) -> float:
    return analyze(meal_text).total_protein_g


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "2 eggs and a chicken breast"
    res = analyze(text)
    print(f"[v2 grounded]  {text!r}   (provider: {config.provider()})")
    for it in res.items:
        if it.no_hit:
            print(f"  - {it.name:<22} {it.grams:>6.0f} g   NO HIT (not in DB)")
        else:
            print(f"  - {it.name:<22} {it.grams:>6.0f} g   {it.protein_per_100g:>5.1f} g/100g   -> {it.protein_g:>5.1f} g")
    print(f"  total protein: {res.total_protein_g:.1f} g  (goal {config.PROTEIN_GOAL_G:.0f} g — {'MET' if res.meets_goal else 'under'})")
    if res.no_hits:
        print(f"  flagged for manual confirm: {', '.join(res.no_hits)}")
