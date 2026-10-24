import pytest

from protein import v2
from protein.errors import InputError
from protein.schema import FoodItem, Meal


def test_analyze_totals_and_no_hits(monkeypatch):
    meal = Meal(
        items=[
            FoodItem(name="chicken breast", quantity=150, unit="g"),   # 46.5 g
            FoodItem(name="egg", quantity=2, unit="count"),            # 12.6 g
            FoodItem(name="moong dal", quantity=1, unit="serving"), # no hit
        ]
    )
    monkeypatch.setattr(v2.llm, "parse_meal", lambda text: meal)
    monkeypatch.setattr(v2.foods, "_off_lookup", lambda name: None)
    res = v2.analyze("150g chicken breast, 2 eggs, moong dal")
    assert res.total_protein_g == pytest.approx(59.1, abs=0.1)
    assert res.no_hits == ["moong dal"]
    assert res.provider in ("claude", "ollama")


def test_analyze_rejects_empty():
    with pytest.raises(InputError):
        v2.analyze("   ")
