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
        # plural units from the parser singularize instead of falling through
        (FoodItem(name="whole wheat bread", quantity=2, unit="slices"), 56.0),
        (FoodItem(name="whey protein", quantity=2, unit="scoops"), 60.0),
        (FoodItem(name="rice", quantity=2, unit="cups"), 400.0),
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
    val, source = foods.lookup("moong dal")
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


def test_off_network_failure_not_cached(monkeypatch):
    monkeypatch.setattr(config, "MATCHER", "fuzzy")

    def raising(*a, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", raising)
    val, source = foods.lookup("some unknown food")
    assert (val, source) == (None, "none")
    assert not config.OFF_CACHE.exists() or "some unknown food" not in json.loads(
        config.OFF_CACHE.read_text()
    )

    # a subsequent successful lookup should retry (not be poisoned by the failure)
    monkeypatch.setattr(httpx, "get", _fake_off_response)
    val, source = foods.lookup("some unknown food")
    assert (val, source) == (19.5, "off")


def test_ground_item_no_hit(monkeypatch):
    monkeypatch.setattr(config, "MATCHER", "fuzzy")
    monkeypatch.setattr(foods, "_off_lookup", lambda name: None)
    g = foods.ground_item(FoodItem(name="moong dal", quantity=1, unit="serving"))
    assert g.no_hit and g.protein_g == 0.0 and g.source == "none"
