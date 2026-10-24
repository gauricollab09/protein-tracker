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
        ("fuzzy", "chedder cheese", None),                    # WRatio 77.14 < 80: typo+extra-word misses vs this small key list (in the real DB it matches the 'cheese' key)
        ("fuzzy", "penut butter", "peanut butter"),           # typo (WRatio 96.0 >= 80)
        ("fuzzy", "quinoa salad", None),                      # out of db -> no hit
    ],
)
def test_matchers(fn_name, name, expected):
    fn = matchers.MATCHERS[fn_name]
    got = fn(name, KEYS, threshold=80.0) if fn_name == "fuzzy" else fn(name, KEYS)
    assert got == expected


def test_registry_unknown_raises():
    with pytest.raises(ValueError):
        matchers.get("nonsense")
