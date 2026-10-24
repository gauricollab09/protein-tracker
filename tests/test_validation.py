import pytest

from protein.errors import InputError
from protein.schema import MAX_MEAL_CHARS, validate_meal_text


def test_valid_text_is_stripped():
    assert validate_meal_text("  2 eggs  ") == "2 eggs"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_text_rejected(bad):
    with pytest.raises(InputError):
        validate_meal_text(bad)


def test_oversized_text_rejected():
    with pytest.raises(InputError):
        validate_meal_text("x" * (MAX_MEAL_CHARS + 1))


def test_max_length_boundary_ok():
    assert validate_meal_text("x" * MAX_MEAL_CHARS) == "x" * MAX_MEAL_CHARS
