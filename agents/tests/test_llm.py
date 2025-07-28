import pytest

from agents.llm import parse_signature


async def func_without_type_hint(param):
    """Test function without type hint."""
    return param


def test_parse_signature_without_type_hint():
    """Test that parse_signature raises ValueError when a function has a parameter without a type hint."""
    with pytest.raises(ValueError) as excinfo:
        parse_signature(func_without_type_hint)

    assert "must have type hints" in str(excinfo.value)
