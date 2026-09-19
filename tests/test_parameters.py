"""
Unit tests for generation parameters, schema validation, and edge cases.
"""
import pytest
from pydantic import ValidationError
from src.models.schemas import GenerationParameters, GenerateRequest, LoreCreateRequest


def test_generation_parameters_validation():
    """Verify bounds and validation on generation parameters."""
    # Valid parameters
    params = GenerationParameters(temperature=0.8, top_p=0.95, repeat_penalty=1.1, max_tokens=256)
    assert params.temperature == 0.8
    assert params.top_p == 0.95
    assert params.repeat_penalty == 1.1
    assert params.max_tokens == 256

    # Invalid temperature > 2.0
    with pytest.raises(ValidationError):
        GenerationParameters(temperature=2.5)

    # Invalid top_p > 1.0
    with pytest.raises(ValidationError):
        GenerationParameters(top_p=1.5)


def test_generate_request_validation():
    """Verify GenerateRequest validation."""
    req = GenerateRequest(prompt="Continue the adventure into the forgotten crypt.")
    assert req.prompt == "Continue the adventure into the forgotten crypt."
    assert req.parameters is None

    # Empty prompt
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")


def test_lore_create_request_validation():
    """Verify LoreCreateRequest validation."""
    req = LoreCreateRequest(content="A secret passage behind the tapestry.", metadata={"secret": True})
    assert req.content == "A secret passage behind the tapestry."
    assert req.metadata["secret"] is True

    with pytest.raises(ValidationError):
        LoreCreateRequest(content="")
