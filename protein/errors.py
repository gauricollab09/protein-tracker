"""Typed failures. A 0.0 result must mean "the model said 0" — anything that
breaks raises one of these instead of being swallowed."""
from __future__ import annotations


class ProteinError(Exception):
    """Base for all protein-tracker errors."""


class InputError(ProteinError):
    """The caller's meal text is unusable (empty, too long)."""


class LLMError(ProteinError):
    """The LLM provider call failed after retries, or returned garbage."""
