"""Exception hierarchy.

Every exception raised deliberately inside the pipeline derives from `AgenthonError`, so the
top-level handler can tell a *known* degraded state (e.g. the LLM is unreachable) from an
unexpected bug. Both end in an admissible forecast; only the log line differs.
"""

from __future__ import annotations


class AgenthonError(Exception):
    """Base class for every deliberate failure."""


class CardError(AgenthonError):
    """card.toml missing, unreadable or semantically unusable."""


class DataError(AgenthonError):
    """Panels or corpus missing/malformed for a required asset."""


class TransformError(AgenthonError):
    """Target transformation or horizon conversion impossible."""


class ModelError(AgenthonError):
    """A numerical model could not produce a valid distribution."""


class LLMUnavailable(AgenthonError):
    """No endpoint configured, network refused, budget exhausted, or timeout."""


class LLMParseError(AgenthonError):
    """The model answered but the answer does not satisfy the schema."""


class BudgetExceeded(AgenthonError):
    """A stage exceeded its wall-clock or request budget."""


class OutputValidationError(AgenthonError):
    """The forecast would fail an admissibility gate; the writer refuses to emit it."""
