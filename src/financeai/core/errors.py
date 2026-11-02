"""Typed errors with stable codes.

Python callers, the CLI and any future transport adapter map failures through these
classes. ``code`` is part of the public contract; messages are for humans. Expected
domain outcomes (abstention, no-fill, unavailable payoff) are *not* errors and are
modelled as typed results elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar


class FinanceAIError(Exception):
    """Base class: a stable ``code`` plus immutable diagnostic context."""

    code: ClassVar[str] = "financeai_error"

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: Mapping[str, object] = MappingProxyType(dict(context or {}))

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({details})"


class InvalidDataError(FinanceAIError):
    """Input data violates its declared schema or a semantic invariant."""

    code = "invalid_data"


class InvalidConfigError(FinanceAIError):
    """A configuration value is malformed, ambiguous or out of its allowed range."""

    code = "invalid_config"


class IncompatibleSchemaError(FinanceAIError):
    """A persisted artifact uses an unknown or incompatible schema version."""

    code = "incompatible_schema"


class UnsupportedCapabilityError(FinanceAIError):
    """An operation was requested from a component that does not declare it."""

    code = "unsupported_capability"


class OptionalDependencyError(FinanceAIError):
    """An optional extra is not installed (distinct from a failure inside it)."""

    code = "optional_dependency_missing"


class InvariantViolationError(FinanceAIError):
    """An internal accounting/state invariant failed; the run must stop."""

    code = "invariant_violation"
