"""Foundational primitives: nanosecond UTC clock, typed error hierarchy, canonical JSON/SHA-256,
Numba JIT wrapper, and the domain-neutral forecast contract.

Importing :mod:`financeai.core` (or ``financeai.core.errors`` inside a fresh worker process) never
loads NumPy or Numba at module import time; :mod:`financeai.core.clock` (which compiles
:func:`days_from_civil` with Numba) and :mod:`financeai.core.forecast` (which uses NumPy arrays)
are loaded lazily on first attribute access (PEP 562).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from financeai.core.errors import (
    FinanceAIError,
    IncompatibleSchemaError,
    InvalidConfigError,
    InvalidDataError,
    InvariantViolationError,
    OptionalDependencyError,
    UnsupportedCapabilityError,
)
from financeai.core.hashing import sha256_bytes, sha256_file, sha256_json
from financeai.core.jit import HAVE_NUMBA, njit
from financeai.core.serialization import canonical_json, loads_strict

if TYPE_CHECKING:
    from financeai.core.clock import (
        MISSING_NS,
        NS_PER_DAY,
        NS_PER_MS,
        NS_PER_S,
        NS_PER_US,
        days_from_civil,
        format_ns,
        parse_recorder_datetime_ns,
    )
    from financeai.core.forecast import (
        AvailabilityRule,
        Capability,
        ForecastAdapter,
        ForecastQuery,
        ForecastTicket,
        MatureBatch,
        Outcome,
        TargetSpec,
        decode_ns,
        encode_ns,
        mixture_moments,
        require_capability,
        require_keys,
        require_schema,
    )

_CLOCK_EXPORTS = frozenset(
    {
        "MISSING_NS",
        "NS_PER_DAY",
        "NS_PER_MS",
        "NS_PER_S",
        "NS_PER_US",
        "days_from_civil",
        "format_ns",
        "parse_recorder_datetime_ns",
    }
)
_FORECAST_EXPORTS = frozenset(
    {
        "AvailabilityRule",
        "Capability",
        "ForecastAdapter",
        "ForecastQuery",
        "ForecastTicket",
        "MatureBatch",
        "Outcome",
        "TargetSpec",
        "decode_ns",
        "encode_ns",
        "mixture_moments",
        "require_capability",
        "require_keys",
        "require_schema",
    }
)

__all__ = [
    "HAVE_NUMBA",
    "MISSING_NS",
    "NS_PER_DAY",
    "NS_PER_MS",
    "NS_PER_S",
    "NS_PER_US",
    "AvailabilityRule",
    "Capability",
    "FinanceAIError",
    "ForecastAdapter",
    "ForecastQuery",
    "ForecastTicket",
    "IncompatibleSchemaError",
    "InvalidConfigError",
    "InvalidDataError",
    "InvariantViolationError",
    "MatureBatch",
    "OptionalDependencyError",
    "Outcome",
    "TargetSpec",
    "UnsupportedCapabilityError",
    "canonical_json",
    "days_from_civil",
    "decode_ns",
    "encode_ns",
    "format_ns",
    "loads_strict",
    "mixture_moments",
    "njit",
    "parse_recorder_datetime_ns",
    "require_capability",
    "require_keys",
    "require_schema",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
]


def __getattr__(name: str) -> Any:
    if name in _CLOCK_EXPORTS:
        from financeai.core import clock

        return getattr(clock, name)
    if name in _FORECAST_EXPORTS:
        from financeai.core import forecast

        return getattr(forecast, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
