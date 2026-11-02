"""Strict JSON for configurations, manifests and run records.

Decoding rejects duplicate keys and ``NaN``/``Infinity`` *before* any normalisation,
so two different documents can never silently map to one configuration. A number spelled
finitely but too large for binary64 (``1e400``) is rejected the same way instead of
decoding to ``inf``. Every decoder failure is the typed :class:`InvalidConfigError`: the
standard decoder also raises plain ``ValueError`` (an integer literal beyond CPython's
int-string limit, invalid UTF-8 bytes) and ``RecursionError`` (nesting deeper than the
interpreter allows), which callers such as the HTTP boundary must report as a malformed
document, not as an internal error. Canonical encoding (sorted keys, compact separators,
finite numbers only) feeds content hashes.
"""

from __future__ import annotations

import json
import math
from typing import Any

from financeai.core.errors import InvalidConfigError


def _reject_constant(token: str) -> Any:
    raise InvalidConfigError("non-finite JSON number is not allowed", context={"token": token})


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise InvalidConfigError("non-finite JSON number is not allowed", context={"token": token[:64]})
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise InvalidConfigError("duplicate JSON key", context={"key": key})
        out[key] = value
    return out


def loads_strict(text: str | bytes) -> Any:
    """Parse JSON, rejecting duplicate keys and non-finite numbers at any depth.

    Raises :class:`InvalidConfigError` (``malformed JSON``) for every undecodable document,
    including the decoder's non-``JSONDecodeError`` failures: oversize integer literals and
    invalid UTF-8 (both ``ValueError``) and over-deep nesting (``RecursionError``).
    """
    try:
        return json.loads(
            text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant, parse_float=_finite_float
        )
    except json.JSONDecodeError as exc:
        raise InvalidConfigError("malformed JSON", context={"error": str(exc)}) from exc
    except (ValueError, RecursionError) as exc:  # JSONDecodeError is caught above; these are its cousins
        detail = f"{type(exc).__name__}: {str(exc)[:160]}"
        raise InvalidConfigError("malformed JSON", context={"error": detail}) from exc


def canonical_json(value: Any) -> str:
    """Deterministic compact JSON used for hashing and identity."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise InvalidConfigError("value is not canonical-JSON serialisable") from exc
