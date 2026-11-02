"""Strict, lossless canonical strategy documents at the service boundary.

:meth:`StrategyConfig.from_mapping` is the shared application validator for structure,
policy types, model documents, and ranges. The service additionally bounds every leaf
and normalizes equivalent spellings before that validator, so the hashed, stored and
published strategy is exactly the one that runs:

* integers -- policy ``direction`` (-1, 0, +1), ``size_units``, ``*_ms`` and ``*_ticks`` (the
  episode-clock knobs ``abs_cap_ms`` and ``early_check_ms`` among them), ``size_units_cap`` and
  ``max_clips``, signal parameters with a unit suffix (``_ms``, ``_units``, ``_ticks``) or one of the
  unsuffixed counts in :data:`_SIGNAL_INTS` (``ticks`` of ``quote_confirm``, ``levels`` of ``ofi``)
  and model ``n_train`` -- are JSON integers (no bool and no float, not even ``1000.0``),
  non-negative and at most :data:`MAX_INT_PARAMETER`, except ``depth_ticks`` (-1 disables it),
  ``max_hold_ms`` (at most :data:`MAX_HOLD_MS`) and ``size_units_cap`` / ``max_clips`` (1 to
  :data:`MAX_MULTIPLIER`);
* binary64 values -- policy ``enter``/``exit``/``stop_bps``/``early_markout_bps``, a composite
  component's ``scale``, session offsets (seconds, at most :data:`MAX_SESSION_S`) and model
  ``alpha``/``coef``/``intercept`` -- accept JSON integers and floats and are stored as floats
  (``-0.0`` as ``0.0``), so equivalent spellings hash identically; an integer that binary64
  cannot represent exactly is rejected;
* ``allow_short``, ``refresh_hold`` and ``reverse`` are JSON booleans; names, labels, features,
  instrument keys and the other signal parameters (kind, venues, ``normalize``) are strings;
* signal recipes nest (:func:`financeai.strategies.leadlag.build_signal`): ``gate`` and
  ``fresh`` objects, composite ``components`` (each a ``spec`` recipe with a ``scale``) and the
  optional ``exit_signal`` recipe follow the same rules at every depth up to
  :data:`MAX_SIGNAL_DEPTH`.

The bounds also keep derived clocks inside int64: a cooldown of at most 10**12 ms (about 31.7
years) added to clocks up to :data:`~financeai.domains.market.synthetic.MAX_START_NS` stays far
below ``2**63``. The policy key sets below mirror ``StrategyConfig``'s ``_POLICY`` and
``_POLICY_OPTIONAL``: a key missing here would be passed through untyped (hashed and published as
written, run coerced), so ``tests/service/test_canonical.py`` fails until a new engine knob is typed.
Recipe-specific signal keys are still checked when the worker builds the signal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from financeai.application.strategy import StrategyConfig, model_from_json
from financeai.core.errors import InvalidConfigError
from financeai.core.serialization import loads_strict

MAX_INT_PARAMETER: Final = 10**12
"""Upper bound of every non-negative integer parameter (as a duration in ms: about 31.7 years)."""
MAX_SESSION_S: Final = 10**9
"""Upper bound of a session offset in seconds (about 31.7 years)."""
MAX_HOLD_MS: Final = 2**62 // 1_000_000
"""Upper bound of ``max_hold_ms``: the engine's "never" (``ThresholdParams.max_hold_ns`` default, 2**62 ns)
in whole milliseconds, as research candidates export it. Holds are only compared with elapsed time,
never added to a clock, so this bound needs no int64 headroom."""
MAX_MULTIPLIER: Final = 1_000
"""Upper bound of ``size_units_cap`` and ``max_clips``. The engine multiplies ``units * size_units`` and then
``clips`` in int64 before it clips a target to the risk cap; with ``size_units <= MAX_INT_PARAMETER`` the
product stays at most 10**18. A wrapped product can flip the target's sign (a long-only entry then sells),
silently in compiled mode."""
MAX_SIGNAL_DEPTH: Final = 8
"""Deepest nested object inside a signal recipe (a composite component is one level, its ``spec`` the
next, a ``gate`` one more): deep enough for composites of composites, shallow enough that a hostile
document gets a typed error instead of exhausting the interpreter's recursion limit."""

_LABELS: Final = ("schema", "schema_version", "name")
_POLICY_FLOATS: Final = ("enter", "exit", "stop_bps", "early_markout_bps")
_POLICY_INTS: Final = (
    ("min_hold_ms", 0, MAX_INT_PARAMETER),
    ("max_hold_ms", 0, MAX_HOLD_MS),
    ("cooldown_ms", 0, MAX_INT_PARAMETER),
    ("entry_slip_ticks", 0, MAX_INT_PARAMETER),
    ("exit_slip_ticks", 0, MAX_INT_PARAMETER),
    ("clip_spacing_ms", 0, MAX_INT_PARAMETER),
    ("depth_ticks", -1, MAX_INT_PARAMETER),
    ("size_units_cap", 1, MAX_MULTIPLIER),
    ("max_clips", 1, MAX_MULTIPLIER),
    ("abs_cap_ms", 0, MAX_INT_PARAMETER),
    ("early_check_ms", 0, MAX_INT_PARAMETER),
)
"""The episode-clock knobs ``abs_cap_ms`` / ``early_check_ms`` are only compared with elapsed time (``t - origin``)
but reach the kernel as int64 nanoseconds: 10**12 ms is 10**18 ns < 2**63, whereas 10**15 ms was admitted and
then failed the worker with a Numba typing error."""
_POLICY_BOOLS: Final = ("allow_short", "refresh_hold", "reverse")
_SIGNAL_UNITS: Final = ("_ms", "_units", "_ticks")
_SIGNAL_INTS: Final = ("ticks", "levels")
"""Integer recipe parameters without a unit suffix -- ``quote_confirm.ticks`` (price ticks) and ``ofi.levels``
(book levels), both read by ``strategies.leadlag`` with its integer-only ``_int`` -- which the string fallback
used to reject (``ticks: 2``) or admit as text (``levels: "3"``)."""
_SIGNAL_FLOATS: Final = ("scale",)


def _type_error(where: str, expected: str, value: object) -> InvalidConfigError:
    return InvalidConfigError(
        f"{where} must be {expected}", context={"field": where, "expected": expected, "got": type(value).__name__}
    )


def _mapping(value: object, where: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise _type_error(where, "an object", value)
    return value


def _array(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        raise _type_error(where, "an array", value)
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise _type_error(where, "a string", value)
    return value


def _int(value: object, where: str, low: int = 0, high: int = MAX_INT_PARAMETER) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(where, "a JSON integer", value)
    if not low <= value <= high:
        raise InvalidConfigError(f"{where} is out of range", context={"field": where, "min": low, "max": high})
    return int(value)


def _float(value: object, where: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _type_error(where, "a JSON number", value)
    try:
        out = float(value)
    except OverflowError:
        out = math.inf
    if not math.isfinite(out) or (isinstance(value, int) and out != value):
        raise InvalidConfigError(
            f"{where} must be a finite number that binary64 represents exactly", context={"field": where}
        )
    if (low is not None and out < low) or (high is not None and out > high):
        raise InvalidConfigError(f"{where} is out of range", context={"field": where, "min": low, "max": high})
    return out + 0.0  # one spelling of zero: -0.0 + 0.0 == +0.0


def _signal(value: object, where: str = "signal", depth: int = 0) -> dict[Any, Any]:
    """Check one signal recipe (or exit recipe). Nested objects -- ``gate``, ``fresh``, each composite
    component and its ``spec`` -- follow the same leaf rules one level deeper."""
    if depth > MAX_SIGNAL_DEPTH:
        raise InvalidConfigError(
            f"{where} nests signal objects too deeply", context={"field": where, "max_depth": MAX_SIGNAL_DEPTH}
        )
    out: dict[Any, Any] = {}
    for key, item in _mapping(value, where).items():
        at = f"{where}.{key}"
        if key == "components":
            out[key] = [_signal(part, f"{at}.{i}", depth + 1) for i, part in enumerate(_array(item, at))]
        elif key in _SIGNAL_FLOATS:
            out[key] = _float(item, at)
        elif key in _SIGNAL_INTS or str(key).endswith(_SIGNAL_UNITS):
            out[key] = _int(item, at)
        elif isinstance(item, Mapping):
            out[key] = _signal(item, at, depth + 1)
        else:
            out[key] = _text(item, at)
    return out


def _policy(value: object) -> dict[Any, Any]:
    policy = _mapping(value, "policy")
    out = dict(policy)
    for name, low, high in (("direction", -1, 1), ("size_units", 0, MAX_INT_PARAMETER)):
        if name in policy:
            out[name] = {
                key: _int(item, f"policy.{name}.{key}", low, high)
                for key, item in _mapping(policy[name], f"policy.{name}").items()
            }
    for name in _POLICY_FLOATS:
        if name in policy:
            out[name] = _float(policy[name], f"policy.{name}")
    for name, low, high in _POLICY_INTS:
        if name in policy:
            out[name] = _int(policy[name], f"policy.{name}", low, high)
    for name in _POLICY_BOOLS:
        if name in policy and not isinstance(policy[name], bool):
            raise _type_error(f"policy.{name}", "a JSON boolean", policy[name])
    return out


def _session(value: object) -> dict[Any, float]:
    return {key: _float(item, f"session.{key}", 0.0, MAX_SESSION_S) for key, item in _mapping(value, "session").items()}


def _model(value: object) -> dict[Any, Any]:
    model = _mapping(value, "model")
    out = dict(model)
    if "kind" in model:
        _text(model["kind"], "model.kind")
    for name in ("alpha", "intercept"):
        if name in model:
            out[name] = _float(model[name], f"model.{name}")
    if "coef" in model:
        out["coef"] = [_float(c, f"model.coef.{i}") for i, c in enumerate(_array(model["coef"], "model.coef"))]
    if "features" in model:
        out["features"] = [
            _text(f, f"model.features.{i}") for i, f in enumerate(_array(model["features"], "model.features"))
        ]
    if "n_train" in model:
        out["n_train"] = _int(model["n_train"], "model.n_train")
    model_from_json(out)  # the shared validator: exact keys, kind, one coefficient per feature, alpha
    return out


def canonical_strategy(doc: object) -> StrategyConfig:
    """Check every leaf of a strategy document and build its canonical :class:`StrategyConfig`.

    Raises :class:`~financeai.core.errors.InvalidConfigError` for a lossy, mistyped or
    out-of-range value (``context["field"]`` names it, e.g. ``policy.direction.venue_a/ALPHA-USD``)
    and whatever the shared validator raises (e.g. an incompatible schema). The result's
    :meth:`~StrategyConfig.to_mapping` is the canonical document; it is a fixed point.
    """
    out = dict(_mapping(doc, "strategy"))
    for name in _LABELS:
        if name in out:
            _text(out[name], name)
    if "instruments" in out:
        out["instruments"] = [
            _text(k, f"instruments.{i}") for i, k in enumerate(_array(out["instruments"], "instruments"))
        ]
    if "signal" in out:
        out["signal"] = _signal(out["signal"])
    if out.get("exit_signal") is not None:
        out["exit_signal"] = _signal(out["exit_signal"], "exit_signal")
    if "policy" in out:
        out["policy"] = _policy(out["policy"])
    if "session" in out:
        out["session"] = _session(out["session"])
    if out.get("model") is not None:
        out["model"] = _model(out["model"])
    return StrategyConfig.from_mapping(out)


def canonical_config(cfg: StrategyConfig) -> StrategyConfig:
    """Canonical form of an already built config (rejects values its engine parameters would coerce)."""
    return canonical_strategy(cfg.to_mapping())


def load_strategy(path: str | Path) -> StrategyConfig:
    """Read a strategy file with strict JSON (duplicate keys, NaN rejected) and canonicalize it."""
    return canonical_strategy(loads_strict(Path(path).read_bytes()))


__all__ = [
    "MAX_HOLD_MS",
    "MAX_INT_PARAMETER",
    "MAX_MULTIPLIER",
    "MAX_SESSION_S",
    "MAX_SIGNAL_DEPTH",
    "canonical_config",
    "canonical_strategy",
    "load_strategy",
]
