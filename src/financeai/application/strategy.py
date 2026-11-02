"""Strategy configuration, model persistence and the end-to-end run use case.

A :class:`StrategyConfig` is the frozen, strict-JSON description of one policy: the
instrument order, the signal recipe (or a persisted H4 ridge model), the threshold-policy
parameters (per-instrument maps, milliseconds) and the session layout relative to the
data window. Python, the CLI and any future transport all go through this one validator.

Values are checked, never coerced: integer fields take ints (not bools, floats or numeric
strings, so ``2.9`` cannot run as 2 nor ``"false"`` as ``True``), binary64 fields take an int or
float spelling of a finite value (``-1000000000`` and ``-1000000000.0`` are one configuration),
flags take booleans and every section is an object. A violation raises
:class:`~financeai.core.errors.InvalidConfigError` whose ``context["field"]`` names the key. Only
JSON types count: research code that assembles a policy from numpy arrays converts each scalar with
``int()``/``float()`` first (an ``np.int64`` is rejected as ``got='int64'``).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.forecast import INT64_MAX, require_schema
from financeai.core.hashing import sha256_json
from financeai.core.serialization import canonical_json, loads_strict
from financeai.domains.market.cache import load_tapes
from financeai.domains.market.engine import (
    REASON_ENTRY,
    PreparedMarket,
    ReplayResult,
    ThresholdParams,
    run_replay,
)
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData
from financeai.models.ridge import RidgeModel
from financeai.strategies.leadlag import build_signal
from financeai.strategies.linear import feature_columns, prediction_signal

SCHEMA: Final = "financeai.strategy"
SCHEMA_VERSION: Final = "1.0"
_MS: Final = 1_000_000
_S: Final = 1_000_000_000
_TOP: Final = {"schema", "schema_version", "name", "instruments", "signal", "policy", "session"}
_POLICY: Final = {
    "direction",
    "size_units",
    "enter",
    "exit",
    "allow_short",
    "min_hold_ms",
    "max_hold_ms",
    "cooldown_ms",
    "stop_bps",
    "entry_slip_ticks",
    "exit_slip_ticks",
}
_SESSION: Final = {"warmup_s", "entry_stop_s", "flatten_s", "exec_stop_s"}
_POLICY_OPTIONAL: Final = {
    "size_units_cap",
    "depth_ticks",
    "max_clips",
    "clip_spacing_ms",
    "refresh_hold",
    "reverse",
    "abs_cap_ms",
    "early_check_ms",
    "early_markout_bps",
}
_LATE_DEFAULTS: Final[dict[str, Any]] = {
    "reverse": False,
    "abs_cap_ns": 0,
    "early_check_ns": 0,
    "early_markout_bps": 0.0,
}
"""Policy fields added after earlier rounds were registered, with the default that reproduces the
earlier behaviour; a field at this value is omitted from :func:`policy_config_id`."""


def policy_config_id(
    signal: Mapping[str, Any],
    params: ThresholdParams,
    exit_signal: Mapping[str, Any] | None = None,
) -> str:
    """Canonical 16-hex policy configuration identity (latency is a scenario dimension, not a config).

    Policy fields in :data:`_LATE_DEFAULTS` are omitted when at their default value so earlier
    identities and registered experiment hashes stay unchanged.
    """
    param_doc = {
        k: v for k, v in dataclasses.asdict(params).items() if not (k in _LATE_DEFAULTS and v == _LATE_DEFAULTS[k])
    }
    doc: dict[str, Any] = {"signal": dict(signal), "params": param_doc}
    if exit_signal is not None:
        doc["exit_signal"] = dict(exit_signal)
    return sha256_json(doc)[:16]


def _exact_keys(doc: Mapping[str, Any], allowed: set[str], where: str, optional: set[str] | None = None) -> None:
    keys = set(doc)
    missing = allowed - keys
    extra = keys - allowed - (optional or set())
    if missing or extra:
        raise InvalidConfigError(f"{where}: keys differ", context={"missing": sorted(missing), "extra": sorted(extra)})


_MISSING: Final = object()


def _type_error(field: str, expected: str, value: object) -> InvalidConfigError:
    return InvalidConfigError(
        f"{field} must be {expected}", context={"field": field, "expected": expected, "got": type(value).__name__}
    )


def _section(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    """A copied object section of the document; anything else is a typed error, never a ``TypeError``."""
    value = doc[key]
    if not isinstance(value, Mapping):
        raise _type_error(key, "an object", value)
    return dict(value)


def _policy_value(policy: Mapping[str, Any], key: str, default: object) -> object:
    value = policy.get(key, default)
    if value is _MISSING:
        raise InvalidConfigError(f"policy.{key} is required", context={"field": f"policy.{key}"})
    return value


def _policy_int(policy: Mapping[str, Any], key: str, default: object = _MISSING) -> int:
    """An integer policy value as written: a real ``int``, never a float (``2.9`` would run as 2), a numeric
    string or a bool (an ``int`` to Python, so ``True`` would run as 1)."""
    value = _policy_value(policy, key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(f"policy.{key}", "an integer", value)
    return value


def _policy_float(policy: Mapping[str, Any], key: str, default: object = _MISSING) -> float:
    """A finite binary64 value: an ``int`` or ``float`` spelling (``-1000000000`` and ``-1000000000.0`` are
    one configuration, as before), never a string, a bool, NaN, infinity or an int beyond the float range."""
    return _finite_number(_policy_value(policy, key, default), f"policy.{key}")


def _finite_number(value: object, field: str) -> float:
    """Read a JSON number without accepting strings, booleans or non-finite values."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _type_error(field, "a finite number", value)
    try:
        out = float(value)
    except OverflowError:
        out = math.inf
    if not math.isfinite(out):
        raise InvalidConfigError(f"{field} must be a finite number", context={"field": field})
    return out


def _policy_bool(policy: Mapping[str, Any], key: str, default: object = _MISSING) -> bool:
    """A flag as written: ``bool`` only, so ``"false"`` cannot run as ``True`` nor ``1`` stand in for it."""
    value = _policy_value(policy, key, default)
    if not isinstance(value, bool):
        raise _type_error(f"policy.{key}", "a boolean", value)
    return value


def _policy_venue_ints(policy: Mapping[str, Any], key: str, instruments: Sequence[str]) -> tuple[int, ...]:
    """A per-instrument integer map (``direction``, ``size_units``) read in instrument order."""
    table = _policy_value(policy, key, _MISSING)
    if not isinstance(table, Mapping):
        raise _type_error(f"policy.{key}", "an object keyed by instrument", table)
    out: list[int] = []
    for instrument in instruments:
        if instrument not in table:
            raise InvalidConfigError(
                f"policy.{key} needs one entry per instrument", context={"field": f"policy.{key}.{instrument}"}
            )
        value = table[instrument]
        if isinstance(value, bool) or not isinstance(value, int):
            raise _type_error(f"policy.{key}.{instrument}", "an integer", value)
        out.append(value)
    return tuple(out)


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    instruments: tuple[str, ...]
    signal: Mapping[str, Any]
    policy: Mapping[str, Any]
    session: Mapping[str, float]
    model: Mapping[str, Any] | None = None
    exit_signal: Mapping[str, Any] | None = None
    """Optional separate exit-signal recipe (e.g. gap closure); entries use ``signal``."""

    @classmethod
    def from_threshold_params(
        cls,
        *,
        name: str,
        instruments: Sequence[str],
        signal: Mapping[str, Any],
        params: ThresholdParams,
        session: Mapping[str, float],
        exit_signal: Mapping[str, Any] | None = None,
    ) -> StrategyConfig:
        """Export an engine policy without dropping fields or truncating its clocks.

        The JSON policy format stores integer milliseconds. Policies using finer hold
        intervals must use the Python replay API instead of silently losing precision.
        """
        keys = tuple(instruments)
        if len(keys) != len(params.direction) or len(keys) != len(params.size_units):
            raise InvalidConfigError("instrument order must match policy dimensions")
        policy: dict[str, Any] = {}
        for key, value in dataclasses.asdict(params).items():
            if key in ("direction", "size_units"):
                policy[key] = dict(zip(keys, value, strict=True))
            elif key.endswith("_ns"):
                if value % _MS:
                    raise InvalidConfigError("JSON policy clocks require whole milliseconds", context={"field": key})
                policy[key[:-3] + "_ms"] = value // _MS
            else:
                policy[key] = value
        doc = {
            "schema": "financeai.strategy",
            "schema_version": "1.0",
            "name": name,
            "instruments": list(keys),
            "signal": dict(signal),
            "policy": policy,
            "session": dict(session),
        }
        if exit_signal is not None:
            doc["exit_signal"] = dict(exit_signal)
        return cls.from_mapping(doc)

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any]) -> StrategyConfig:
        if not isinstance(doc, Mapping):
            raise InvalidConfigError("strategy config must be an object")
        _exact_keys(doc, _TOP, "strategy", optional={"model", "exit_signal"})
        require_schema(doc, schema=SCHEMA, version=SCHEMA_VERSION, where="strategy")
        if not isinstance(doc["name"], str) or not doc["name"].strip():
            raise _type_error("name", "a non-empty string", doc["name"])
        keys = doc["instruments"]
        if isinstance(keys, str | bytes) or not isinstance(keys, Sequence) or not all(isinstance(k, str) for k in keys):
            raise _type_error("instruments", "an array of instrument keys", keys)
        instruments = tuple(keys)
        if not instruments or len(set(instruments)) != len(instruments):
            raise InvalidConfigError("instruments must be a non-empty list of distinct keys")
        policy = _section(doc, "policy")
        _exact_keys(policy, _POLICY, "policy", optional=_POLICY_OPTIONAL)
        for field in ("direction", "size_units"):
            table = policy[field]
            if not isinstance(table, Mapping):
                raise _type_error(f"policy.{field}", "an object keyed by instrument", table)
            if set(table) != set(instruments):
                raise InvalidConfigError(
                    f"policy.{field} needs one entry per instrument", context={"field": f"policy.{field}"}
                )
        session = _section(doc, "session")
        _exact_keys(session, _SESSION, "session")
        for key, value in session.items():
            seconds = _finite_number(value, f"session.{key}")
            if not 0 <= seconds <= INT64_MAX // _S:
                raise InvalidConfigError(
                    "session offset must fit non-negative int64 nanoseconds", context={"field": f"session.{key}"}
                )
        cfg = cls(
            name=doc["name"],
            instruments=instruments,
            signal=_section(doc, "signal"),
            policy=policy,
            session=session,
            model=_section(doc, "model") if doc.get("model") is not None else None,
            exit_signal=_section(doc, "exit_signal") if doc.get("exit_signal") is not None else None,
        )
        cfg.threshold_params()  # validates the policy values
        if cfg.signal.get("kind") == "h4_ridge" and cfg.model is None:
            raise InvalidConfigError("h4_ridge signal requires a model")
        if cfg.model is not None:
            model_from_json(cfg.model)
        return cfg

    @classmethod
    def from_json(cls, text: str | bytes) -> StrategyConfig:
        return cls.from_mapping(loads_strict(text))

    @classmethod
    def load(cls, path: str | Path) -> StrategyConfig:
        return cls.from_json(Path(path).read_bytes())

    def to_mapping(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "instruments": list(self.instruments),
            "signal": dict(self.signal),
            "policy": dict(self.policy),
            "session": dict(self.session),
        }
        if self.model is not None:
            doc["model"] = dict(self.model)
        if self.exit_signal is not None:
            doc["exit_signal"] = dict(self.exit_signal)
        return doc

    def to_json(self) -> str:
        return canonical_json(self.to_mapping())

    @property
    def signal_id(self) -> str:
        """Canonical 16-hex hash of the entry signal recipe."""
        return sha256_json(dict(self.signal))[:16]

    @property
    def config_id(self) -> str:
        """Canonical 16-hex policy configuration identity matching :attr:`GridExperiment.config_id`."""
        return policy_config_id(self.signal, self.threshold_params(), self.exit_signal)

    def threshold_params(self) -> ThresholdParams:
        """The engine parameters, checking every policy value's type on the way (nothing is coerced, so the
        document spells exactly what runs); absent optional keys leave their options off as before."""
        p, keys = self.policy, self.instruments
        return ThresholdParams(
            direction=_policy_venue_ints(p, "direction", keys),
            size_units=_policy_venue_ints(p, "size_units", keys),
            enter=_policy_float(p, "enter"),
            exit=_policy_float(p, "exit"),
            allow_short=_policy_bool(p, "allow_short"),
            min_hold_ns=_policy_int(p, "min_hold_ms") * _MS,
            max_hold_ns=_policy_int(p, "max_hold_ms") * _MS,
            cooldown_ns=_policy_int(p, "cooldown_ms") * _MS,
            stop_bps=_policy_float(p, "stop_bps"),
            entry_slip_ticks=_policy_int(p, "entry_slip_ticks"),
            exit_slip_ticks=_policy_int(p, "exit_slip_ticks"),
            size_units_cap=_policy_int(p, "size_units_cap", 1),
            depth_ticks=_policy_int(p, "depth_ticks", -1),
            max_clips=_policy_int(p, "max_clips", 1),
            clip_spacing_ns=_policy_int(p, "clip_spacing_ms", 0) * _MS,
            refresh_hold=_policy_bool(p, "refresh_hold", False),
            reverse=_policy_bool(p, "reverse", False),
            abs_cap_ns=_policy_int(p, "abs_cap_ms", 0) * _MS,
            early_check_ns=_policy_int(p, "early_check_ms", 0) * _MS,
            early_markout_bps=_policy_float(p, "early_markout_bps", 0.0),
        )

    def session_plan(self, first_ns: int, last_ns: int) -> SessionPlan:
        s = self.session
        return SessionPlan.for_window(
            first_ns,
            last_ns,
            warmup_ns=round(s["warmup_s"] * _S),
            entry_stop_ns=round(s["entry_stop_s"] * _S),
            flatten_ns=round(s["flatten_s"] * _S),
            exec_stop_ns=round(s["exec_stop_s"] * _S),
        )


def model_to_json(model: RidgeModel, features: Sequence[str]) -> str:
    """Persist a fitted ridge model with one unique feature name per coefficient."""
    if model.coef_ is None:
        raise InvalidConfigError("cannot persist an unfitted model")
    doc = {
        "kind": "ridge",
        "alpha": model.alpha,
        "features": list(features),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_,
        "n_train": model.n_train_,
    }
    model_from_json(doc)
    return canonical_json(doc)


def model_from_json(text: str | bytes | Mapping[str, Any]) -> tuple[RidgeModel, list[str]]:
    """Load original-unit coefficients, rejecting malformed or lossy model documents."""
    doc = loads_strict(text) if isinstance(text, str | bytes) else text
    if not isinstance(doc, Mapping):
        raise _type_error("model", "an object", doc)
    _exact_keys(doc, {"kind", "alpha", "features", "coef", "intercept", "n_train"}, "model")
    names, coefficients = doc["features"], doc["coef"]
    if not isinstance(names, list | tuple) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise _type_error("model.features", "an array of non-empty feature names", names)
    if len(set(names)) != len(names):
        raise InvalidConfigError("model feature names must be unique", context={"field": "model.features"})
    if not isinstance(coefficients, list | tuple):
        raise _type_error("model.coef", "an array of finite numbers", coefficients)
    if doc["kind"] != "ridge" or len(names) != len(coefficients):
        raise InvalidConfigError("unsupported or inconsistent model")
    n_train = doc["n_train"]
    if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train < 0:
        raise _type_error("model.n_train", "a non-negative integer", n_train)
    model = RidgeModel(alpha=_finite_number(doc["alpha"], "model.alpha"))
    model.coef_ = np.asarray(
        [_finite_number(c, f"model.coef.{i}") for i, c in enumerate(coefficients)], dtype=np.float64
    )
    model.intercept_ = _finite_number(doc["intercept"], "model.intercept")
    model.n_train_ = n_train
    return model, list(names)


def load_market(cache_dirs: Sequence[str | Path], instruments: Sequence[str]) -> PreparedMarket:
    """Prepare caches in strategy order, rejecting ambiguous duplicate venues."""
    tapes = {}
    for directory in cache_dirs:
        tape = load_tapes(directory)
        key = tape.instrument.key
        if key in tapes:
            raise InvalidConfigError("duplicate cache for instrument", context={"instrument": key})
        tapes[key] = tape
    missing = [k for k in instruments if k not in tapes]
    if missing:
        raise InvalidConfigError("caches do not cover the configured instruments", context={"missing": missing})
    return PreparedMarket.build(MarketData(tuple(tapes[k] for k in instruments)))


def build_strategy_signal(prep: PreparedMarket, cfg: StrategyConfig) -> NDArray[np.float64]:
    if cfg.signal.get("kind") != "h4_ridge":
        return build_signal(prep, cfg.signal)
    _exact_keys(cfg.signal, {"kind", "leader", "follower", "gate_ticks"}, "signal")
    assert cfg.model is not None
    model, names = model_from_json(cfg.model)
    keys = list(cfg.instruments)
    follower, leader = keys.index(cfg.signal["follower"]), keys.index(cfg.signal["leader"])
    available = dict(feature_columns(prep, follower, leader))
    if any(n not in available for n in names):
        raise InvalidConfigError("model uses unknown features")
    cols = [(n, available[n]) for n in names]
    return prediction_signal(prep, model, cols, follower=follower, gate_ticks=int(cfg.signal["gate_ticks"]))


def run_strategy(
    prep: PreparedMarket, cfg: StrategyConfig, scenario: Scenario, limits: RiskLimits | None = None
) -> ReplayResult:
    """Replay a strategy against a market with the same instrument order as its policy."""
    market_keys = tuple(instrument.key for instrument in prep.data.instruments)
    if market_keys != cfg.instruments:
        raise InvalidConfigError(
            "market instrument order differs from strategy", context={"expected": cfg.instruments, "got": market_keys}
        )
    recv = prep.stream.recv_ns
    session = cfg.session_plan(int(recv[0]), int(recv[-1]))
    exit_signal = build_signal(prep, cfg.exit_signal) if cfg.exit_signal is not None else None
    return run_replay(
        prep,
        build_strategy_signal(prep, cfg),
        cfg.threshold_params(),
        scenario,
        session,
        limits or RiskLimits(),
        exit_signal=exit_signal,
    )


def summarize(result: ReplayResult, prep: PreparedMarket, cfg: StrategyConfig, scenario: Scenario) -> dict[str, Any]:
    st = result.settlement
    entries = (result.orders.reason == REASON_ENTRY) & (result.orders.filled_units > 0)
    latencies = {spec.key: scenario.latency_for(spec).to_mapping() for spec in prep.data.instruments}
    delays = {latency["order_latency_ns"] for latency in latencies.values()}
    out = {
        "strategy": cfg.name,
        "scenario": scenario.name,
        "order_latency_ms": next(iter(delays)) / _MS if len(delays) == 1 else None,
        "latency_by_instrument": latencies,
        "timing": scenario.timing.name,
        "fees": {k: str(v) for k, v in scenario.fees.items()},
        "fees_per_unit": {k: str(v) for k, v in scenario.fees_per_unit.items()},
        "instruments": list(cfg.instruments),
        "instrument_specs": [instrument.to_mapping() for instrument in prep.data.instruments],
        "scenario_config": scenario.to_mapping(),
        "risk_limits": result.limits.to_mapping() if result.limits is not None else None,
        "valid": st.valid,
        "reason": st.reason,
        "net_pnl": st.net_pnl,
        "settlement_currencies": list(st.currencies),
        "pnl_by_currency": st.pnl_by_currency,
        "cash_by_venue": list(st.cash),
        "fees_by_venue": list(st.fees),
        "final_position_units": list(st.final_position_units),
        "max_abs_position_units": list(st.max_abs_position_units),
        "traded_units": list(st.traded_units),
        "fills": len(result.fills),
        "orders": len(result.orders),
        "round_trips": int(entries.sum()),
        "decisions": result.decisions,
        "first_event_ns": int(prep.stream.recv_ns[0]),
        "last_event_ns": int(prep.stream.recv_ns[-1]),
    }
    if scenario.depth_fraction != 1.0:  # a haircut run must not read as a full-depth one; others stay unchanged
        out["depth_fraction"] = scenario.depth_fraction
    return out


def write_run_outputs(
    result: ReplayResult, prep: PreparedMarket, cfg: StrategyConfig, scenario: Scenario, out_dir: str | Path
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    ticks = [float(i.tick_size) for i in prep.data.instruments]
    keys = [i.key for i in prep.data.instruments]
    f = result.fills
    with (out / "fills.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["order_id", "venue", "side", "px_ticks", "price", "qty_units", "submit_ns", "arrival_ns", "ack_ns", "cash"]
        )
        for i in range(len(f)):
            v = int(f.venue[i])
            w.writerow(
                [
                    int(f.order_id[i]),
                    keys[v],
                    int(f.side[i]),
                    int(f.px_ticks[i]),
                    repr(int(f.px_ticks[i]) * ticks[v]),
                    int(f.qty_units[i]),
                    int(f.submit_ns[i]),
                    int(f.arrival_ns[i]),
                    int(f.ack_ns[i]),
                    repr(float(f.cash[i])),
                ]
            )
    o = result.orders
    with (out / "orders.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "order_id",
                "venue",
                "side",
                "qty_units",
                "limit_ticks",
                "submit_ns",
                "arrival_ns",
                "ack_ns",
                "filled_units",
                "reason",
                "book_row",
            ]
        )
        for i in range(len(o)):
            w.writerow(
                [
                    int(o.order_id[i]),
                    keys[int(o.venue[i])],
                    int(o.side[i]),
                    int(o.qty_units[i]),
                    int(o.limit_ticks[i]),
                    int(o.submit_ns[i]),
                    int(o.arrival_ns[i]),
                    int(o.ack_ns[i]),
                    int(o.filled_units[i]),
                    int(o.reason[i]),
                    int(o.book_row[i]),
                ]
            )
    summary = summarize(result, prep, cfg, scenario)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "strategy.json").write_text(json.dumps(cfg.to_mapping(), indent=2, sort_keys=True) + "\n")
    return summary
