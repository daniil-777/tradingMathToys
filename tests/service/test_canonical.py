"""Strict canonical strategies at the service boundary: lossy or mistyped values are rejected with typed
errors, equivalent numeric spellings hash identically and the recorded strategy is exactly what runs."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from financeai.application.strategy import _POLICY, _POLICY_OPTIONAL, StrategyConfig
from financeai.core.errors import FinanceAIError, InvalidConfigError
from financeai.core.serialization import canonical_json, loads_strict
from financeai.domains.market.synthetic import SyntheticMarketSpec
from financeai.service import canonical
from financeai.service.artifacts import ArtifactStore, register_strategy
from financeai.service.canonical import (
    MAX_HOLD_MS,
    MAX_INT_PARAMETER,
    MAX_MULTIPLIER,
    MAX_SESSION_S,
    MAX_SIGNAL_DEPTH,
    canonical_strategy,
)
from financeai.service.jobs import (
    SyntheticReplayJob,
    job_from_mapping,
    request_hash,
    run_strategy_replay,
    run_synthetic_replay,
)
from financeai.service.models import EXAMPLE_STRATEGY
from financeai.service.store import JobStore
from financeai.strategies import leadlag

from .support import synthetic_job
from .test_jobs import write_caches

VENUE_A, VENUE_B = "venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"
MS = 1_000_000
LEAD_GAP: dict[str, Any] = {
    "kind": "lead_gap",
    "leader": VENUE_B,
    "follower": VENUE_A,
    "window_ms": 20,
    "depth_units": 1000,
    "gate_ticks": 2,
}
MOVE: dict[str, Any] = {"kind": "move", "venue": VENUE_B, "window_ms": 20, "depth_units": 1000, "gate_ticks": 2}
QUOTE_CONFIRM: dict[str, Any] = {"kind": "quote_confirm", "venue": VENUE_B, "window_ms": 20, "ticks": 2}
OFI: dict[str, Any] = {
    "kind": "ofi",
    "venue": VENUE_B,
    "levels": 3,
    "window_ms": 20,
    "normalize": "none",
    "measure": "units",
}
"""Flow recipes whose integer counts (``ticks``, ``levels``) carry no unit suffix."""
OPTIONS: dict[str, Any] = {
    "size_units_cap": 2,
    "depth_ticks": 3,
    "max_clips": 2,
    "clip_spacing_ms": 5,
    "refresh_hold": True,
    "reverse": True,
    "abs_cap_ms": 30,
    "early_check_ms": 10,
    "early_markout_bps": -2.0,
}
"""Every optional policy key of ``StrategyConfig``, as research candidates export them."""
EXAMPLE_LEAVES: dict[str, Any] = {
    "leader": VENUE_B,
    "follower": VENUE_A,
    "venue": VENUE_B,
    "source": VENUE_B,
    "window_ms": 20,
    "tau_ms": 1000,
    "depth_units": 1000,
    "min_units": 1000,
    "gate_ticks": 2,
    "ticks": 2,
    "levels": 3,
    "normalize": "none",
    "measure": "units",
}
"""One valid value per key of the registered flat recipes (``strategies.leadlag._SPEC_KEYS``)."""


def composite(*parts: tuple[dict[str, Any], Any], kind: str = "any") -> dict[str, Any]:
    """A composite signal recipe (``strategies.leadlag``): components with their scales."""
    return {"kind": kind, "components": [{"spec": spec, "scale": scale} for spec, scale in parts]}


def with_signal(signal: Any) -> dict[str, Any]:
    doc = strategy()
    doc["signal"] = signal
    return doc


def strategy(**changes: Any) -> dict[str, Any]:
    """The example strategy with top-level replacements; dict values are merged into dict sections."""
    doc: dict[str, Any] = copy.deepcopy(EXAMPLE_STRATEGY)
    for key, value in changes.items():
        doc[key] = {**doc[key], **value} if isinstance(doc.get(key), dict) and isinstance(value, dict) else value
    return doc


def job_with(**changes: Any) -> dict[str, Any]:
    return synthetic_job(strategy=strategy(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"policy": {"direction": {VENUE_A: 0.9, VENUE_B: 0}}},  # int() would run it as 0
        {"policy": {"size_units": {VENUE_A: 1000.9, VENUE_B: 0}}},  # int() would run it as 1000
        {"policy": {"size_units": {VENUE_A: 1000.0, VENUE_B: 0}}},  # integer fields take JSON integers only
        {"policy": {"direction": {VENUE_A: True, VENUE_B: 0}}},
        {"policy": {"allow_short": 1}},
        {"policy": {"allow_short": "false"}},  # bool("false") is True
        {"policy": {"enter": "0.5"}},
        {"policy": {"stop_bps": False}},
        {"policy": {"exit": -(2**53 + 1)}},  # not exactly representable as binary64
        {"policy": {"exit": 10**400}},  # float() would overflow
        {"policy": {"min_hold_ms": 1.5}},
        {"policy": {"cooldown_ms": 10.0}},
        {"policy": {"exit_slip_ticks": "20"}},
        {"policy": {"cooldown_ms": MAX_INT_PARAMETER + 1}},
        {"policy": {"size_units": {VENUE_A: MAX_INT_PARAMETER + 1, VENUE_B: 0}}},
        {"signal": {"window_ms": 20.0}},
        {"signal": {"gate_ticks": True}},
        {"signal": {"depth_units": "1000"}},
        {"signal": {"leader_delay_ms": 0.5}},
        {"signal": {"window_ms": -20}},  # a negative window would look into the future
        {"session": {"warmup_s": "0"}},
        {"session": {"exec_stop_s": True}},
        {"session": {"warmup_s": MAX_SESSION_S + 1}},
        # optional policy keys of StrategyConfig (int()/bool() coercions in threshold_params)
        {"policy": {"reverse": "false"}},  # bool("false") is True
        {"policy": {"refresh_hold": 1}},
        {"policy": {"max_clips": 2.5}},  # int() would run it as 2
        {"policy": {"max_clips": True}},  # int(True) is 1
        {"policy": {"size_units_cap": 2.0}},
        {"policy": {"clip_spacing_ms": 0.5}},  # int() would run it as 0 ms
        {"policy": {"depth_ticks": "3"}},
        {"policy": {"depth_ticks": -2}},  # -1 disables depth sizing; nothing lower exists
        {"policy": {"max_clips": 0}},
        {"policy": {"size_units_cap": MAX_MULTIPLIER + 1}},  # units * clips * size_units must stay inside int64
        {"policy": {"max_clips": MAX_MULTIPLIER + 1}},
        {"policy": {"max_hold_ms": MAX_HOLD_MS + 1}},
        # episode-clock knobs (c5b8b92) were passed through untyped: 2.5 ran as 2 ms, "5" as 5 ms, True as 1 ms
        {"policy": {"abs_cap_ms": 2.5}},
        {"policy": {"abs_cap_ms": True}},
        {"policy": {"early_check_ms": "5"}},
        {"policy": {"early_check_ms": -1}},
        {"policy": {"abs_cap_ms": MAX_INT_PARAMETER + 1}},  # 10**12 ms = 10**18 ns keeps the kernel argument in int64
        {"policy": {"abs_cap_ms": 10**15}},  # was admitted, then FAILED in the worker (numba: int value too large)
        {"policy": {"early_markout_bps": "1"}},
        {"policy": {"early_markout_bps": 2**53 + 1}},
        {"exit_signal": {**LEAD_GAP, "window_ms": 20.0}},  # the exit recipe follows the signal rules
    ],
)
def test_lossy_or_mistyped_values_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(InvalidConfigError) as caught:
        job_from_mapping(job_with(**changes))
    section = next(iter(changes))
    assert str(caught.value.context["field"]).startswith(section + ".")


@pytest.mark.parametrize(
    "changes",
    [
        {"policy": {"enter": None}},
        {"policy": {"direction": {VENUE_A: "x", VENUE_B: 0}}},
        {"policy": {"direction": 5}},
        {"policy": {"min_hold_ms": [1]}},
        {"policy": [1, 2]},
        {"session": "abc"},
        {"signal": 5},
        {"signal": {"leader": None}},
        {"instruments": 5},
        {"instruments": VENUE_A},
        {"instruments": [VENUE_A, 5]},
        {"name": 5},
        {"schema_version": 1.0},
        {"model": [1]},
        {"model": {"kind": "ridge", "alpha": "1", "features": [], "coef": [], "intercept": 0.0, "n_train": 2}},
        {"model": {"kind": "ridge", "alpha": 1.0, "features": ["a"], "coef": [0.5], "intercept": 0.0, "n_train": 2.5}},
        {
            "model": {
                "kind": "ridge",
                "alpha": 1.0,
                "features": ["a"],
                "coef": [0.5, 1.0],
                "intercept": 0.0,
                "n_train": 2,
            }
        },
        {"exit_signal": 5},
        {"exit_signal": [LEAD_GAP]},
        {"signal": {"gate": [1]}},
        {"signal": {"fresh": None}},
    ],
)
def test_malformed_shapes_raise_typed_errors_not_builtin_exceptions(changes: dict[str, Any]) -> None:
    with pytest.raises(FinanceAIError):  # previously TypeError/ValueError, i.e. an HTTP 500
        job_from_mapping(job_with(**changes))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ({"policy": {"exit": -1000000000}}, {"policy": {"exit": -1000000000.0}}),
        ({"policy": {"stop_bps": 0}}, {"policy": {"stop_bps": 0.0}}),
        ({"policy": {"enter": 1}}, {"policy": {"enter": 1.0}}),
        ({"policy": {"exit": -0.0}}, {"policy": {"exit": 0}}),
        ({"policy": {"early_markout_bps": -2}}, {"policy": {"early_markout_bps": -2.0}}),
        ({"session": {"warmup_s": 0, "entry_stop_s": 0.4}}, {"session": {"warmup_s": 0.0, "entry_stop_s": 0.4}}),
        ({"schema_version": "1.0"}, {"schema_version": "1.2"}),
        ({"model": None}, {}),
        ({"exit_signal": None}, {}),
    ],
)
def test_equivalent_spellings_have_one_canonical_form_and_hash(a: dict[str, Any], b: dict[str, Any]) -> None:
    ja, jb = job_from_mapping(job_with(**a)), job_from_mapping(job_with(**b))
    assert canonical_json(ja.to_mapping()) == canonical_json(jb.to_mapping())
    assert request_hash("c", ja) == request_hash("c", jb)


def test_the_canonical_mapping_carries_exactly_the_engine_types() -> None:
    job = job_from_mapping(job_with(policy={"exit": -1000000000, "stop_bps": 0}, session={"warmup_s": 0}))
    assert isinstance(job, SyntheticReplayJob)
    doc = job.to_mapping()["strategy"]
    policy, session = doc["policy"], doc["session"]
    assert all(type(v) is int for m in (policy["direction"], policy["size_units"]) for v in m.values())
    assert all(type(policy[k]) is float for k in ("enter", "exit", "stop_bps"))
    assert all(
        type(policy[k]) is int
        for k in ("min_hold_ms", "max_hold_ms", "cooldown_ms", "entry_slip_ticks", "exit_slip_ticks")
    )
    assert policy["allow_short"] is True
    assert all(type(v) is float for v in session.values())
    assert all(type(v) is int for k, v in doc["signal"].items() if k not in ("kind", "leader", "follower"))
    reread = StrategyConfig.from_mapping(loads_strict(canonical_json(doc)))
    assert reread.threshold_params() == job.strategy.threshold_params()
    assert canonical_json(reread.to_mapping()) == canonical_json(doc)  # canonical form is a fixed point
    assert job_from_mapping(job.to_mapping()) == job


@pytest.mark.parametrize(
    ("signal", "field"),
    [
        ({**LEAD_GAP, "gate": {"venue": VENUE_A, "tau_ms": 1.5, "min_units": 1000}}, "signal.gate.tau_ms"),
        ({**LEAD_GAP, "gate": {"venue": VENUE_A, "tau_ms": 1000, "min_units": 1000.5}}, "signal.gate.min_units"),
        ({**LEAD_GAP, "gate": {"venue": 1, "tau_ms": 1000, "min_units": 1000}}, "signal.gate.venue"),
        ({**LEAD_GAP, "fresh": {"max_age_ms": "50"}}, "signal.fresh.max_age_ms"),
        (composite((LEAD_GAP, 1.0), ({**MOVE, "window_ms": 20.0}, 2.0)), "signal.components.1.spec.window_ms"),
        (composite((LEAD_GAP, "1")), "signal.components.0.scale"),
        (composite((LEAD_GAP, True)), "signal.components.0.scale"),
        (composite((LEAD_GAP, 2**53 + 1)), "signal.components.0.scale"),  # not exactly a binary64
        ({"kind": "any", "components": LEAD_GAP}, "signal.components"),
        ({"kind": "any", "components": [5]}, "signal.components.0"),
        ({**QUOTE_CONFIRM, "ticks": 2.0}, "signal.ticks"),
        ({**OFI, "levels": "3"}, "signal.levels"),
        (composite(({**OFI, "levels": True}, 1.0)), "signal.components.0.spec.levels"),
    ],
)
def test_nested_signal_leaves_are_checked_and_named(signal: dict[str, Any], field: str) -> None:
    with pytest.raises(InvalidConfigError) as caught:
        job_from_mapping(synthetic_job(strategy=with_signal(signal)))
    assert caught.value.context["field"] == field


def test_signal_nesting_is_bounded() -> None:
    spec = LEAD_GAP
    for _ in range(MAX_SIGNAL_DEPTH // 2):  # each composite level nests a component and its spec
        spec = composite((spec, 1.0))
    canonical_strategy(with_signal(spec))  # at the bound: accepted
    with pytest.raises(InvalidConfigError) as caught:
        canonical_strategy(with_signal(composite((spec, 1.0))))
    assert str(caught.value.context["field"]).startswith("signal.components.0.spec.")


def test_policy_options_and_an_exit_signal_are_canonical_and_run(tmp_path: Path) -> None:
    exit_signal = {**LEAD_GAP, "window_ms": 50}
    job = job_from_mapping(synthetic_job(seed=3, strategy=strategy(policy=OPTIONS, exit_signal=exit_signal)))
    assert isinstance(job, SyntheticReplayJob)
    doc = job.to_mapping()["strategy"]
    assert {k: doc["policy"][k] for k in OPTIONS} == OPTIONS
    assert all(type(doc["policy"][k]) is type(v) for k, v in OPTIONS.items())  # bool stays bool, int stays int
    assert doc["exit_signal"] == exit_signal
    p = job.strategy.threshold_params()
    assert (p.size_units_cap, p.depth_ticks, p.max_clips, p.clip_spacing_ns, p.refresh_hold, p.reverse) == (
        2,
        3,
        2,
        5 * MS,
        True,
        True,
    )
    assert (p.abs_cap_ns, p.early_check_ns, p.early_markout_bps) == (30 * MS, 10 * MS, -2.0)
    assert canonical_json(StrategyConfig.from_mapping(doc).to_mapping()) == canonical_json(doc)  # fixed point
    run_synthetic_replay(job, tmp_path / "out")
    published = loads_strict((tmp_path / "out" / "run" / "strategy.json").read_bytes())
    assert canonical_json(published) == canonical_json(doc)


def test_the_never_hold_of_exported_candidates_is_accepted() -> None:
    never_ms = 2**62 // MS  # ThresholdParams' default max_hold_ns, as research candidates write it
    assert never_ms == MAX_HOLD_MS > MAX_INT_PARAMETER
    cfg = canonical_strategy(strategy(policy={**OPTIONS, "max_hold_ms": never_ms}))
    assert cfg.threshold_params().max_hold_ns == never_ms * MS < 2**63  # compared with elapsed time only


def test_composite_gated_and_fresh_recipes_are_accepted_and_run(tmp_path: Path) -> None:
    gated = {**LEAD_GAP, "gate": {"venue": VENUE_B, "tau_ms": 1000, "min_units": 0}, "fresh": {"max_age_ms": 500}}
    doc = with_signal(composite((gated, 2), ({**MOVE, "normalize_tau_ms": 1000}, 1.5)))
    job = job_from_mapping(synthetic_job(seed=3, strategy=doc))
    signal = job.to_mapping()["strategy"]["signal"]
    assert [c["scale"] for c in signal["components"]] == [2.0, 1.5]
    assert all(type(c["scale"]) is float for c in signal["components"])
    assert signal["components"][0]["spec"] == gated  # nested recipes keep their exact integers
    twin = copy.deepcopy(doc)
    twin["signal"]["components"][0]["scale"] = 2.0
    assert request_hash("c", job) == request_hash("c", job_from_mapping(synthetic_job(seed=3, strategy=twin)))
    assert isinstance(job, SyntheticReplayJob)
    run_synthetic_replay(job, tmp_path / "out")  # the worker builds the composite with strategies.leadlag
    published = loads_strict((tmp_path / "out" / "run" / "strategy.json").read_bytes())
    assert canonical_json(published) == canonical_json(job.to_mapping()["strategy"])


def test_the_canonicaliser_types_every_policy_key_of_strategy_config() -> None:
    """A new ``ThresholdParams`` knob must be typed in the canonicaliser before the service admits it: an
    untyped key is passed through, i.e. hashed and published as written but run through ``int()``/``float()``
    (``abs_cap_ms`` 2.5 ran as 2 ms and 10**15 crashed the worker instead of returning 422)."""
    typed = (
        {name for name, _, _ in canonical._POLICY_INTS}
        | set(canonical._POLICY_FLOATS)
        | set(canonical._POLICY_BOOLS)
        | {"direction", "size_units"}
    )
    assert typed == _POLICY | _POLICY_OPTIONAL
    assert set(OPTIONS) == _POLICY_OPTIONAL  # and every optional key is exampled end to end above


@pytest.mark.parametrize(("signal", "key"), [(QUOTE_CONFIRM, "ticks"), (OFI, "levels")])
def test_unsuffixed_integer_recipe_keys_are_integers(signal: dict[str, Any], key: str) -> None:
    """``quote_confirm.ticks`` and ``ofi.levels`` are integers to ``strategies.leadlag`` although they carry no
    unit suffix; the string fallback used to reject both recipes ("must be a string")."""
    cfg = canonical_strategy(with_signal(signal))
    assert cfg.signal == signal and type(cfg.signal[key]) is int
    assert canonical_strategy(with_signal(composite((signal, 1.0)))).signal["components"][0]["spec"] == signal
    for lossy in (2.0, "2", True):
        with pytest.raises(InvalidConfigError) as caught:
            canonical_strategy(with_signal({**signal, key: lossy}))
        assert caught.value.context == {
            "field": f"signal.{key}",
            "expected": "a JSON integer",
            "got": type(lossy).__name__,
        }
    with pytest.raises(InvalidConfigError) as caught:
        canonical_strategy(with_signal(composite(({**signal, key: -1}, 1.0))))
    assert caught.value.context["field"] == f"signal.components.0.spec.{key}"


@pytest.mark.parametrize("kind", sorted(leadlag._SPEC_KEYS))
def test_every_registered_recipe_kind_is_canonical_and_runs(kind: str, tmp_path: Path) -> None:
    """One valid recipe per registered flat kind, with every optional stage, passes the canonicaliser with its
    exact leaves, is a fixed point and runs in the worker. A recipe key this table does not know fails here,
    so the canonicaliser learns new keys when ``strategies.leadlag`` does."""
    keys = leadlag._SPEC_KEYS[kind] - {"kind"}
    assert keys <= set(EXAMPLE_LEAVES), f"{kind}: no example value for {sorted(keys - set(EXAMPLE_LEAVES))}"
    recipe: dict[str, Any] = {
        "kind": kind,
        **{k: EXAMPLE_LEAVES[k] for k in sorted(keys)},
        "gate": {"venue": VENUE_B, "tau_ms": 1000, "min_units": 0},
        "fresh": {"max_age_ms": 500},
        "normalize_tau_ms": 1000,
    }
    if kind == "lead_gap":
        recipe["leader_delay_ms"] = 5
    job = job_from_mapping(synthetic_job(seed=3, strategy=with_signal(recipe)))
    assert isinstance(job, SyntheticReplayJob)
    doc = job.to_mapping()["strategy"]
    assert doc["signal"] == recipe
    assert all(type(doc["signal"][k]) is type(v) for k, v in recipe.items())  # ints stay ints, strings strings
    assert canonical_json(StrategyConfig.from_mapping(doc).to_mapping()) == canonical_json(doc)  # fixed point
    run_synthetic_replay(job, tmp_path / "out")
    published = loads_strict((tmp_path / "out" / "run" / "strategy.json").read_bytes())
    assert published["signal"] == recipe


def test_bounds_are_inclusive() -> None:
    cfg = canonical_strategy(
        strategy(
            policy={"cooldown_ms": MAX_INT_PARAMETER, "max_hold_ms": MAX_INT_PARAMETER, "exit": -(2**53)},
            session={"warmup_s": MAX_SESSION_S},
        )
    )
    assert cfg.policy["cooldown_ms"] == MAX_INT_PARAMETER and cfg.policy["exit"] == -(2.0**53)
    assert cfg.session["warmup_s"] == float(MAX_SESSION_S)


def test_the_published_strategy_is_exactly_the_command_that_ran(tmp_path: Path) -> None:
    job = job_from_mapping(
        synthetic_job(seed=3, strategy=strategy(policy={"exit": -1000000000}, session={"warmup_s": 0}))
    )
    assert isinstance(job, SyntheticReplayJob)
    run_synthetic_replay(job, tmp_path / "out")
    published = loads_strict((tmp_path / "out" / "run" / "strategy.json").read_bytes())
    assert canonical_json(published) == canonical_json(job.to_mapping()["strategy"])
    assert type(published["policy"]["exit"]) is float and type(published["session"]["warmup_s"]) is float


def test_direct_construction_is_checked_and_canonicalized() -> None:
    spec = SyntheticMarketSpec(seed=1)
    lossy_policy = {**strategy()["policy"], "direction": {VENUE_A: 0.9, VENUE_B: 0}}
    with pytest.raises(InvalidConfigError):  # the application validator is strict: 0.9 never runs as 0
        StrategyConfig.from_mapping(strategy(policy=lossy_policy))
    lossy = dataclasses.replace(StrategyConfig.from_mapping(strategy()), policy=lossy_policy)
    with pytest.raises(InvalidConfigError):
        lossy.threshold_params()
    with pytest.raises(InvalidConfigError):
        SyntheticReplayJob(market=spec, scenario="base", strategy=lossy)
    loose = StrategyConfig.from_mapping(strategy(policy={"exit": -1000000000}, session={"warmup_s": 0}))
    job = SyntheticReplayJob(market=spec, scenario="base", strategy=loose)
    assert type(job.strategy.policy["exit"]) is float and type(job.strategy.session["warmup_s"]) is float


def test_registration_and_strategy_replay_refuse_lossy_files(tmp_path: Path) -> None:
    lossy = tmp_path / "lossy.json"
    lossy.write_text(json.dumps(strategy(policy={"size_units": {VENUE_A: 1000.9, VENUE_B: 0}})))
    store = JobStore(tmp_path / "state" / "service.sqlite3")
    try:
        artifacts = ArtifactStore(tmp_path / "state" / "artifacts")
        with pytest.raises(InvalidConfigError):
            register_strategy(store, artifacts, lossy)
        assert not (artifacts.root / "inputs").exists()
        loose = tmp_path / "loose.json"  # lossless but not canonical: registered with its exact bytes
        loose.write_text(json.dumps(strategy(policy={"exit": -1000000000}, session={"warmup_s": 0})))
        record = register_strategy(store, artifacts, loose)
        assert artifacts.verify(record).read_bytes() == loose.read_bytes()
    finally:
        store.close()
    caches = write_caches(SyntheticMarketSpec(seed=7), tmp_path / "caches")
    with pytest.raises(InvalidConfigError):
        run_strategy_replay(lossy, caches, "base", tmp_path / "lossy-run")
    run_strategy_replay(loose, caches, "base", tmp_path / "loose-run")
    published = loads_strict((tmp_path / "loose-run" / "run" / "strategy.json").read_bytes())
    assert type(published["policy"]["exit"]) is float  # the run records the canonical form it executed
