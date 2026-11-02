"""Strategy configuration, model persistence and the CLI end to end on synthetic data."""

from __future__ import annotations

import csv
import dataclasses
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from financeai.application.strategy import StrategyConfig, model_from_json, model_to_json
from financeai.cli import main
from financeai.core.errors import InvalidConfigError
from financeai.domains.market.cache import SourceInfo, load_tapes, save_tapes
from financeai.models.ridge import RidgeModel

from ..market.conftest import SYNTHETIC_RECORDER_TEXT
from ..market.synthetic import random_market

CONFIG = {
    "schema": "financeai.strategy",
    "schema_version": "1.0",
    "name": "demo",
    "instruments": ["venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"],
    "signal": {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 20,
        "depth_units": 1_000,
        "gate_ticks": 2,
    },
    "policy": {
        "direction": {"venue_a/ALPHA-USD": 1, "venue_b/ALPHA-PERP": 0},
        "size_units": {"venue_a/ALPHA-USD": 1_000, "venue_b/ALPHA-PERP": 0},
        "enter": 0.5,
        "exit": -1e9,
        "allow_short": True,
        "min_hold_ms": 0,
        "max_hold_ms": 50,
        "cooldown_ms": 0,
        "stop_bps": 0.0,
        "entry_slip_ticks": 1,
        "exit_slip_ticks": 20,
    },
    "session": {"warmup_s": 0, "entry_stop_s": 0.4, "flatten_s": 0.3, "exec_stop_s": 0.06},
}


def test_config_round_trip_and_strictness() -> None:
    cfg = StrategyConfig.from_mapping(CONFIG)
    assert StrategyConfig.from_mapping(json.loads(cfg.to_json())) == cfg
    bad = dict(CONFIG, extra=1)
    with pytest.raises(InvalidConfigError):
        StrategyConfig.from_mapping(bad)
    with pytest.raises(InvalidConfigError):
        StrategyConfig.from_json('{"schema": "financeai.strategy", "schema": "x"}')


def test_ridge_model_json_round_trip_preserves_predictions() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 3)), rng.normal(size=200)
    model = RidgeModel(alpha=2.0).fit(x, y)
    again = model_from_json(model_to_json(model, ["a", "b", "c"]))
    np.testing.assert_array_equal(again[0].predict(x), model.predict(x))
    assert again[1] == ["a", "b", "c"]


def _caches(tmp: Path) -> list[str]:
    data = random_market(5)
    dirs = []
    for tapes in data.venues:
        d = tmp / tapes.instrument.symbol
        save_tapes(tapes, d, source=SourceInfo(name="synthetic", sha256="0" * 64, bytes=0))
        dirs.append(str(d))
    return dirs


def test_cli_run_writes_ledger_outputs(tmp_path: Path) -> None:
    caches = _caches(tmp_path)
    cfg_path = tmp_path / "strategy.json"
    cfg_path.write_text(json.dumps(CONFIG))
    out = tmp_path / "run"
    code = main(
        ["run", str(cfg_path), "--cache", caches[0], "--cache", caches[1], "--scenario", "base", "--out", str(out)]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["scenario"] == "base" and isinstance(summary["valid"], bool)
    with (out / "fills.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == summary["fills"]
    if rows:
        assert set(rows[0]) >= {"order_id", "venue", "side", "price", "qty_units", "cash"}


def test_cli_suite_runs_all_demo_scenarios(tmp_path: Path) -> None:
    caches = _caches(tmp_path)
    cfg_path = tmp_path / "strategy.json"
    cfg_path.write_text(json.dumps(CONFIG))
    out = tmp_path / "suite"
    assert main(["suite", str(cfg_path), "--cache", caches[0], "--cache", caches[1], "--out", str(out)]) == 0
    table = json.loads((out / "suite.json").read_text())
    assert [r["scenario"] for r in table["results"]] == ["base", "higher_fees", "longer_delay", "combined_stress"]


def test_cli_ingest_builds_a_loadable_cache(tmp_path: Path) -> None:
    raw = tmp_path / "x.txt.gz"
    raw.write_bytes(gzip.compress(SYNTHETIC_RECORDER_TEXT))
    out = tmp_path / "cache"
    assert main(["ingest", str(raw), "--instrument", "venue_a/ALPHA-USD", "--out", str(out)]) == 0
    assert len(load_tapes(out).book) == 4


def test_cli_reports_config_errors_with_a_nonzero_exit(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text("{}")
    assert main(["run", str(cfg_path), "--cache", str(tmp_path), "--out", str(tmp_path / "o")]) == 2


def test_config_supports_sizing_accumulation_and_exit_signal(tmp_path: Path) -> None:
    from financeai.application.strategy import StrategyConfig, load_market, run_strategy
    from financeai.domains.market.scenario import demo_scenarios

    doc = json.loads(json.dumps(CONFIG))
    doc["policy"].update(
        {
            "size_units_cap": 2,
            "depth_ticks": 1,
            "max_clips": 2,
            "clip_spacing_ms": 5,
            "refresh_hold": True,
            "reverse": True,
        }
    )
    doc["exit_signal"] = {
        "kind": "ewma_dev",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "tau_ms": 100,
        "depth_units": 1_000,
        "gate_ticks": 0,
    }
    cfg = StrategyConfig.from_mapping(doc)
    params = cfg.threshold_params()
    assert (params.size_units_cap, params.depth_ticks, params.max_clips, params.refresh_hold) == (2, 1, 2, True)
    assert params.reverse is True
    assert params.clip_spacing_ns == 5_000_000
    assert StrategyConfig.from_mapping(json.loads(cfg.to_json())) == cfg
    caches = _caches(tmp_path)
    prep = load_market(caches, cfg.instruments)
    res = run_strategy(prep, cfg, demo_scenarios()["base"])
    assert isinstance(res.settlement.valid, bool)


def test_config_supports_the_duration_cap_and_early_markout_check() -> None:
    doc = json.loads(json.dumps(CONFIG))
    doc["policy"].update({"abs_cap_ms": 250, "early_check_ms": 40, "early_markout_bps": -0.5})
    cfg = StrategyConfig.from_mapping(doc)
    params = cfg.threshold_params()
    assert (params.abs_cap_ns, params.early_check_ns, params.early_markout_bps) == (250_000_000, 40_000_000, -0.5)
    assert StrategyConfig.from_mapping(json.loads(cfg.to_json())) == cfg
    plain = StrategyConfig.from_mapping(CONFIG).threshold_params()  # absent keys leave the options off
    assert (plain.abs_cap_ns, plain.early_check_ns, plain.early_markout_bps) == (0, 0, 0.0)
    for key, bad in (("abs_cap_ms", -1), ("early_check_ms", -5), ("early_markout_bps", float("inf")), ("abs_cap_s", 1)):
        broken = json.loads(json.dumps(CONFIG))
        broken["policy"][key] = bad
        with pytest.raises(InvalidConfigError):
            StrategyConfig.from_mapping(broken)


def test_run_summary_records_a_depth_haircut_only_when_one_is_used(tmp_path: Path) -> None:
    import dataclasses

    from financeai.application.strategy import load_market, run_strategy, summarize
    from financeai.domains.market.scenario import demo_scenarios

    cfg = StrategyConfig.from_mapping(CONFIG)
    prep = load_market(_caches(tmp_path), cfg.instruments)
    base = demo_scenarios()["base"]
    assert "depth_fraction" not in summarize(run_strategy(prep, cfg, base), prep, cfg, base)  # unchanged outputs
    thin = dataclasses.replace(base, depth_fraction=0.5)
    assert summarize(run_strategy(prep, cfg, thin), prep, cfg, thin)["depth_fraction"] == 0.5


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("reverse", "false"),
        ("refresh_hold", 1),
        ("allow_short", "true"),
        ("allow_short", 0),
        ("max_clips", 2.9),
        ("max_clips", True),
        ("size_units_cap", "2"),
        ("min_hold_ms", 10.0),
        ("max_hold_ms", "10000"),
        ("cooldown_ms", False),
        ("entry_slip_ticks", 1.0),
        ("exit_slip_ticks", "20"),
        ("depth_ticks", 1.0),
        ("clip_spacing_ms", 0.5),
        ("abs_cap_ms", 250.0),
        ("early_check_ms", "40"),
        ("enter", "0.5"),
        ("enter", True),
        ("exit", float("nan")),
        ("exit", 10**400),
        ("stop_bps", float("inf")),
        ("early_markout_bps", "-0.5"),
        ("early_markout_bps", None),
        ("max_clips", np.int64(2)),
        ("min_hold_ms", np.int32(0)),  # numpy scalars are not JSON integers: int() first
        ("direction", {"venue_a/ALPHA-USD": 0.9, "venue_b/ALPHA-PERP": 0}),
        ("direction", {"venue_a/ALPHA-USD": 1.0, "venue_b/ALPHA-PERP": 0}),
        ("direction", {"venue_a/ALPHA-USD": True, "venue_b/ALPHA-PERP": 0}),
        ("direction", [1, 0]),
        ("size_units", {"venue_a/ALPHA-USD": 1000.0, "venue_b/ALPHA-PERP": 0}),
        ("size_units", {"venue_a/ALPHA-USD": "1000", "venue_b/ALPHA-PERP": 0}),
    ],
)
def test_policy_values_are_checked_not_coerced(key: str, bad: object) -> None:
    doc = json.loads(json.dumps(CONFIG))
    doc["policy"][key] = bad
    with pytest.raises(InvalidConfigError) as info:
        StrategyConfig.from_mapping(doc)
    assert str(info.value.context["field"]).startswith(f"policy.{key}")


def test_equivalent_spellings_of_float_fields_stay_accepted() -> None:
    ints = json.loads(json.dumps(CONFIG))
    ints["policy"].update({"exit": -1_000_000_000, "enter": 1, "stop_bps": 0, "early_markout_bps": -1})
    floats = json.loads(json.dumps(CONFIG))
    floats["policy"].update({"exit": -1_000_000_000.0, "enter": 1.0, "stop_bps": 0.0, "early_markout_bps": -1.0})
    a, b = StrategyConfig.from_mapping(ints).threshold_params(), StrategyConfig.from_mapping(floats).threshold_params()
    assert a == b and (a.exit, a.enter, a.stop_bps, a.early_markout_bps) == (-1e9, 1.0, 0.0, -1.0)
    assert all(type(v) is float for v in (a.exit, a.enter, a.stop_bps, a.early_markout_bps))


def test_threshold_params_is_strict_for_directly_built_configs() -> None:
    cfg = StrategyConfig.from_mapping(CONFIG)
    lossy = dataclasses.replace(cfg, policy={**cfg.policy, "max_clips": 2.9})
    with pytest.raises(InvalidConfigError) as info:
        lossy.threshold_params()
    assert info.value.context["field"] == "policy.max_clips"
    missing = dataclasses.replace(cfg, policy={k: v for k, v in cfg.policy.items() if k != "enter"})
    with pytest.raises(InvalidConfigError) as info:
        missing.threshold_params()
    assert info.value.context["field"] == "policy.enter"
    # from_mapping checks the instrument set; a directly built policy must still name every instrument
    short = dataclasses.replace(cfg, policy={**cfg.policy, "direction": {"venue_a/ALPHA-USD": 1}})
    with pytest.raises(InvalidConfigError) as info:
        short.threshold_params()
    assert info.value.context["field"] == "policy.direction.venue_b/ALPHA-PERP"


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("exit_signal", 5),
        ("model", "ridge"),
        ("signal", ["lead_gap"]),
        ("policy", 3),
        ("session", None),
        ("instruments", "venue_a/ALPHA-USD"),
        ("instruments", [1, 2]),
    ],
)
def test_sections_must_be_objects_and_instruments_an_array_of_keys(key: str, bad: object) -> None:
    doc: dict[str, Any] = dict(CONFIG)
    doc[key] = bad
    with pytest.raises(InvalidConfigError) as info:
        StrategyConfig.from_mapping(doc)
    assert info.value.context["field"] == key


def test_null_model_and_exit_signal_mean_absent() -> None:
    nulls = StrategyConfig.from_mapping({**CONFIG, "model": None, "exit_signal": None})
    assert nulls == StrategyConfig.from_mapping(CONFIG)


def test_export_preserves_the_complete_execution_configuration(tmp_path: Path) -> None:
    from decimal import Decimal

    from financeai.application.strategy import load_market, run_strategy, write_run_outputs
    from financeai.domains.market.instruments import InstrumentSpec
    from financeai.domains.market.scenario import Scenario, demo_scenarios

    cfg = StrategyConfig.from_mapping(CONFIG)
    prep = load_market(_caches(tmp_path), cfg.instruments)
    scenario = dataclasses.replace(
        demo_scenarios(prep.data.instruments)["base"],
        ack_latency_ns=7_000_000,
        consumption_ttl_ns=2_000_000,
        depth_fraction=0.5,
        fees_per_unit={spec.key: Decimal("0.00000001") for spec in prep.data.instruments},
    )
    out = tmp_path / "export"
    write_run_outputs(run_strategy(prep, cfg, scenario), prep, cfg, scenario, out)
    summary = json.loads((out / "summary.json").read_text())
    assert Scenario.from_mapping(summary["scenario_config"]) == scenario
    assert tuple(InstrumentSpec.from_mapping(value) for value in summary["instrument_specs"]) == prep.data.instruments


def test_cli_accepts_market_specific_risk_and_latency_files(tmp_path: Path) -> None:
    from decimal import Decimal

    from financeai.domains.market import InstrumentLatency, RiskLimits, Scenario
    from financeai.domains.market.instruments import INSTRUMENTS

    caches = _caches(tmp_path)
    config_path = tmp_path / "strategy.json"
    config_path.write_text(json.dumps(CONFIG))
    key = CONFIG["instruments"][0]
    limits = RiskLimits(1000, per_instrument={key: 100})
    scenario = Scenario(
        "custom",
        {i: Decimal("0.0001") for i in CONFIG["instruments"]},
        1_000_000,
        latency_by_instrument={key: InstrumentLatency(2_000_000, 3_000_000)},
    )
    risk_path, scenario_path = tmp_path / "risk.json", tmp_path / "scenario.json"
    risk_path.write_text(json.dumps(limits.to_mapping()))
    scenario_path.write_text(json.dumps(scenario.to_mapping()))
    out = tmp_path / "custom-run"
    assert (
        main(
            [
                "run",
                str(config_path),
                "--cache",
                caches[0],
                "--cache",
                caches[1],
                "--risk-file",
                str(risk_path),
                "--scenario-file",
                str(scenario_path),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["risk_limits"] == limits.to_mapping()
    assert summary["scenario_config"] == scenario.to_mapping()
    assert summary["max_abs_position_units"][0] <= limits.cap_for(INSTRUMENTS[key])


@pytest.mark.parametrize(
    "text",
    [
        '{"max_abs_position_units":10,"max_abs_position_units":1000}',
        '{"per_instrument":{"exchange/ASSET":10,"exchange/ASSET":1000}}',
        '{"depth_fraction":NaN}',
        '{"depth_fraction":1e400}',
    ],
)
def test_cli_configuration_decoder_rejects_ambiguous_or_non_finite_json(tmp_path: Path, text: str) -> None:
    from financeai.cli import _json

    path = tmp_path / "configuration.json"
    path.write_text(text)
    with pytest.raises(InvalidConfigError):
        _json(str(path))
