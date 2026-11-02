"""Grid evaluation: fee re-scoring must equal a replay under that fee schedule."""

from __future__ import annotations

import numpy as np
import pytest

from financeai.application.grid import GridExperiment, Window, evaluate_experiment, rescore
from financeai.domains.market.engine import PreparedMarket, ThresholdParams, run_replay
from financeai.domains.market.scenario import (
    DEMO_BASE_FEES,
    DEMO_HIGH_FEES,
    RiskLimits,
    Scenario,
    SessionPlan,
)

from ..market.synthetic import random_market

MS = 1_000_000


def test_rescore_equals_replaying_with_other_fees() -> None:
    prep = PreparedMarket.build(random_market(21))
    signal = np.random.default_rng(3).normal(size=len(prep.stream))
    first, last = int(prep.stream.recv_ns[0]), int(prep.stream.recv_ns[-1])
    session = SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    params = ThresholdParams(direction=(1, -1), size_units=(2_000, 2_000), enter=0.8, exit=0.0, max_hold_ns=100 * MS)
    base = run_replay(prep, signal, params, Scenario("b", DEMO_BASE_FEES, 5 * MS), session, RiskLimits())
    high = run_replay(prep, signal, params, Scenario("h", DEMO_HIGH_FEES, 5 * MS), session, RiskLimits())
    instruments = prep.data.instruments
    again = rescore(base.fills, instruments, [DEMO_HIGH_FEES[i.key] for i in instruments], 1_000_000)
    assert again == high.settlement


def test_block_attribution_partitions_the_full_day_result() -> None:
    from financeai.application.grid import attribute_episodes

    prep = PreparedMarket.build(random_market(23))
    signal = np.random.default_rng(4).normal(size=len(prep.stream))
    first, last = int(prep.stream.recv_ns[0]), int(prep.stream.recv_ns[-1])
    session = SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    params = ThresholdParams(direction=(1, 1), size_units=(500, 500), enter=0.6, exit=0.0, max_hold_ns=40 * MS)
    res = run_replay(prep, signal, params, Scenario("b", DEMO_BASE_FEES, 5 * MS), session, RiskLimits())
    mid = (first + last) // 2
    edges = [("A", first, mid), ("B", mid, last + 1)]
    nets, counts = attribute_episodes(res.fills, len(prep.data), edges)
    assert res.settlement.valid and len(res.fills) > 0
    assert sum(nets) == pytest.approx(res.settlement.net_pnl, rel=1e-12, abs=1e-15)
    assert sum(counts) >= 1


def test_evaluate_experiment_with_attribution_emits_block_rows() -> None:
    prep = PreparedMarket.build(random_market(24))
    first, last = int(prep.stream.recv_ns[0]), int(prep.stream.recv_ns[-1])
    windows = [Window("FULL", SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last))]
    signal = np.random.default_rng(5).normal(size=len(prep.stream))
    experiment = GridExperiment(
        "TEST",
        {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 100},
        ThresholdParams(direction=(1, 0), size_units=(1_000, 0), enter=0.8, exit=-1e9, max_hold_ns=30 * MS),
        latency_ms=5,
    )
    mid = (first + last) // 2
    rows = evaluate_experiment(
        prep,
        signal,
        experiment,
        windows,
        {"base": DEMO_BASE_FEES},
        1_000_000,
        attribution=[("B1", first, mid), ("B2", mid, last + 1)],
    )
    by_window = {r["window"]: r for r in rows}
    assert set(by_window) == {"FULL", "B1", "B2"}
    assert by_window["B1"]["net_pnl"] + by_window["B2"]["net_pnl"] == pytest.approx(by_window["FULL"]["net_pnl"])
    assert by_window["B1"]["valid"] == by_window["FULL"]["valid"]


def test_evaluate_experiment_reports_every_window_and_fee_set() -> None:
    prep = PreparedMarket.build(random_market(22))
    first, last = int(prep.stream.recv_ns[0]), int(prep.stream.recv_ns[-1])
    windows = [Window("ALL", SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last))]
    experiment = GridExperiment(
        "TEST",
        {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 100},
        ThresholdParams(direction=(1, 0), size_units=(1_000, 0), enter=0.5, exit=-1e9, max_hold_ns=50 * MS),
        latency_ms=5,
    )
    signal = np.zeros(len(prep.stream))
    rows = evaluate_experiment(
        prep, signal, experiment, windows, {"base": DEMO_BASE_FEES, "high": DEMO_HIGH_FEES}, 1_000_000
    )
    assert [(r["window"], r["fees"]) for r in rows] == [("ALL", "base"), ("ALL", "high")]
    assert all(r["valid"] and r["net_pnl"] == 0.0 and r["round_trips"] == 0 for r in rows)


def test_grid_task_with_exit_signal_matches_a_direct_replay() -> None:
    from financeai.strategies.leadlag import build_signal

    prep = PreparedMarket.build(random_market(25))
    first, last = int(prep.stream.recv_ns[0]), int(prep.stream.recv_ns[-1])
    plan = SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    entry = {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 20,
        "depth_units": 1_000,
        "gate_ticks": 0,
    }
    exit_spec = {
        "kind": "ewma_dev",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "tau_ms": 100,
        "depth_units": 1_000,
        "gate_ticks": 0,
    }
    params = ThresholdParams(direction=(1, 0), size_units=(1_000, 0), enter=0.5, exit=0.0, max_hold_ns=80 * MS)
    experiment = GridExperiment("TEST", entry, params, latency_ms=5, exit_signal=exit_spec)
    rows = evaluate_experiment(
        prep,
        build_signal(prep, entry),
        experiment,
        [Window("FULL", plan)],
        {"base": DEMO_BASE_FEES},
        1_000_000,
        exit_signal=build_signal(prep, exit_spec),
    )
    direct = run_replay(
        prep,
        build_signal(prep, entry),
        params,
        Scenario("x", DEMO_BASE_FEES, 5 * MS),
        plan,
        RiskLimits(),
        exit_signal=build_signal(prep, exit_spec),
    )
    assert rows[0]["net_pnl"] == direct.settlement.net_pnl
    assert rows[0]["exit_signal"] == exit_spec
    assert experiment.config_id != GridExperiment("TEST", entry, params, latency_ms=5).config_id


def test_episode_nets_partition_the_account_and_feed_concentration_stats() -> None:
    from financeai.application.grid import concentration, episode_nets

    prep = PreparedMarket.build(random_market(23))
    signal = np.random.default_rng(4).normal(size=len(prep.stream))
    first, last = int(prep.stream.recv_ns[0]), int(prep.stream.recv_ns[-1])
    session = SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    params = ThresholdParams(direction=(1, 1), size_units=(500, 500), enter=0.6, exit=0.0, max_hold_ns=40 * MS)
    res = run_replay(prep, signal, params, Scenario("b", DEMO_BASE_FEES, 5 * MS), session, RiskLimits())
    eps = episode_nets(res.fills, len(prep.data))
    assert len(eps) >= 2
    assert res.settlement.net_pnl is not None
    assert sum(net for _, net in eps) == pytest.approx(res.settlement.net_pnl, rel=1e-12, abs=1e-15)
    stats = concentration([net for _, net in eps])
    best = max(net for _, net in eps)
    assert stats["episodes"] == len(eps)
    assert stats["top1_pnl"] == best
    assert stats["net_ex_top1_pnl"] == pytest.approx(res.settlement.net_pnl - max(best, 0.0))


def test_concentration_never_credits_a_losing_best_episode() -> None:
    from financeai.application.grid import concentration

    stats = concentration([-0.2, -0.1])
    assert stats["top1_pnl"] == -0.1
    assert stats["net_ex_top1_pnl"] == pytest.approx(-0.3)  # nothing is trimmed when no episode made money
    assert concentration([])["episodes"] == 0 and concentration([])["net_ex_top1_pnl"] == 0.0


def test_evaluate_experiment_full_rows_carry_concentration_stats() -> None:
    prep = PreparedMarket.build(random_market(24))
    first, last = int(prep.stream.recv_ns[0]), int(prep.stream.recv_ns[-1])
    windows = [Window("FULL", SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last))]
    signal = np.random.default_rng(5).normal(size=len(prep.stream))
    experiment = GridExperiment(
        "TEST",
        {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 100},
        ThresholdParams(direction=(1, 0), size_units=(1_000, 0), enter=0.8, exit=-1e9, max_hold_ns=30 * MS),
        latency_ms=5,
    )
    rows = evaluate_experiment(
        prep, signal, experiment, windows, {"base": DEMO_BASE_FEES, "high": DEMO_HIGH_FEES}, 1_000_000
    )
    for row in rows:
        assert {"episodes", "top1_pnl", "top2_pnl", "net_ex_top1_pnl"} <= set(row)
        if row["valid"]:
            assert row["net_ex_top1_pnl"] == pytest.approx(row["net_pnl"] - max(row["top1_pnl"], 0.0))


LATE_FIELDS = ("reverse", "abs_cap_ns", "early_check_ns", "early_markout_bps")


def test_config_identity_omits_late_fields_at_their_defaults() -> None:
    import dataclasses as dc
    from typing import Any

    from financeai.core.hashing import sha256_json

    params = ThresholdParams(direction=(1, 0), size_units=(1_000, 0), enter=0.5, exit=-1e9)
    experiment = GridExperiment("X", {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 1}, params, 5)
    doc = dc.asdict(params)
    for name in LATE_FIELDS:
        doc.pop(name)
    assert (
        experiment.config_id == sha256_json({"signal": dict(experiment.signal), "params": doc})[:16]
    )  # pre-v5 identities hold
    changes: tuple[dict[str, Any], ...] = (
        {"reverse": True},
        {"abs_cap_ns": 1},
        {"early_check_ns": 1},
        {"early_markout_bps": -0.5},
        {"early_check_ns": 30 * MS, "early_markout_bps": 1.0},
    )
    for change in changes:
        assert GridExperiment("X", experiment.signal, dc.replace(params, **change), 5).config_id != experiment.config_id


def test_default_clock_options_do_not_change_config_identity() -> None:
    """Canonical identities ignore optional clocks when those options are disabled."""
    import dataclasses as dc

    params = ThresholdParams(direction=(1, 0), size_units=(1_000, 0), enter=0.5, exit=-1e9)
    signal = {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 1}
    assert GridExperiment("X", signal, params, 5).config_id == "19d5fba318e633f7"
    richer = dc.replace(params, reverse=True, max_clips=2, refresh_hold=True, stop_bps=1.5)
    assert GridExperiment("X", signal, richer, 5).config_id == "b25f92e747753eab"
    explicit = dc.replace(richer, abs_cap_ns=0, early_check_ns=0, early_markout_bps=0.0)
    assert GridExperiment("X", signal, explicit, 5).config_id == "b25f92e747753eab"


def test_grid_rows_carry_the_clock_options_and_match_a_direct_replay() -> None:
    import dataclasses as dc

    prep = PreparedMarket.build(random_market(26))
    first, last = int(prep.stream.recv_ns[0]), int(prep.stream.recv_ns[-1])
    plan = SessionPlan(first, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    signal = np.random.default_rng(6).normal(size=len(prep.stream))
    params = ThresholdParams(
        direction=(1, 0),
        size_units=(1_000, 0),
        enter=0.8,
        exit=-1e9,
        max_hold_ns=80 * MS,
        abs_cap_ns=20 * MS,
        early_check_ns=5 * MS,
        early_markout_bps=0.25,
    )
    experiment = GridExperiment(
        "TEST", {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 100}, params, latency_ms=5
    )
    row = evaluate_experiment(prep, signal, experiment, [Window("FULL", plan)], {"base": DEMO_BASE_FEES}, 1_000_000)[0]
    direct = run_replay(prep, signal, params, Scenario("x", DEMO_BASE_FEES, 5 * MS), plan, RiskLimits())
    assert row["net_pnl"] == direct.settlement.net_pnl and row["orders"] == len(direct.orders)
    assert (row["params"]["abs_cap_ns"], row["params"]["early_check_ns"], row["params"]["early_markout_bps"]) == (
        20 * MS,
        5 * MS,
        0.25,
    )
    plain = run_replay(
        prep,
        signal,
        dc.replace(params, abs_cap_ns=0, early_check_ns=0),
        Scenario("x", DEMO_BASE_FEES, 5 * MS),
        plan,
        RiskLimits(),
    )
    assert len(plain.orders) != len(direct.orders)  # the options are live through the grid path
