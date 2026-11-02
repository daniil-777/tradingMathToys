"""Tests for cross-venue confirmation signal builders, episode diagnostics, and rational enclosures."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from financeai.application.strategy import StrategyConfig, run_strategy
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.instruments import ContractType, InstrumentSpec
from financeai.domains.market.ledger import Fills
from financeai.domains.market.reference import ThresholdPolicy, run_reference
from financeai.domains.market.scenario import DEMO_BASE_FEES, RiskLimits, demo_scenarios
from financeai.domains.market.synthetic import SyntheticMarketSpec, synthetic_market
from financeai.research.cross_venue_confirmation import (
    CrossVenueConfirmationSpec,
    build_cross_venue_signal_spec,
    build_cross_venue_strategy_config,
    certify_fill_ledger_interval,
    diagnose_episode_confirmation,
    example_asymmetric_confirmation,
    example_dual_confirmation,
    example_single_confirmation,
    rational_interval,
)
from financeai.strategies.leadlag import build_signal


def _base_strategy_config() -> StrategyConfig:
    return StrategyConfig(
        name="test_cross_venue",
        instruments=("venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"),
        signal={"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 100_000},
        policy={
            "direction": {"venue_a/ALPHA-USD": 1, "venue_b/ALPHA-PERP": 0},
            "size_units": {"venue_a/ALPHA-USD": 200_000, "venue_b/ALPHA-PERP": 0},
            "enter": 2.0,
            "exit": 0.4,
            "allow_short": True,
            "min_hold_ms": 5_000,
            "max_hold_ms": 90_000,
            "cooldown_ms": 10,
            "stop_bps": 30.0,
            "entry_slip_ticks": 2,
            "exit_slip_ticks": 20,
        },
        session={
            "warmup_s": 0.05,
            "entry_stop_s": 0.2,
            "flatten_s": 0.1,
            "exec_stop_s": 0.01,
        },
    )


def test_presets_build_valid_signal_specs_and_configs() -> None:
    base_cfg = _base_strategy_config()
    presets = (
        example_single_confirmation(),
        example_dual_confirmation(),
        example_asymmetric_confirmation(),
    )
    ids = set()
    for spec in presets:
        sig = build_cross_venue_signal_spec(spec)
        assert sig["kind"] == "any"
        assert len(sig["components"]) == 2
        cfg = build_cross_venue_strategy_config(spec, base_cfg)
        assert cfg.policy["exit_slip_ticks"] == spec.exit_slip_ticks
        # Round-trip through JSON validator
        reloaded = StrategyConfig.from_json(cfg.to_json())
        assert reloaded.config_id == cfg.config_id
        ids.add(cfg.config_id)
    assert len(ids) == 3


def test_dual_venue_gate_uses_kind_all_conjunction() -> None:
    spec = example_dual_confirmation()
    sig = build_cross_venue_signal_spec(spec)
    comp0 = sig["components"][0]["spec"]
    comp1 = sig["components"][1]["spec"]
    assert comp0["kind"] == "all"
    assert len(comp0["components"]) == 2
    assert comp1["kind"] == "all"
    assert len(comp1["components"]) == 2


def test_cross_venue_spec_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="sweep_min_units"):
        CrossVenueConfirmationSpec(name="bad", sweep_min_units=0.0)
    with pytest.raises(ValueError, match="gate thresholds"):
        CrossVenueConfirmationSpec(name="bad", sweep_venue_a_min_units=-1.0)
    with pytest.raises(ValueError, match="lead_scale"):
        CrossVenueConfirmationSpec(name="bad", lead_scale=0.0)
    with pytest.raises(ValueError, match="fresh_max_age_ms"):
        CrossVenueConfirmationSpec(name="bad", fresh_max_age_ms=0)
    with pytest.raises(ValueError, match="exit_slip_ticks"):
        CrossVenueConfirmationSpec(name="bad", exit_slip_ticks=-1)


def test_rational_interval_encloses_exact_fraction() -> None:
    val = Fraction(1, 3)
    enc = rational_interval(val, places=18)
    lo = Fraction(Decimal(enc.lower_str))
    hi = Fraction(Decimal(enc.upper_str))
    assert lo <= val <= hi
    assert Decimal(enc.width_str) == Decimal("1e-18")

    neg_val = Fraction(-2, 7)
    neg_enc = rational_interval(neg_val, places=18)
    assert Fraction(Decimal(neg_enc.lower_str)) <= neg_val <= Fraction(Decimal(neg_enc.upper_str))


def test_certify_fill_ledger_interval_and_validation() -> None:
    instruments = (
        InstrumentSpec("venue_a", "ALPHA-USD", Decimal("0.5"), 100, ContractType.INVERSE),
        InstrumentSpec("venue_b", "ALPHA-PERP", Decimal("0.5"), 10, ContractType.INVERSE),
    )
    fees = (Decimal("0.00005"), Decimal("0.0001"))
    fills = Fills(
        order_id=np.array([0, 1], dtype=np.int64),
        venue=np.array([0, 0], dtype=np.int8),
        side=np.array([1, -1], dtype=np.int8),
        px_ticks=np.array([40_000, 40_040], dtype=np.int32),
        qty_units=np.array([100_000, 100_000], dtype=np.int64),
        submit_ns=np.array([1_000, 2_000], dtype=np.int64),
        arrival_ns=np.array([1_500, 2_500], dtype=np.int64),
        ack_ns=np.array([2_000, 3_000], dtype=np.int64),
        cash=np.array([0.0, 0.0], dtype=np.float64),
    )
    enc = certify_fill_ledger_interval(fills, instruments, fees, cap_units=1_000_000)
    expected = Fraction(Decimal("1") - fees[0]) * 100_000 / (Fraction(1, 2) * 40_000) + Fraction(
        Decimal("-1") - fees[0]
    ) * 100_000 / (Fraction(1, 2) * 40_040)
    assert enc.exact == expected
    assert Decimal(enc.lower_str) > Decimal("0")

    non_flat = Fills(
        order_id=fills.order_id[:1],
        venue=fills.venue[:1],
        side=fills.side[:1],
        px_ticks=fills.px_ticks[:1],
        qty_units=fills.qty_units[:1],
        submit_ns=fills.submit_ns[:1],
        arrival_ns=fills.arrival_ns[:1],
        ack_ns=fills.ack_ns[:1],
        cash=fills.cash[:1],
    )
    with pytest.raises(ValueError, match="non-flat"):
        certify_fill_ledger_interval(non_flat, instruments, fees)


def test_synthetic_replay_reference_parity_and_episode_diagnostics() -> None:
    market = synthetic_market(SyntheticMarketSpec(seed=11))
    prep = PreparedMarket.build(market)
    spec = CrossVenueConfirmationSpec(
        name="synth_test",
        sweep_min_units=100.0,
        sweep_venue_a_min_units=0.0,
        sweep_venue_b_min_units=100.0,
        lead_venue_a_min_units=100.0,
        lead_venue_b_min_units=100.0,
        fresh_max_age_ms=5_000,
        exit_slip_ticks=10,
    )
    cfg = build_cross_venue_strategy_config(spec, _base_strategy_config())
    sig = build_signal(prep, cfg.signal)
    assert sig.shape == (len(prep.stream.recv_ns),)

    scenarios = demo_scenarios()
    run_res = run_strategy(prep, cfg, scenarios["base"])
    recv = prep.stream.recv_ns
    session = cfg.session_plan(int(recv[0]), int(recv[-1]))
    ref_res = run_reference(
        prep.data,
        sig,
        ThresholdPolicy(cfg.threshold_params()),
        scenarios["base"],
        session,
        RiskLimits(),
        stream=prep.stream,
    )
    assert np.array_equal(run_res.fills.px_ticks, ref_res.fills.px_ticks)
    assert np.array_equal(run_res.fills.qty_units, ref_res.fills.qty_units)

    fees = [DEMO_BASE_FEES[i.key] for i in prep.data.instruments]
    diags, conc = diagnose_episode_confirmation(prep, run_res.fills, prep.data.instruments, fees)
    assert conc["episodes"] == len(diags)
    enc = certify_fill_ledger_interval(run_res.fills, prep.data.instruments, fees)
    assert Decimal(enc.lower_str) <= Decimal(enc.upper_str)
