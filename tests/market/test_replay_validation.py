"""Replay boundaries reject unsafe inputs before entering numerical kernels."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.domains.market import engine
from financeai.domains.market.engine import ThresholdParams, run_replay
from financeai.domains.market.reference import ThresholdPolicy, run_reference
from financeai.domains.market.scenario import RiskLimits, SessionPlan
from financeai.domains.market.stream import MarketData

from .test_engine import BUY_ONCE, SESSION, one_venue_market, scenario

MAX_TIME = 2**63 - 2  # int64 max itself is the kernel's no-event sentinel


def shifted_market(offset: int) -> MarketData:
    tape = one_venue_market().venues[0]
    book = dataclasses.replace(tape.book, recv_ns=tape.book.recv_ns + offset, exch_ns=tape.book.exch_ns + offset)
    return MarketData((dataclasses.replace(tape, book=book),))


def shifted_session(offset: int) -> SessionPlan:
    return SessionPlan(**{k: v + offset for k, v in dataclasses.asdict(SESSION).items()})


@pytest.mark.parametrize("source", ["event", "drain", "cooldown"])
def test_unsafe_clocks_never_enter_the_compiled_kernel(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    data, session, params = one_venue_market(), SESSION, BUY_ONCE
    if source == "event":
        data = shifted_market(MAX_TIME - 30_000_000)
    elif source == "drain":
        session = dataclasses.replace(SESSION, end_ns=MAX_TIME)
    else:
        offset = 2**62
        data, session = shifted_market(offset), shifted_session(offset)
        params = dataclasses.replace(params, cooldown_ns=2**62)

    def forbidden(*args: Any) -> None:
        pytest.fail("unsafe inputs reached the numerical kernel")

    monkeypatch.setattr(engine, "_run_kernel", forbidden)
    with pytest.raises(InvalidDataError, match="int64"):
        run_replay(data, np.ones(3), params, scenario(), session, RiskLimits())


def test_reference_engine_checks_the_same_clock_contract() -> None:
    late = dataclasses.replace(SESSION, end_ns=MAX_TIME)
    with pytest.raises(InvalidDataError, match="int64"):
        run_reference(one_venue_market(), np.zeros(3), ThresholdPolicy(BUY_ONCE), scenario(), late, RiskLimits())


def test_safe_large_clocks_preserve_orders_and_settlement() -> None:
    signal = np.array([1.0, 0.0, 0.0])
    offset = 2**62
    a = run_replay(one_venue_market(), signal, BUY_ONCE, scenario(), SESSION, RiskLimits())
    b = run_replay(shifted_market(offset), signal, BUY_ONCE, scenario(), shifted_session(offset), RiskLimits())
    assert a.settlement == b.settlement
    for field in ("submit_ns", "arrival_ns", "ack_ns"):
        np.testing.assert_array_equal(getattr(b.orders, field) - offset, getattr(a.orders, field))


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("exit_only", [False, True])
@pytest.mark.parametrize("signal", [np.array([0.0, np.inf, 0.0]), np.array([0.0, -np.inf, 0.0]), np.zeros((3, 1))])
def test_invalid_signals_fail_consistently(reference: bool, exit_only: bool, signal: np.ndarray) -> None:
    entry = np.zeros(3) if exit_only else signal
    kwargs = {"exit_signal": signal} if exit_only else {}
    with pytest.raises(InvalidConfigError):
        if reference:
            run_reference(
                one_venue_market(), entry, ThresholdPolicy(BUY_ONCE), scenario(), SESSION, RiskLimits(), **kwargs
            )
        else:
            run_replay(one_venue_market(), entry, BUY_ONCE, scenario(), SESSION, RiskLimits(), **kwargs)


def test_nan_signals_remain_supported_as_unavailable_values() -> None:
    result = run_replay(one_venue_market(), np.full(3, np.nan), BUY_ONCE, scenario(), SESSION, RiskLimits())
    assert len(result.orders) == 0 and result.settlement.valid


@pytest.mark.parametrize("field", ["order_latency_ns", "ack_latency_ns", "consumption_ttl_ns"])
def test_scenario_durations_fit_the_engine_clock(field: str) -> None:
    with pytest.raises(InvalidConfigError):
        dataclasses.replace(scenario(), **{field: 2**63})


@pytest.mark.parametrize("value", [-1, 2**63 - 1, 2**63])
def test_session_clocks_exclude_negative_values_and_the_sentinel(value: int) -> None:
    with pytest.raises(InvalidConfigError):
        dataclasses.replace(SESSION, **({"start_ns": value} if value < 0 else {"end_ns": value}))


def test_risk_cap_leaves_room_for_a_full_position_reversal() -> None:
    assert RiskLimits(2**62 - 1).max_abs_position_units == 2**62 - 1
    with pytest.raises(InvalidConfigError):
        RiskLimits(2**62)


@pytest.mark.parametrize("changes", [{"direction": (1.0,)}, {"allow_short": "false"}, {"enter": True}, {"exit": "0"}])
def test_direct_threshold_parameters_do_not_bypass_type_checks(changes: dict[str, Any]) -> None:
    with pytest.raises(InvalidConfigError):
        ThresholdParams(**{**dataclasses.asdict(BUY_ONCE), **changes})
