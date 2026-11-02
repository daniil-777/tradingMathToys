"""Market-specific controls survive heterogeneous portfolios and public configuration boundaries."""

from __future__ import annotations

import dataclasses
import json
import pickle
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.domains.market import InstrumentLatency, RiskLimits
from financeai.domains.market.engine import PreparedMarket, ThresholdParams, run_replay
from financeai.domains.market.instruments import ContractType, InstrumentSpec
from financeai.domains.market.ledger import Fills, settle
from financeai.domains.market.recorder import _reference_ticks, _reference_units
from financeai.domains.market.reference import ThresholdPolicy, run_reference
from financeai.domains.market.scenario import MAX_REPLAY_NS, Scenario
from financeai.domains.market.stream import MarketData

from .test_engine import SESSION, one_venue_market

MS = 1_000_000


def portfolio(count: int = 3) -> MarketData:
    template = one_venue_market().venues[0]
    specs = (
        InstrumentSpec(
            "exchange",
            "SHARE-CHF",
            Decimal("0.01"),
            contract_type=ContractType.SPOT,
            quantity_step=Decimal("0.001"),
            quote_currency="CHF",
        ),
        InstrumentSpec(
            "exchange",
            "INDEX-CHF",
            Decimal("0.25"),
            lot_size=10,
            contract_multiplier=Decimal("50"),
            quote_currency="CHF",
        ),
        InstrumentSpec(
            "other", "COIN-USD", Decimal("0.5"), lot_size=2, contract_type=ContractType.INVERSE, base_currency="COIN"
        ),
    )
    return MarketData(
        tuple(
            dataclasses.replace(
                template, instrument=dataclasses.replace(specs[v % 3], symbol=f"{specs[v % 3].symbol}-{v}")
            )
            for v in range(count)
        )
    )


def compare(data: MarketData, limits: RiskLimits, overrides: dict[str, InstrumentLatency]):
    prep = PreparedMarket.build(data)
    scenario = Scenario(
        "heterogeneous",
        {i.key: Decimal("0.0002") for i in data.instruments},
        MS,
        ack_latency_ns=2 * MS,
        latency_by_instrument=overrides,
    )
    params = ThresholdParams(
        (1,) * len(data), (10_000,) * len(data), 0.5, -10.0, entry_slip_ticks=10, exit_slip_ticks=20
    )
    signal = np.ones(len(prep.stream))
    fast = run_replay(prep, signal, params, scenario, SESSION, limits)
    slow = run_reference(data, signal, ThresholdPolicy(params), scenario, SESSION, limits)
    for name in ("orders", "fills"):
        for field in dataclasses.fields(getattr(fast, name)):
            np.testing.assert_array_equal(
                getattr(getattr(fast, name), field.name),
                getattr(getattr(slow, name), field.name),
                err_msg=f"{name}.{field.name}",
            )
    assert fast.settlement == slow.settlement
    assert fast.settlement.valid
    assert fast.limits == slow.limits == limits
    for v, instrument in enumerate(data.instruments):
        order = fast.orders.venue == v
        latency = scenario.latency_for(instrument)
        np.testing.assert_array_equal(
            fast.orders.arrival_ns[order] - fast.orders.submit_ns[order], np.full(order.sum(), latency.order_latency_ns)
        )
        np.testing.assert_array_equal(
            fast.orders.ack_ns[order] - fast.orders.arrival_ns[order], np.full(order.sum(), latency.ack_ns)
        )
        assert fast.settlement.max_abs_position_units[v] <= limits.cap_for(instrument)
    return fast, prep, scenario


@pytest.mark.parametrize("count", [1, 3, 8])
def test_multiple_assets_on_one_exchange_have_independent_caps_and_clocks(count: int) -> None:
    data = portfolio(count)
    caps = {i.key: 23 + 20 * v for v, i in enumerate(data.instruments)}
    delays = {i.key: InstrumentLatency((v % 3) * MS, (3 - v % 3) * MS) for v, i in enumerate(data.instruments)}
    result, _, _ = compare(data, RiskLimits(100, per_instrument=caps), delays)
    assert len(result.fills) > 0
    for v, instrument in enumerate(data.instruments):
        assert (
            result.settlement.max_abs_position_units[v]
            == caps[instrument.key] // instrument.lot_size * instrument.lot_size
        )
    assert result.settlement.net_pnl is None if count > 1 else result.settlement.net_pnl is not None


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_random_instrument_controls_match_the_independent_engine(seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = portfolio()
    caps = {i.key: int(rng.integers(0, 500)) for i in data.instruments}
    delays = {i.key: InstrumentLatency(int(rng.integers(0, 5)) * MS, MS) for i in data.instruments}
    compare(data, RiskLimits(1000, per_instrument=caps), delays)


def test_zero_cap_disables_only_the_selected_instrument_and_metrics_use_actual_limits() -> None:
    from financeai.domains.market.metrics import summary

    data = portfolio()
    disabled = data.instruments[1].key
    result, prep, scenario = compare(data, RiskLimits(23, per_instrument={disabled: 0}), {})
    assert not np.any(result.orders.venue == 1)
    with pytest.raises(InvalidConfigError, match="common settlement currency"):
        summary(result, prep, scenario)
    first = MarketData(data.venues[:1])
    result, prep, scenario = compare(first, RiskLimits(1000, per_instrument={first.instruments[0].key: 23}), {})
    report = summary(result, prep, scenario)
    assert report["reconciled"] and report["cap_units_by_instrument"] == [23]


def test_json_and_process_serialization_preserve_immutable_instrument_controls() -> None:
    data = portfolio()
    spec = data.instruments[0]
    overrides = {spec.key: InstrumentLatency(MS, 3 * MS)}
    limits = RiskLimits(100, per_instrument={spec.key: 23})
    assert len({limits, RiskLimits.from_mapping(limits.to_mapping())}) == 1
    scenario = Scenario(
        "custom", {i.key: Decimal("0.0001") for i in data.instruments}, MS, latency_by_instrument=overrides
    )
    for value in (limits, scenario):
        assert type(value).from_mapping(json.loads(json.dumps(value.to_mapping()))) == value
        assert pickle.loads(pickle.dumps(value)) == value
    overrides.clear()
    assert scenario.latency_for(spec).ack_ns == 3 * MS
    with pytest.raises(TypeError):
        limits.per_instrument[spec.key] = 1


@pytest.mark.parametrize("bad", [True, -1, 1.5, 2**62, None])
def test_bad_instrument_caps_are_rejected(bad) -> None:
    with pytest.raises(InvalidConfigError):
        RiskLimits(per_instrument={"exchange/symbol": bad})


@pytest.mark.parametrize("order,ack", [(0, 0), (-1, 1), (True, 1), (1, False), (1, 1.5), (MAX_REPLAY_NS + 1, 1)])
def test_bad_instrument_latencies_are_rejected(order, ack) -> None:
    with pytest.raises(InvalidConfigError):
        InstrumentLatency(order, ack)


def test_unknown_controls_fail_instead_of_silently_using_a_default() -> None:
    with pytest.raises(InvalidConfigError, match="unknown instruments"):
        compare(portfolio(), RiskLimits(per_instrument={"typo/symbol": 1}), {})
    with pytest.raises(InvalidConfigError, match="unknown instruments"):
        compare(portfolio(), RiskLimits(), {"typo/symbol": InstrumentLatency(MS)})


def test_clock_overflow_validation_includes_the_longest_instrument_delay() -> None:
    data = portfolio(1)
    with pytest.raises(InvalidDataError, match="int64 room"):
        compare(data, RiskLimits(), {data.instruments[0].key: InstrumentLatency(MAX_REPLAY_NS, 1)})


def test_label_context_uses_the_selected_instruments_cap_and_latency() -> None:
    from financeai.domains.market.labels import executable_labels

    data = portfolio(1)
    instrument = data.instruments[0]
    scenario = Scenario(
        "labels",
        {instrument.key: Decimal("0.0001")},
        MS,
        latency_by_instrument={instrument.key: InstrumentLatency(4 * MS, 2 * MS)},
    )
    limits = RiskLimits(1000, per_instrument={instrument.key: 23})
    args = (data, [MS], 1, 0, 23, 10 * MS)
    actual = executable_labels(*args, scenario, limits=limits, session_end_ns=40 * MS, entry_slip_ticks=10)
    expected = executable_labels(
        *args, scenario.for_instrument(instrument), limits=RiskLimits(23), session_end_ns=40 * MS, entry_slip_ticks=10
    )
    for field in dataclasses.fields(actual):
        np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))


@pytest.mark.parametrize("precision", [3, 28, 60])
def test_decimal_grid_checks_do_not_depend_on_the_callers_decimal_context(precision: int) -> None:
    spec = InstrumentSpec("exchange", "EXACT", Decimal("0.01"), quantity_step=Decimal("0.001"))
    with localcontext() as context:
        context.prec = precision
        assert _reference_ticks("12345.67", spec, "") == 1_234_567
        assert _reference_units("12345.678", spec, "") == 12_345_678
        for convert, token in [
            (_reference_ticks, "1.00000000000000000000000000000001"),
            (_reference_units, "1.00000000000000000000000000000001"),
        ]:
            with pytest.raises(InvalidDataError, match="grid"):
                convert(token, spec, "")
        unusual = dataclasses.replace(
            spec, tick_size=Decimal("0.50000000000000000000000000000001"), quantity_step=Decimal(1)
        )
        assert not unusual.fast_text_compatible


@pytest.mark.parametrize("token", ["1e999999999", "1e-999999999"])
def test_decimal_grid_checks_reject_unrepresentable_exponents_before_integer_conversion(token: str) -> None:
    with pytest.raises(InvalidDataError):
        _reference_ticks(token, portfolio(1).instruments[0], "")


def test_settlement_refuses_integer_position_wraparound() -> None:
    size = 2**62 - 1
    qty = np.array([size, size, size, size, size, size], np.int64)
    fills = Fills(
        np.arange(6, dtype=np.int64),
        np.zeros(6, np.int8),
        np.array([1, 1, 1, -1, -1, -1], np.int8),
        np.ones(6, np.int32),
        qty,
        np.zeros(6, np.int64),
        np.zeros(6, np.int64),
        np.zeros(6, np.int64),
        np.zeros(6, np.float64),
    )
    with pytest.raises(InvalidDataError, match="overflows int64"):
        settle(fills, portfolio(1).instruments, [Decimal(0)], cap_units=size)


def test_instrument_reordering_preserves_each_instruments_execution() -> None:
    data = portfolio()
    limits = RiskLimits(100, per_instrument={i.key: 23 + 20 * v for v, i in enumerate(data.instruments)})
    delays = {i.key: InstrumentLatency((v + 1) * MS, MS) for v, i in enumerate(data.instruments)}
    original, _, _ = compare(data, limits, delays)
    reordered_data = MarketData(tuple(reversed(data.venues)))
    reordered, _, _ = compare(reordered_data, limits, delays)
    for instrument in data.instruments:
        a = [i.key for i in data.instruments].index(instrument.key)
        b = [i.key for i in reordered_data.instruments].index(instrument.key)
        for name in ("side", "px_ticks", "qty_units", "submit_ns", "arrival_ns", "ack_ns", "cash"):
            np.testing.assert_array_equal(
                getattr(original.fills, name)[original.fills.venue == a],
                getattr(reordered.fills, name)[reordered.fills.venue == b],
            )


def test_inverse_research_uses_the_same_selected_limits_and_latency() -> None:
    from financeai.research.hindsight import depth_oracle

    tape = portfolio().venues[2]
    scenario = Scenario(
        "research",
        {tape.instrument.key: Decimal(0)},
        7 * MS,
        consumption_ttl_ns=0,
        latency_by_instrument={tape.instrument.key: InstrumentLatency(MS, 3 * MS)},
    )
    limits = RiskLimits(1000, per_instrument={tape.instrument.key: 6})
    actual = depth_oracle(tape, scenario, SESSION, limits, step_units=2, spacing_ns=4 * MS)
    expected = depth_oracle(
        tape, scenario.for_instrument(tape.instrument), SESSION, RiskLimits(6), step_units=2, spacing_ns=4 * MS
    )
    assert np.any(actual.targets_units != 0)
    for field in dataclasses.fields(actual):
        np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))


@pytest.mark.parametrize("token", ["0e999999999", "0e-999999999"])
def test_zero_normalization_does_not_expand_its_decimal_exponent(token: str) -> None:
    spec = portfolio(1).instruments[0]
    assert _reference_ticks(token, spec, "") == _reference_units(token, spec, "") == 0


def test_label_latency_headroom_checks_the_override_not_the_default() -> None:
    from financeai.domains.market.labels import executable_labels

    data = portfolio(1)
    spec = data.instruments[0]
    scenario = Scenario(
        "overflow", {spec.key: Decimal(0)}, MS, latency_by_instrument={spec.key: InstrumentLatency(2**60 + 1, 1)}
    )
    with pytest.raises(InvalidConfigError, match=r"2\*\*60"):
        executable_labels(data, [MS], 1, 0, 10, MS, scenario, session_end_ns=40 * MS)


@pytest.mark.parametrize("count", [1, 2])
def test_reports_distinguish_uniform_and_instrument_specific_latencies(count: int) -> None:
    from financeai.domains.market.metrics import summary

    data = portfolio(count)
    key = data.instruments[-1].key
    result, prep, scenario = compare(data, RiskLimits(100), {key: InstrumentLatency(3 * MS, 4 * MS)})
    report = summary(result, prep, scenario)
    assert report["order_latency_ns"] == (3 * MS if count == 1 else None)
    assert report["ack_latency_ns"] == (4 * MS if count == 1 else None)
    assert report["latency_by_instrument"][key] == {"order_latency_ns": 3 * MS, "ack_latency_ns": 4 * MS}
    assert report["reconciled"]
