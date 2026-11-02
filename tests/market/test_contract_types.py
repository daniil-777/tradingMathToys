"""The same execution engine must settle different markets in their declared units."""

from __future__ import annotations

import dataclasses
from decimal import Decimal

import numpy as np
import pytest

from financeai.domains.market.engine import ThresholdParams, run_replay
from financeai.domains.market.instruments import ContractType, InstrumentSpec
from financeai.domains.market.ledger import exact_score
from financeai.domains.market.reference import ThresholdPolicy, run_reference
from financeai.domains.market.scenario import RiskLimits, Scenario
from financeai.domains.market.stream import MarketData

from .test_engine import SESSION, one_venue_market


@pytest.mark.parametrize("kind", list(ContractType))
@pytest.mark.parametrize("step,multiplier", [("1", "1"), ("0.001", "1"), ("1", "50")])
@pytest.mark.parametrize("side", [1, -1])
def test_contracts_match_reference_and_exact_cash(kind: ContractType, step: str, multiplier: str, side: int) -> None:
    spec = InstrumentSpec(
        "example",
        "INSTRUMENT",
        Decimal("0.01"),
        contract_type=kind,
        quantity_step=Decimal(step),
        contract_multiplier=Decimal(multiplier),
        base_currency="ASSET",
        quote_currency="EUR",
    )
    tape = dataclasses.replace(one_venue_market().venues[0], instrument=spec)
    data = MarketData((tape,))
    params = ThresholdParams((1,), (100,), 0.5, -10.0, entry_slip_ticks=10, exit_slip_ticks=20)
    scenario = Scenario("custom", {spec.key: Decimal("0.0001")}, 5_000_000, fees_per_unit={spec.key: Decimal("0.003")})
    signal = np.array([side, 0, 0], dtype=float)
    fast = run_replay(data, signal, params, scenario, SESSION, RiskLimits())
    slow = run_reference(data, signal, ThresholdPolicy(params), scenario, SESSION, RiskLimits())
    for table in ("fills", "orders"):
        a, b = getattr(fast, table), getattr(slow, table)
        for column in dataclasses.fields(a):
            np.testing.assert_array_equal(getattr(a, column.name), getattr(b, column.name))
    oracle = exact_score(fast.fills, data.instruments, (scenario.fee_for(spec),), (scenario.fee_per_unit_for(spec),))
    assert oracle is not None
    assert fast.settlement.net_pnl == pytest.approx(float(oracle), abs=1e-8)
    assert fast.settlement.currency == ("ASSET" if kind == ContractType.INVERSE else "EUR")
    assert fast.settlement.valid
    if kind == ContractType.SPOT and side < 0:
        assert len(fast.fills) == 0
    else:
        assert len(fast.fills) > 0


def test_mixed_currencies_are_reported_separately() -> None:
    tape = one_venue_market().venues[0]
    eur = InstrumentSpec("one", "ASSET-EUR", Decimal("0.5"), quote_currency="EUR")
    usd = dataclasses.replace(eur, venue="two", symbol="ASSET-USD", quote_currency="USD", settlement_currency=None)
    data = MarketData((dataclasses.replace(tape, instrument=eur), dataclasses.replace(tape, instrument=usd)))
    params = ThresholdParams((1, 1), (100, 100), 0.5, -10.0, entry_slip_ticks=10, exit_slip_ticks=20)
    scenario = Scenario("two currencies", {eur.key: Decimal(0), usd.key: Decimal(0)}, 5_000_000)
    result = run_replay(data, np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]), params, scenario, SESSION, RiskLimits())
    assert result.settlement.valid
    assert result.settlement.net_pnl is None
    assert result.settlement.currency is None
    assert set(result.settlement.pnl_by_currency or {}) == {"EUR", "USD"}


def test_explicit_spot_borrowing_allows_shorts() -> None:
    spec = InstrumentSpec("example", "ASSET", Decimal("0.01"), contract_type=ContractType.SPOT, shortable=True)
    assert spec.shortable
    assert not InstrumentSpec("example", "ASSET", Decimal("0.01"), contract_type=ContractType.SPOT).shortable


def test_fractional_grid_metadata_round_trip() -> None:
    spec = InstrumentSpec(
        "example",
        "SMALL-COIN",
        Decimal("0.00000001"),
        contract_type=ContractType.SPOT,
        quantity_step=Decimal("0.000001"),
    )
    assert InstrumentSpec.from_mapping(spec.to_mapping()) == spec
    assert spec.quantity_of(125) == Decimal("0.000125")


@pytest.mark.parametrize("kind", list(ContractType))
def test_labels_marks_and_rescoring_use_the_contract_convention(kind: ContractType) -> None:
    from financeai.application.grid import rescore
    from financeai.domains.market.engine import PreparedMarket
    from financeai.domains.market.labels import executable_labels, label_fills
    from financeai.domains.market.metrics import marked_value, marked_value_path, summary

    from .test_labels import one_shot_replay

    spec = InstrumentSpec(
        "example",
        "CONTRACT",
        Decimal("0.01"),
        contract_type=kind,
        quantity_step=Decimal("0.01"),
        contract_multiplier=Decimal("10"),
    )
    data = MarketData((dataclasses.replace(one_venue_market().venues[0], instrument=spec),))
    prep = PreparedMarket.build(data)
    scenario = Scenario("costs", {spec.key: Decimal("0.0001")}, 5_000_000, fees_per_unit={spec.key: Decimal("0.003")})
    fast, slow = one_shot_replay(prep, 1_000_000, 1, 0, 100, 10_000_000, scenario, SESSION.exec_cutoff_ns, 10, 20)
    lab = executable_labels(
        prep, [1_000_000], 1, 0, 100, 10_000_000, scenario, entry_slip_ticks=10, session_end_ns=SESSION.exec_cutoff_ns
    )
    fills = label_fills(
        prep, 1_000_000, 1, 0, 100, 10_000_000, scenario, entry_slip_ticks=10, session_end_ns=SESSION.exec_cutoff_ns
    )
    np.testing.assert_array_equal(fills.cash, fast.fills.cash)
    assert lab.net_pnl[0] == fast.settlement.net_pnl == slow.settlement.net_pnl
    assert lab.fees[0] == fast.settlement.fees[0]
    assert summary(fast, prep, scenario)["reconciled"]
    times, values = marked_value_path(fast, prep)
    np.testing.assert_allclose(values, marked_value(fast, prep, times), rtol=0, atol=1e-10)
    assert values[-1] == fast.settlement.net_pnl
    from financeai.research.performance import account_increments

    path = account_increments(fast, prep, 1_000_000, "exit", start_ns=0, end_ns=SESSION.exec_cutoff_ns)
    assert path.currency == spec.settlement_currency
    q = path.position_units[:, 0]
    row = np.maximum(np.searchsorted(prep.book_recv[0], path.times_ns, side="right") - 1, 0)
    prices = np.where(q > 0, prep.bpx[0][row, 0], prep.apx[0][row, 0]) * spec.tick_float
    inventory = -q * spec.cash_factor / prices if spec.inverse else q * spec.cash_factor * prices
    np.testing.assert_allclose(path.marked_pnl, path.cash + inventory, rtol=0, atol=1e-9)
    assert path.marked_pnl[-1] == fast.settlement.net_pnl
    alternative = rescore(fast.fills, data.instruments, [Decimal("0.0003")], 1000, [Decimal("0.004")])
    exact = exact_score(fast.fills, data.instruments, [Decimal("0.0003")], [Decimal("0.004")])
    assert alternative.net_pnl == pytest.approx(float(exact), abs=1e-9)


def test_fractional_recorder_ingestion_and_cache_roundtrip(tmp_path) -> None:
    from financeai.domains.market.cache import SourceInfo, load_tapes, save_tapes
    from financeai.domains.market.recorder import parse_recorder_bytes, read_recorder_file

    from .conftest import SYNTHETIC_RECORDER_TEXT

    spec = InstrumentSpec(
        "custom",
        "SHARES",
        Decimal("0.01"),
        contract_type=ContractType.SPOT,
        quantity_step=Decimal("0.001"),
        quote_currency="CHF",
    )
    tape = parse_recorder_bytes(SYNTHETIC_RECORDER_TEXT, spec)
    raw = tmp_path / "recorder.txt"
    raw.write_bytes(SYNTHETIC_RECORDER_TEXT)
    streamed = read_recorder_file(raw, spec)
    np.testing.assert_array_equal(tape.book.ask_sz, streamed.book.ask_sz)
    assert tape.report.bytes_parsed == len(SYNTHETIC_RECORDER_TEXT)
    cache = tmp_path / "cache"
    save_tapes(tape, cache, source=SourceInfo.from_file(raw))
    loaded = load_tapes(cache)
    assert loaded.instrument == spec
    np.testing.assert_array_equal(loaded.trades.sz, tape.trades.sz)


def test_custom_synthetic_market_preserves_identity() -> None:
    from financeai.domains.market.synthetic import SyntheticMarketSpec, synthetic_market

    spec = InstrumentSpec(
        "custom",
        "EUR-CHF",
        Decimal("0.00001"),
        contract_type=ContractType.SPOT,
        quantity_step=Decimal("0.01"),
        base_currency="EUR",
        quote_currency="CHF",
    )
    description = SyntheticMarketSpec(seed=3, instruments=(spec,), rows_per_venue=10)
    assert SyntheticMarketSpec.from_mapping(description.to_mapping()) == description
    assert synthetic_market(description).instruments == (spec,)


def test_scenario_costs_roundtrip() -> None:
    scenario = Scenario(
        "commission", {"broker/STOCK": Decimal("0.001")}, 1_000_000, fees_per_unit={"broker/STOCK": Decimal("0.005")}
    )
    assert Scenario.from_mapping(scenario.to_mapping()) == scenario
