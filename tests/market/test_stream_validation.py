"""A supplied stream must preserve the identity and receipt order of its observations."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.domains.market.engine import PreparedMarket, run_replay
from financeai.domains.market.instruments import DEMO_INVERSE_B
from financeai.domains.market.reference import ThresholdPolicy, run_reference
from financeai.domains.market.scenario import RiskLimits
from financeai.domains.market.stream import EventStream

from .test_engine import BUY_ONCE, SESSION, one_venue_market, scenario


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("row", np.array([-1, 1, 2], np.int64)),
        ("row", np.array([0, 1, 3], np.int64)),
        ("venue", np.array([0, 1, 0], np.int8)),
        ("kind", np.array([0, 2, 0], np.int8)),
        ("row", np.array([0.0, 1.0, 2.0])),
        ("row", np.array([[0, 1, 2]], np.int64)),
        ("row", np.array([0, 1], np.int64)),
        ("recv_ns", np.array([1_000_000, 8_000_000, 30_000_000], np.int64)),
        ("recv_ns", np.array([1_000_000, 30_000_000, 9_000_000], np.int64)),
    ],
)
def test_invalid_streams_are_rejected_before_replay(reference: bool, field: str, values: np.ndarray) -> None:
    data = one_venue_market()
    stream = dataclasses.replace(EventStream.build(data), **{field: values})
    with pytest.raises(InvalidDataError):
        if reference:
            run_reference(
                data, np.zeros(3), ThresholdPolicy(BUY_ONCE), scenario(), SESSION, RiskLimits(), stream=stream
            )
        else:
            PreparedMarket.build(data, stream=stream)


def test_a_reduced_stream_can_retain_genuine_observations() -> None:
    data = one_venue_market()
    full = EventStream.build(data)
    reduced = EventStream(*(getattr(full, field)[[0, 2]] for field in ("recv_ns", "venue", "kind", "row")))
    prep = PreparedMarket.build(data, stream=reduced)
    result = run_replay(prep, np.zeros(2), BUY_ONCE, scenario(), SESSION, RiskLimits())
    assert result.settlement.valid and len(result.orders) == 0


@pytest.mark.parametrize("rows", [[0, 0, 2], [1, 0, 2]])
def test_tied_source_rows_cannot_be_repeated_or_reordered(rows: list[int]) -> None:
    tape = one_venue_market().venues[0]
    book = dataclasses.replace(tape.book, recv_ns=np.full(3, 1_000_000, np.int64))
    data = dataclasses.replace(one_venue_market(), venues=(dataclasses.replace(tape, book=book),))
    stream = dataclasses.replace(EventStream.build(data), row=np.array(rows, np.int64))
    with pytest.raises(InvalidDataError, match="repeats or reorders"):
        PreparedMarket.build(data, stream=stream)


def test_tied_observations_preserve_venue_order() -> None:
    tape = one_venue_market().venues[0]
    data = dataclasses.replace(one_venue_market(), venues=(tape, dataclasses.replace(tape, instrument=DEMO_INVERSE_B)))
    full = EventStream.build(data)
    order = [1, 0, 2, 3, 4, 5]
    swapped = EventStream(*(getattr(full, field)[order] for field in ("recv_ns", "venue", "kind", "row")))
    with pytest.raises(InvalidDataError, match="tie order"):
        PreparedMarket.build(data, stream=swapped)


def test_a_prepared_market_does_not_silently_ignore_another_stream() -> None:
    prep = PreparedMarket.build(one_venue_market())
    with pytest.raises(InvalidConfigError, match="stream"):
        run_replay(prep, np.zeros(3), BUY_ONCE, scenario(), SESSION, RiskLimits(), stream=prep.stream)


def test_repeated_replays_do_not_rescan_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    prep = PreparedMarket.build(one_venue_market())

    def forbidden(*args: object) -> None:
        pytest.fail("a prepared market was redundantly revalidated")

    monkeypatch.setattr(EventStream, "validate_for", forbidden)
    for _ in range(2):
        assert run_replay(prep, np.zeros(3), BUY_ONCE, scenario(), SESSION, RiskLimits()).settlement.valid
