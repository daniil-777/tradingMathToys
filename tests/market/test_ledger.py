"""Inverse ledger: settlement result formula, exact Fraction oracle, cap and terminal validity."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.ledger import Fills, exact_score, fill_cash, settle

INSTRUMENTS = (DEMO_INVERSE_A, DEMO_INVERSE_B)


def make_fills(rows: list[tuple[int, int, int, int]], fees: tuple[Decimal, ...]) -> Fills:
    """rows: (venue, side, px_ticks, qty_units)."""
    n = len(rows)
    venue = np.array([r[0] for r in rows], np.int8)
    side = np.array([r[1] for r in rows], np.int8)
    px = np.array([r[2] for r in rows], np.int32)
    qty = np.array([r[3] for r in rows], np.int64)
    ticks = np.array([float(INSTRUMENTS[v].tick_size) for v in venue], np.float64)
    fee = np.array([float(fees[v]) for v in venue], np.float64)
    cash = np.array(
        [
            fill_cash(int(s), int(q), float(p) * t, float(f))
            for s, q, p, t, f in zip(side, qty, px, ticks, fee, strict=True)
        ],
        np.float64,
    )
    zeros = np.zeros(n, np.int64)
    return Fills(
        order_id=np.arange(n, dtype=np.int64),
        venue=venue,
        side=side,
        px_ticks=px,
        qty_units=qty,
        submit_ns=zeros,
        arrival_ns=zeros,
        ack_ns=zeros,
        cash=cash,
    )


@pytest.mark.parametrize(
    ("fee", "expected"),
    [
        (Decimal("0.00005"), Fraction(646_633, 2_680_134_000)),
        (Decimal("0.0001"), Fraction(316_633, 1_340_067_000)),
        (Decimal("0.000175"), Fraction(1_226_431, 5_360_268_000)),
        (Decimal("0.0005"), Fraction(52_633, 268_013_400)),
    ],
)
def test_exact_round_trip_fixtures(fee: Decimal, expected: Fraction) -> None:
    # Buy 1,000 stored quantity units at 20,001 and sell it at 20,100 on one venue.
    fills = make_fills([(0, 1, 40_002, 1_000), (0, -1, 40_200, 1_000)], (fee, fee))
    assert exact_score(fills, INSTRUMENTS, (fee, fee)) == expected
    result = settle(fills, INSTRUMENTS, (fee, fee), cap_units=1_000_000)
    assert result.valid
    assert result.net_pnl == pytest.approx(float(expected), rel=1e-12)


def test_long_round_trip_equals_inverse_pnl_minus_fees() -> None:
    fee = Decimal("0.0001")
    fills = make_fills([(1, 1, 40_000, 1_000), (1, -1, 42_000, 1_000)], (fee, fee))
    expected = 1_000 / 20_000 - 1_000 / 21_000 - 0.0001 * (1_000 / 20_000 + 1_000 / 21_000)
    assert settle(fills, INSTRUMENTS, (fee, fee), cap_units=10**6).net_pnl == pytest.approx(expected, rel=1e-13)


def test_open_inventory_has_no_score() -> None:
    fee = Decimal("0.0001")
    fills = make_fills([(0, 1, 40_000, 1_000), (1, -1, 40_000, 1_000)], (fee, fee))
    result = settle(fills, INSTRUMENTS, (fee, fee), cap_units=10**6)
    assert not result.valid and result.net_pnl is None
    assert result.reason == "not_flat"
    assert result.final_position_units == (1_000, -1_000)
    assert exact_score(fills, INSTRUMENTS, (fee, fee)) is None


def test_cap_breach_invalidates_the_account() -> None:
    fee = Decimal("0.0001")
    fills = make_fills([(0, 1, 40_000, 600_000), (0, 1, 40_000, 600_000), (0, -1, 40_000, 1_200_000)], (fee, fee))
    result = settle(fills, INSTRUMENTS, (fee, fee), cap_units=1_000_000)
    assert not result.valid and result.reason == "cap_exceeded"
    assert result.max_abs_position_units[0] == 1_200_000


@st.composite
def flat_fill_sequences(draw: st.DrawFn) -> list[tuple[int, int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    for venue in (0, 1):
        lot = INSTRUMENTS[venue].lot_size
        position = 0
        for _ in range(draw(st.integers(0, 12))):
            side = draw(st.sampled_from((-1, 1)))
            qty = lot * draw(st.integers(1, 2_000))
            rows.append((venue, side, draw(st.integers(30_000, 60_000)), qty))
            position += side * qty
        if position:
            rows.append((venue, -1 if position > 0 else 1, draw(st.integers(30_000, 60_000)), abs(position)))
    return draw(st.permutations(rows))


@given(
    rows=flat_fill_sequences(),
    fee_a=st.sampled_from(["0.00005", "0.000175"]),
    fee_b=st.sampled_from(["0.0001", "0.0005"]),
)
@settings(max_examples=200, deadline=None)
def test_float_settlement_matches_exact_oracle(rows: list[tuple[int, int, int, int]], fee_a: str, fee_b: str) -> None:
    fees = (Decimal(fee_a), Decimal(fee_b))
    fills = make_fills(rows, fees)
    exact = exact_score(fills, INSTRUMENTS, fees)
    result = settle(fills, INSTRUMENTS, fees, cap_units=10**12)
    assert exact is not None and result.valid and result.net_pnl is not None
    assert abs(result.net_pnl - float(exact)) <= 1e-12 * max(1.0, abs(float(exact))) + 1e-12
