"""Synthetic figure-QA fixture (invented values, not market data).

One whole UTC day (2024-01-01) on both venues. Drawn at 80 dpi the 24 h axis has 960 display
buckets of 90 s (plus 1 ns), and every hard case sits inside a series the display must reduce,
in the interior of its bucket, so only the matching envelope rule can keep it:

* venue_a: 4,000 small long round trips (8,000 fills; the position path is reduced), with
  - a 1 ms cap excursion to 1,000,010 USD at 12:00:37.500, between two round trips of the same
    bucket (intrabucket and sub-pixel: only the bucket-maximum rule keeps it), and
  - an open +10 USD position from 17:00 to the end of the session (an invalid ending, tiny on a
    1,000,000 USD axis; it must be the last point drawn);
* venue_b: two small flat round trips (a series too small to be reduced);
* books every 30 s on each venue; the venue_a book of 18:01:00 is unusable (no bids) while venue_a
  is held, so M is undefined (NaN) at that one change point, which has a neighbour on each side
  in its bucket (only the gap rule keeps it).
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from financeai.core.clock import NS_PER_MS, NS_PER_S
from financeai.domains.market.engine import Orders, PreparedMarket, ReplayResult
from financeai.domains.market.instruments import DEMO_INVERSE_A, DEMO_INVERSE_B
from financeai.domains.market.ledger import Fills, fill_cash, settle
from financeai.domains.market.scenario import DEMO_BASE_FEES, RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData

from ..market.synthetic import Row, ladder, tapes_from_rows

INSTRUMENTS = (DEMO_INVERSE_A, DEMO_INVERSE_B)
FEES = tuple(DEMO_BASE_FEES[i.key] for i in INSTRUMENTS)
QA_SCENARIO = Scenario("qa_base", DEMO_BASE_FEES, 5_000_000)
QA_LIMITS = RiskLimits(1_000_000)
DAY0_NS = 1_675_036_800 * NS_PER_S  # 2024-01-01T00:00:00Z
MINUTE = 60 * NS_PER_S
HOUR = 60 * MINUTE
QA_SESSION = SessionPlan(
    start_ns=DAY0_NS + 5 * MINUTE,
    entry_cutoff_ns=DAY0_NS + 23 * HOUR + 45 * MINUTE,
    flatten_start_ns=DAY0_NS + 23 * HOUR + 50 * MINUTE,
    exec_cutoff_ns=DAY0_NS + 24 * HOUR - 5 * NS_PER_S,
    end_ns=DAY0_NS + 24 * HOUR,
)
QA_DPI = 80  # 12 in x 80 dpi = 960 buckets; the placements below rely on it
EXCURSION_NS = DAY0_NS + 12 * HOUR + 37_500 * NS_PER_MS  # venue_a flat between round trips
OPEN_NS = DAY0_NS + 17 * HOUR  # after the last round trip (16:49:46)
BAD_BOOK_NS = DAY0_NS + 18 * HOUR + MINUTE
US = 1_000


BASE_TICKS = (46_000, 46_010)
BOOK_EVERY_NS = 30 * NS_PER_S


def mid_ticks(venue: int, t_ns: int) -> int:
    """Mid (in ticks) of the book in force at ``t_ns``: a slow deterministic wave."""
    k = (t_ns - DAY0_NS) // BOOK_EVERY_NS
    return BASE_TICKS[venue] + round(150 * float(np.sin(k / 70.0)))


def qa_market() -> PreparedMarket:
    venues = []
    for v, spec in enumerate(INSTRUMENTS):
        rows: list[Row] = []
        for k in range(24 * 120):  # one book every 30 s
            t_ns = DAY0_NS + k * BOOK_EVERY_NS
            mid = mid_ticks(v, t_ns)
            bids, asks = ladder(mid - 1, mid + 1, 5_000_000)
            if spec is DEMO_INVERSE_A and t_ns == BAD_BOOK_NS:
                bids = []
            rows.append((t_ns // US, (t_ns - 3_000_000) // US, bids, asks))
        venues.append(tapes_from_rows(spec, rows))
    return PreparedMarket.build(MarketData(tuple(venues)))


def qa_result(round_trips: int = 4_000) -> ReplayResult:
    """Fills of the QA day (arrival order), one order per fill, settled with a 1,000,000 cap."""
    rows: list[tuple[int, int, int, int, int]] = []  # (arrival_ns, venue, side, px, qty)

    def taker(t_ns: int, venue: int, side: int, qty: int) -> None:  # buy the ask, sell the bid
        rows.append((t_ns, venue, side, mid_ticks(venue, t_ns) + side, qty))

    for j in range(round_trips):
        t_ns = DAY0_NS + 10 * MINUTE + j * 15 * NS_PER_S
        taker(t_ns, 0, 1, 100)
        taker(t_ns + NS_PER_S, 0, -1, 100)
    taker(EXCURSION_NS, 0, 1, 1_000_010)
    taker(EXCURSION_NS + NS_PER_MS, 0, -1, 1_000_010)
    taker(OPEN_NS, 0, 1, 10)
    for t_ns, size_units in ((DAY0_NS + 6 * HOUR + 7 * NS_PER_S, 1_000), (DAY0_NS + 20 * HOUR + 7 * NS_PER_S, 2_000)):
        taker(t_ns, 1, 1, size_units)
        taker(t_ns + 5 * NS_PER_S, 1, -1, size_units)
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    arrival = np.array([r[0] for r in rows], np.int64)
    venue = np.array([r[1] for r in rows], np.int8)
    side = np.array([r[2] for r in rows], np.int8)
    px = np.array([r[3] for r in rows], np.int32)
    qty = np.array([r[4] for r in rows], np.int64)
    cash = np.array(
        [
            fill_cash(int(s), int(q), float(p) * float(INSTRUMENTS[v].tick_size), float(FEES[v]))
            for v, s, p, q in zip(venue, side, px, qty, strict=True)
        ],
        np.float64,
    )
    ids = np.arange(n, dtype=np.int64)
    submit = arrival - 5_000_000
    ack = arrival + 5_000_000
    fills = Fills(
        order_id=ids,
        venue=venue,
        side=side,
        px_ticks=px,
        qty_units=qty,
        submit_ns=submit,
        arrival_ns=arrival,
        ack_ns=ack,
        cash=cash,
    )
    orders = Orders(
        order_id=ids.copy(),
        venue=venue.copy(),
        side=side.copy(),
        qty_units=qty.copy(),
        limit_ticks=px.astype(np.int64),
        submit_ns=submit.copy(),
        arrival_ns=arrival.copy(),
        ack_ns=ack.copy(),
        filled_units=qty.copy(),
        reason=np.ones(n, np.int8),
        book_row=np.zeros(n, np.int64),
    )
    fees: tuple[Decimal, ...] = FEES
    settlement = settle(fills, INSTRUMENTS, fees, cap_units=QA_LIMITS.max_abs_position_units)
    return ReplayResult(fills=fills, orders=orders, settlement=settlement, decisions=n)
