"""Fixed-time account statistics for replay results.

Connects the ledger-derived paths of :mod:`financeai.domains.market.metrics` with the inference
in :mod:`financeai.research.stats`:

* :func:`account_increments` samples one replay on a fixed time grid: per-venue positions in stored units,
  example cash ``C`` and the marked value ``M(t) = C(t) - sum_v q_v(t) / P_v(t)`` (inverse
  contracts, ``q`` signed stored quantity units), plus the fixed-interval increments ``dM`` *including*
  intervals without trading: an idle interval is a zero observation, not a gap, so candidates
  that trade at different times still compare on one clock.
* :func:`paired_block_bootstrap` compares candidates with an incumbent by resampling the *same*
  stationary-bootstrap blocks for every column (paired), so the common market path cancels; the
  block lengths share their random numbers (:func:`common_bootstrap_indices`), so the block-length
  sensitivity is not confounded with Monte Carlo noise.
* :func:`hac_mean` is the Newey-West test of a mean increment with an explicit lag rule.
* :func:`selection_diagnostics` wraps the deflated Sharpe ratio and CSCV/PBO and spells out the
  caveats of applying them to one exposed day.

Conventions (``docs/statistics.md`` has the formulas and worked numbers):

* Fills enter the account at **exchange arrival** (``Fills.arrival_ns``), the time the simulated
  exchange matched them, not at the acknowledgement the controller sees ``L_ack`` later. The
  account's exposure to the market starts when the exchange holds the position; the controller's
  delayed knowledge is a property of the decision process, not of the P&L. This is also the
  convention of ``metrics.position_path``/``cash_path``/``marked_value``, so every figure here
  reconciles with those paths and with ``ledger.settle``.
* ``valuation="mid"`` prices every held venue at the mid of the last book *received* at or before
  the grid time (``metrics.marked_value``'s convention, reused as is). ``valuation="exit"`` prices a
  long at that book's best bid and a short at its best ask: the top-of-book price a liquidation
  would start at, so ``M_exit <= M_mid`` whenever a position is held. Both are zero-fee, top-level
  valuation conventions, not executable liquidation values (no depth, no fee); the two coincide
  whenever the account is flat, where both equal ``C``.
* A held venue without a valid received book (none yet, a clear, a one-sided book) has no mark:
  ``M`` is NaN there and so are the two adjacent increments. A missing mark is never replaced by
  a stale or fabricated price; the statistics below refuse NaN rather than drop it, so a grid
  landing in such a gap must be moved or coarsened explicitly (``AccountIncrements.n_unmarked``).
* Reconciliation: the grid must cover every fill (flat at the first grid time, every fill arrived
  by the last), so ``sum(dM) = M(end) - 0``, and for an account that ends flat ``M(end)`` equals the
  settlement's example cash, which is ``Settlement.net_pnl`` when the account is valid. This is
  checked on every call and a disagreement is an error, not a flag.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.domains.market.engine import PreparedMarket, ReplayResult
from financeai.domains.market.metrics import (
    RECONCILE_ABS_TOL,
    RECONCILE_REL_TOL,
    cash_path,
    marked_value,
    position_path,
)
from financeai.research.stats import (
    PBO_METRICS,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    newey_west_se,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    return_moments,
)

VALUATIONS: Final = ("mid", "exit")
MAX_GRID_INTERVALS: Final = 1 << 22
"""Finest grid accepted (24 h at ~20 ms); a finer one is almost always a unit mistake (ms for ns)."""
MAX_BOOTSTRAP_ELEMENTS: Final = 1 << 26
"""Largest ``n_obs * n_boot`` accepted per call. :func:`paired_block_bootstrap` draws and consumes the
indices in ~8 MiB chunks, so this bounds its work (a few seconds per block length at the cap), not its
memory; :func:`common_bootstrap_indices` materialises 512 MiB of int64 per block length at the cap."""
DEFAULT_BLOCK_LENGTHS: Final = (5.0, 15.0, 30.0)
"""Mean block lengths in grid intervals: 5/15/30 minutes on a one-minute grid."""
SELECTION_CAVEATS: Final = (
    "One exposed sample: every column was replayed on the same, already exposed trading day, the day the "
    "selection itself was made on. The probabilities describe that in-sample evidence; they are not a fresh "
    "(out-of-sample) test and do not become one by deflation.",
    "Dependence across trials: the columns share one market path and overlapping configurations, so the "
    "trials are not independent as the DSR assumes. n_trials should be the number of effectively independent "
    "trials (cluster correlated configurations first) and must count every trial in the ledger, including "
    "failed and invalid ones; sr_variance is estimated from the columns supplied only.",
    "Dependence within a column: the PSR/DSR standard error assumes independent observations, so serially "
    "dependent increments overstate the evidence. Use hac_mean or paired_block_bootstrap for errors that "
    "allow for dependence.",
    "PBO (CSCV) re-cuts the same single sample into a few contiguous, mutually dependent blocks; on one day "
    "it is a relative indicator of selection overfitting, not a probability of out-of-sample loss.",
)

_SQRT2: Final = math.sqrt(2.0)
_CHUNK_ELEMENTS: Final = 1 << 20
"""Resample rows handled per step are sized so that each per-chunk matrix (uniforms, starts, indices,
counts) stays near 8 MiB; the bootstrap's peak memory is a few such chunks, whatever ``n_boot`` is."""
_INT64_MIN: Final = int(np.iinfo(np.int64).min)
_INT64_MAX: Final = int(np.iinfo(np.int64).max)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _frozen[S: np.generic](arr: NDArray[S]) -> NDArray[S]:
    arr.setflags(write=False)
    return arr


def _integer(name: str, value: object, *, minimum: int, maximum: int | None = _INT64_MAX) -> int:
    """A Python int in ``[minimum, maximum]``; the default int64 ceiling keeps every count and step
    representable on the numpy side (a larger Python int would end in an OverflowError deep inside numpy).
    ``maximum=None`` lifts it (seeds: ``stats.seed_from`` yields 64-bit values, which ``default_rng`` takes)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | np.integer)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = "" if maximum is None else f" <= {maximum}"
        raise InvalidConfigError(f"{name} must be an integer >= {minimum}{bound}", context={"got": value})
    return int(value)


def _clock(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or not _INT64_MIN <= value <= _INT64_MAX:
        raise InvalidConfigError(f"{name} must be an int64 of epoch nanoseconds", context={"got": value})
    return int(value)


def _real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise InvalidConfigError(f"{name} must be a real number", context={"got": value})
    out = float(value)
    if not math.isfinite(out):
        raise InvalidConfigError(f"{name} must be finite", context={"got": value})
    return out


def _matrix(x: ArrayLike, *, name: str, allow_vector: bool = False, min_cols: int = 1) -> NDArray[np.float64]:
    """A finite ``(T, K)`` float64 matrix (a 1-D series becomes one column when ``allow_vector``)."""
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidDataError(f"{name} is not numeric", context={"error": str(exc)}) from exc
    if a.ndim == 1 and allow_vector:
        a = a[:, None]
    if a.ndim != 2:
        raise InvalidDataError(f"{name} must be a (T, K) matrix of increments", context={"shape": a.shape})
    if a.shape[0] < 2:
        raise InvalidDataError(f"{name} needs at least 2 observations", context={"shape": a.shape})
    if a.shape[1] < min_cols:
        raise InvalidDataError(f"{name} needs at least {min_cols} columns", context={"shape": a.shape})
    if not bool(np.all(np.isfinite(a))):
        raise InvalidDataError(f"{name} contains non-finite values")
    return a


# ---------------------------------------------------------------------------
# Fixed-grid account path
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class AccountIncrements:
    """One replay sampled on a fixed grid (all arrays read-only).

    ``times_ns`` holds the ``n_intervals + 1`` grid times; the other paths are post-event values at
    those times (fills count from exchange arrival, books from receipt), so
    ``increments_pnl[i] = marked_pnl[i + 1] - marked_pnl[i]`` is the change of marked value over
    interval ``i``: exactly zero over an idle interval, NaN when either end is unmarked.
    """

    times_ns: NDArray[np.int64]
    step_ns: int
    valuation: str
    position_units: NDArray[np.int64]
    """``(n_intervals + 1, venues)`` signed stored quantity units held at each grid time."""
    cash: NDArray[np.float64]
    """Example cash ``C`` (compensated running sum of the fills' ``cash``) at each grid time."""
    marked_pnl: NDArray[np.float64]
    increments_pnl: NDArray[np.float64]
    net_pnl: float | None
    """The settlement's score (``None`` for an invalid account); equals ``final_pnl`` when present."""

    currency: str | None = None

    @property
    def n_intervals(self) -> int:
        return int(self.increments_pnl.shape[0])

    @property
    def n_unmarked(self) -> int:
        """Grid times at which a held venue had no valid received book (``marked_pnl`` is NaN)."""
        return int(np.count_nonzero(np.isnan(self.marked_pnl)))

    @property
    def final_pnl(self) -> float:
        return float(self.marked_pnl[-1])


def _venues(result: ReplayResult, prep: PreparedMarket) -> int:
    n = len(result.settlement.final_position_units)
    if len(prep.data) != n:
        raise InvalidConfigError(
            "prepared market does not match the replayed venues",
            context={"result_venues": n, "market_venues": len(prep.data)},
        )
    return n


def _step_at[S: np.generic](
    times: NDArray[np.int64], values: NDArray[S], grid: NDArray[np.int64], before: S | int | float
) -> NDArray[S]:
    """Sample a post-event step function (``values[i]`` in force from ``times[i]``) at ``grid``."""
    if values.shape[0] == 0:
        return np.full(grid.shape[0], before, dtype=values.dtype)
    k = np.searchsorted(times, grid, side="right") - 1
    out: NDArray[S] = np.where(k >= 0, values[np.maximum(k, 0)], before).astype(values.dtype)
    return out


def _exit_marked(
    prep: PreparedMarket, grid: NDArray[np.int64], cash: NDArray[np.float64], position: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Mark inventory with each contract convention at the best bid (long) or ask (short) of the last book
    received at or before each grid time; NaN where a held venue has no valid book. Mirrors
    ``metrics.marked_value`` step for step so the two valuations differ only in the price."""
    n = grid.shape[0]
    inv = np.zeros(n, np.float64)
    unmarked = np.zeros(n, np.bool_)
    for v in range(position.shape[1]):
        q = position[:, v]
        held = q != 0
        recv = prep.book_recv[v]
        row = np.searchsorted(recv, grid, side="right") - 1
        rr = np.maximum(row, 0)
        if recv.shape[0]:
            ok = (row >= 0) & prep.valid[v][rr]
            bid = prep.bpx[v][rr, 0].astype(np.float64) * prep.tick[v]
            ask = prep.apx[v][rr, 0].astype(np.float64) * prep.tick[v]
            price = np.where(q > 0, bid, ask)
        else:
            ok = np.zeros(n, np.bool_)
            price = np.ones(n, np.float64)
        priced = held & ok
        if prep.inverse[v]:
            inv = inv + np.divide(q * prep.cash_factor[v], price, out=np.zeros(n, np.float64), where=priced)
        else:
            inv = inv - np.where(priced, q * prep.cash_factor[v] * price, 0.0)
        unmarked |= held & ~ok
    out: NDArray[np.float64] = cash - inv
    out[unmarked] = np.nan
    return out


def account_increments(
    result: ReplayResult,
    prep: PreparedMarket,
    grid_ns: int,
    valuation: str = "mid",
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> AccountIncrements:
    """Positions, cash, marked value and fixed-interval increments of one replay on a fixed grid.

    The grid is ``start_ns, start_ns + grid_ns, ...`` up to the first point at or after ``end_ns``;
    both anchors default to the market's first and last receipt time, so every replay of one
    prepared market lands on the same grid (``increment_matrix`` then pairs them). The grid must
    cover the account: no fill may arrive at or before ``start_ns`` (the account is flat there,
    ``M = 0``) nor after the last grid time, otherwise the increments would not sum to the result.
    ``valuation`` is ``"mid"`` or ``"exit"`` (module notes). Grids finer than
    :data:`MAX_GRID_INTERVALS` intervals are refused as a unit mistake.

    Reconciliation: when the fill records end flat, the final mark must equal the settlement's
    example cash within :data:`metrics.RECONCILE_REL_TOL` relative plus
    :data:`metrics.RECONCILE_ABS_TOL` absolute (the absolute floor matters because the
    compensated path sums large opening and closing flows whose difference can be tiny); a
    settlement that disagrees with its own fill records is refused.
    """
    step = _integer("grid_ns", grid_ns, minimum=1)
    if valuation not in VALUATIONS:
        raise InvalidConfigError("unknown valuation", context={"got": valuation, "allowed": VALUATIONS})
    nv = _venues(result, prep)
    recv = prep.stream.recv_ns
    if (start_ns is None or end_ns is None) and recv.shape[0] == 0:
        raise InvalidDataError("the market has no events to anchor the grid; pass start_ns and end_ns")
    start = int(recv[0]) if start_ns is None else _clock("start_ns", start_ns)
    end = int(recv[-1]) if end_ns is None else _clock("end_ns", end_ns)
    if end <= start:
        raise InvalidConfigError("end_ns must lie after start_ns", context={"start_ns": start, "end_ns": end})
    n = (end - start + step - 1) // step
    if n > MAX_GRID_INTERVALS:
        raise InvalidConfigError(
            "grid is too fine for the span (grid_ns is in nanoseconds)",
            context={"intervals": n, "max": MAX_GRID_INTERVALS, "grid_ns": step},
        )
    if start + n * step > _INT64_MAX:  # a grid coarser than the span ends beyond end_ns; it must still be a clock time
        raise InvalidConfigError(
            "the last grid time does not fit the int64 clock",
            context={"start_ns": start, "grid_ns": step, "intervals": n},
        )
    grid = start + step * np.arange(n + 1, dtype=np.int64)
    fills = result.fills
    if len(fills):
        first, last = int(fills.arrival_ns.min()), int(fills.arrival_ns.max())
        if first <= start:
            raise InvalidConfigError(
                "a fill arrives at or before start_ns: the account is not flat at the first grid time",
                context={"start_ns": start, "first_arrival_ns": first},
            )
        if last > int(grid[-1]):
            raise InvalidConfigError(
                "a fill arrives after the last grid time: the grid does not cover the account",
                context={"last_grid_ns": int(grid[-1]), "last_arrival_ns": last},
            )
    position = np.zeros((n + 1, nv), np.int64)
    for v in range(nv):
        times, q = position_path(result, v)
        position[:, v] = _step_at(times, q, grid, 0)
    cash_times, c = cash_path(result)
    cash = _step_at(cash_times, c, grid, 0.0)
    marked = marked_value(result, prep, grid) if valuation == "mid" else _exit_marked(prep, grid, cash, position)
    increments = np.diff(marked)

    settlement = result.settlement
    final_position = tuple(int(x) for x in position[-1])
    if final_position != settlement.final_position_units:
        raise InvalidDataError(
            "settlement positions disagree with the fill records",
            context={"fills": final_position, "settlement": settlement.final_position_units},
        )
    if not any(final_position):
        total = math.fsum(settlement.cash)
        final = float(marked[-1])
        if not (
            math.isfinite(final)
            and abs(final - total) <= RECONCILE_ABS_TOL + RECONCILE_REL_TOL * max(abs(final), abs(total))
        ):
            raise InvalidDataError(
                "final marked value does not reconcile with the settlement's example cash",
                context={"final_mark_pnl": final, "settlement_cash": total},
            )
    return AccountIncrements(
        times_ns=_frozen(grid),
        step_ns=step,
        valuation=valuation,
        position_units=_frozen(position),
        cash=_frozen(cash),
        marked_pnl=_frozen(marked),
        increments_pnl=_frozen(increments),
        net_pnl=settlement.net_pnl,
        currency=settlement.currency,
    )


def increment_matrix(paths: Sequence[AccountIncrements]) -> NDArray[np.float64]:
    """``(T, K)`` increments of ``K`` paths sampled on one grid with one valuation (read-only)."""
    if not paths:
        raise InvalidConfigError("increment_matrix needs at least one path")
    head = paths[0]
    if any(path.currency != head.currency for path in paths):
        raise InvalidConfigError("increment comparisons require a common settlement currency")
    for k, path in enumerate(paths[1:], start=1):
        if (
            path.step_ns != head.step_ns
            or path.valuation != head.valuation
            or path.times_ns.shape != head.times_ns.shape
            or not np.array_equal(path.times_ns, head.times_ns)
        ):
            raise InvalidConfigError(
                "paths are not sampled on one grid with one valuation",
                context={
                    "column": k,
                    "step_ns": (head.step_ns, path.step_ns),
                    "valuation": (head.valuation, path.valuation),
                    "intervals": (head.n_intervals, path.n_intervals),
                },
            )
    out = np.ascontiguousarray(np.column_stack([path.increments_pnl for path in paths]), dtype=np.float64)
    return _frozen(out)


# ---------------------------------------------------------------------------
# Paired stationary block bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class PairedBootstrap:
    """Candidate-minus-incumbent mean increments under paired block resampling (arrays read-only).

    Per-candidate arrays have one column per input column; the incumbent's own column is exactly
    zero everywhere. Means are per grid interval; multiply by ``n_obs`` for the total over the grid.
    """

    incumbent: int
    block_lengths: tuple[float, ...]
    n_boot: int
    n_obs: int
    alpha: float
    seed: int
    mean_diff: NDArray[np.float64]
    """``(K,)`` sample mean of ``x_k - x_incumbent`` (the same for every block length)."""
    lower: NDArray[np.float64]
    """``(B, K)`` ``alpha / 2`` percentile of the resampled mean difference per block length."""
    upper: NDArray[np.float64]
    """``(B, K)`` ``1 - alpha / 2`` percentile."""
    share_le_zero: NDArray[np.float64]
    """``(B, K)`` share of resamples with a non-positive mean difference: one minus the percentile-bootstrap
    confidence that the candidate is better. Not a p-value under a null (the difference series is not
    recentred before resampling)."""
    replicates: NDArray[np.float64]
    """``(B, n_boot, K)`` resampled mean differences; row ``r`` of every column uses index row ``r``."""


def _block_length(value: object) -> float:
    length = _real("block_lengths", value)
    if length < 1.0:
        raise InvalidConfigError("block lengths must be >= 1 grid interval", context={"got": value})
    return length


def _resample_plan(
    n: object, block_lengths: Sequence[float], n_boot: object, seed: object
) -> tuple[int, tuple[float, ...], int, int]:
    size = _integer("n", n, minimum=1)
    blocks = tuple(_block_length(b) for b in block_lengths)
    if not blocks:
        raise InvalidConfigError("block_lengths must name at least one block length")
    boots = _integer("n_boot", n_boot, minimum=1)
    seed_value = _integer("seed", seed, minimum=0, maximum=None)
    if size * boots > MAX_BOOTSTRAP_ELEMENTS:
        raise InvalidConfigError(
            "resample index matrix would be too large: coarsen the grid or lower n_boot",
            context={"n_obs": size, "n_boot": boots, "max_elements": MAX_BOOTSTRAP_ELEMENTS},
        )
    return size, blocks, boots, seed_value


def _common_index_chunks(
    n: int, blocks: tuple[float, ...], n_boot: int, rng: np.random.Generator
) -> Iterator[tuple[int, list[NDArray[np.int64]]]]:
    """Yield ``(first_row, [indices per block length])`` in fixed-size row chunks (a pure function of
    ``n``, ``n_boot`` and the seed; the chunk size is a module constant, so the draws are reproducible).

    Per chunk one uniform ``u`` and one uniform start are drawn for *every* position, then each block
    length ``b`` applies its restart rule ``u < 1 / b`` to the same two arrays. Position 0 always restarts;
    a restart jumps to that position's start, otherwise the row continues circularly (Politis & Romano
    1994, the law of ``stats.stationary_bootstrap_indices``). Because the uniforms are shared, the
    restart set of a longer block length is a subset of a shorter one's and both land on the same start
    wherever both restart: common random numbers across block lengths.
    """
    rows = max(1, _CHUNK_ELEMENTS // n)
    col = np.arange(n, dtype=np.int64)
    for first in range(0, n_boot, rows):
        k = min(rows, n_boot - first)
        u = rng.random((k, n))
        starts = rng.integers(0, n, size=(k, n), dtype=np.int64)
        per_block: list[NDArray[np.int64]] = []
        for block in blocks:
            restart = u < 1.0 / block
            restart[:, 0] = True
            # Latest restart position at or before each column; index = its start + distance, mod n.
            last = np.maximum.accumulate(np.where(restart, col, 0), axis=1)
            idx = np.take_along_axis(starts, last, axis=1)
            idx += col - last
            np.subtract(idx, n, out=idx, where=idx >= n)
            per_block.append(idx)
        yield first, per_block


def common_bootstrap_indices(
    n: int, *, block_lengths: Sequence[float], n_boot: int, seed: int
) -> tuple[NDArray[np.int64], ...]:
    """One read-only ``(n_boot, n)`` stationary-bootstrap index matrix per block length, drawn with
    common random numbers across block lengths (:func:`_common_index_chunks`).

    Each matrix alone is distributed exactly as ``stats.stationary_bootstrap_indices(n, block_mean=b,
    n_boot=n_boot, seed=seed)`` (uniform first position, restart probability ``1 / b``, circular
    continuation) but is not the same draw: the uniforms and the start positions are drawn once per
    position here, so the matrices of different block lengths are nested (a longer block length only
    removes restarts and keeps their starts) and the matrix of one block length does not depend on which
    others are requested. :func:`paired_block_bootstrap` consumes exactly these indices chunk by chunk;
    materialising them here costs ``8 * len(block_lengths) * n * n_boot`` bytes, so ``n * n_boot`` above
    :data:`MAX_BOOTSTRAP_ELEMENTS` is refused.
    """
    size, blocks, boots, seed_value = _resample_plan(n, block_lengths, n_boot, seed)
    out = [np.empty((boots, size), np.int64) for _ in blocks]
    for first, per_block in _common_index_chunks(size, blocks, boots, np.random.default_rng(seed_value)):
        for b, idx in enumerate(per_block):
            out[b][first : first + idx.shape[0]] = idx
    return tuple(_frozen(m) for m in out)


def paired_block_bootstrap(
    increments: ArrayLike,
    incumbent: int,
    *,
    block_lengths: Sequence[float] = DEFAULT_BLOCK_LENGTHS,
    n_boot: int = 4_000,
    alpha: float = 0.10,
    seed: int,
) -> PairedBootstrap:
    """Stationary-bootstrap comparison of every column of ``increments`` with column ``incumbent``.

    For each mean block length ``b`` (in grid intervals) the index rows are those of
    :func:`common_bootstrap_indices` (a Politis-Romano stationary bootstrap: uniform first position,
    restart probability ``1 / b``, circular continuation) and the *same* rows resample every column,
    so the per-resample mean differences are those of the paired difference series
    ``x_k - x_incumbent``: the common market path cancels, identical columns differ by exactly zero
    in every resample, and the interval/share answer "does the candidate beat the incumbent on the
    same resampled days?". The block lengths share their random numbers too: one uniform and one
    start position are drawn per grid position and every block length applies its restart rule to
    them, so a longer block length only merges blocks of a shorter one. Part of the Monte Carlo
    noise is thereby common to the 5/15/30 rows (the replicates of neighbouring block lengths
    correlate at roughly 0.3-0.5 on iid data, so a difference between rows is less noisy than with
    independent draws, though not free of noise: raise ``n_boot`` to shrink it), and the row of one
    block length does not depend on which others were requested. ``stats.stationary_bootstrap_indices`` has the same law
    but draws its starts only where a given ``p`` restarts, so its matrices for different block
    lengths under one seed are effectively independent; ``stats.stationary_bootstrap_ci`` resamples
    one series at a time. Neither pairs columns and block lengths at once, hence the generator here.

    The resampled means are computed from per-resample index counts (``counts @ diff / T``), which
    is exact for zero columns and avoids gathering a ``(n_boot, T, K)`` array. Indices are drawn
    and consumed in row chunks of about 8 MiB each, so the peak memory is around 100 MiB (a dozen
    such chunk-sized temporaries plus the ``(B, n_boot, K)`` replicates) whatever ``n_boot`` is;
    ``T * n_boot`` above :data:`MAX_BOOTSTRAP_ELEMENTS` is still refused as a bound on the work
    (a few seconds per block length at the cap; 4 000 resamples of a one-minute day take about
    0.5 s per block length). Percentiles are ``alpha / 2`` and
    ``1 - alpha / 2`` (5 % and 95 % by default).
    """
    x = _matrix(increments, name="increments", min_cols=2)
    t, k = x.shape
    inc = _integer("incumbent", incumbent, minimum=0)
    if inc >= k:
        raise InvalidConfigError("incumbent column out of range", context={"incumbent": inc, "columns": k})
    level = _real("alpha", alpha)
    if not 0.0 < level < 1.0:
        raise InvalidConfigError("alpha must lie in (0, 1)", context={"got": level})
    _, blocks, boots, seed_value = _resample_plan(t, block_lengths, _integer("n_boot", n_boot, minimum=2), seed)
    diff = x - x[:, inc][:, None]  # exactly 0.0 for the incumbent and for any column identical to it
    replicates = np.empty((len(blocks), boots, k), np.float64)
    for first, per_block in _common_index_chunks(t, blocks, boots, np.random.default_rng(seed_value)):
        for b, part in enumerate(per_block):
            m = part.shape[0]
            flat = (np.arange(m, dtype=np.int64)[:, None] * t + part).ravel()
            counts = np.bincount(flat, minlength=m * t).reshape(m, t).astype(np.float64)
            replicates[b, first : first + m] = (counts @ diff) / t
    quantiles = np.asarray(np.quantile(replicates, [level / 2.0, 1.0 - level / 2.0], axis=1), dtype=np.float64)
    share = np.asarray((replicates <= 0.0).mean(axis=1), dtype=np.float64)
    return PairedBootstrap(
        incumbent=inc,
        block_lengths=blocks,
        n_boot=boots,
        n_obs=t,
        alpha=level,
        seed=seed_value,
        mean_diff=_frozen(np.asarray(diff.mean(axis=0), dtype=np.float64)),
        lower=_frozen(np.ascontiguousarray(quantiles[0])),
        upper=_frozen(np.ascontiguousarray(quantiles[1])),
        share_le_zero=_frozen(share),
        replicates=_frozen(replicates),
    )


# ---------------------------------------------------------------------------
# HAC mean
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class HACMean:
    """Per-column mean increment with its Newey-West standard error (arrays read-only, one per column).

    ``t_stat``/``p_value`` are NaN for a column without dispersion (an idle account has no test
    statistic); ``p_value`` is the two-sided normal probability ``2 Phi(-|t|)`` under a zero mean.
    """

    mean: NDArray[np.float64]
    std_error: NDArray[np.float64]
    t_stat: NDArray[np.float64]
    p_value: NDArray[np.float64]
    lags: int
    n_obs: int


def newey_west_lags(n_obs: int) -> int:
    """The Newey-West (1994) lag rule ``floor(4 (n / 100)^(2/9))``, capped at ``n - 1`` (as in
    ``stats.newey_west_se(lags=None)``): 4 lags at n = 100, 6 at 1,000, 8 at 10,000."""
    n = _integer("n_obs", n_obs, minimum=2)
    growth: float = (n / 100.0) ** (2.0 / 9.0)
    return min(math.floor(4.0 * growth), n - 1)


def hac_mean(increments: ArrayLike, lags: int | None = None) -> HACMean:
    """Mean of each column with a Bartlett-kernel HAC (Newey-West) standard error.

    ``increments`` is ``(T, K)`` or one series. ``lags=None`` applies :func:`newey_west_lags` and the
    lag actually used is reported, so no result depends on an implicit rule. The error is
    ``stats.newey_west_se`` (biased autocovariances, weights ``1 - j / (L + 1)``), which equals
    statsmodels' ``cov_hac(..., use_correction=False)`` for a regression on a constant.
    """
    x = _matrix(increments, name="increments", allow_vector=True)
    t, k = x.shape
    lag = newey_west_lags(t) if lags is None else _integer("lags", lags, minimum=0)
    mean = np.array([float(np.mean(x[:, j])) for j in range(k)], np.float64)
    std_error = np.array([newey_west_se(x[:, j], lag) for j in range(k)], np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(std_error > 0.0, mean / std_error, np.nan)
    # 2 Phi(-|t|) = erfc(|t| / sqrt 2), exact to the float64 floor; 2 (1 - Phi(|t|)) and NormalDist.cdf(-|t|)
    # (erf-based) are both exactly 0 beyond |t| ~ 8.3 and lose digits well before that.
    p_value = np.array(
        [math.erfc(abs(v) / _SQRT2) if math.isfinite(v) else math.nan for v in t_stat.tolist()], np.float64
    )
    return HACMean(
        mean=_frozen(mean),
        std_error=_frozen(std_error),
        t_stat=_frozen(t_stat),
        p_value=_frozen(p_value),
        lags=lag,
        n_obs=t,
    )


# ---------------------------------------------------------------------------
# Selection diagnostics
# ---------------------------------------------------------------------------


def selection_diagnostics(
    trial_matrix: ArrayLike, n_trials: int, *, selected: int | None = None, n_splits: int = 8, metric: str = "sharpe"
) -> dict[str, Any]:
    """Deflated Sharpe ratio and CSCV/PBO for a selection among the columns of ``trial_matrix``.

    ``trial_matrix`` is ``(T, N)``: one increment column per configuration *tried* (the whole
    search, not the finalists); ``n_trials`` is the ledger's trial count, at least ``N``. The
    selected column defaults to the best per-observation Sharpe ratio among the columns that traded
    (first on ties). The DSR deflates it against the expected maximum of ``n_trials`` null Sharpe
    ratios whose variance is the ``ddof=1`` variance of the ``N`` column Sharpe ratios; an idle
    column (all exactly zero) counts with Sharpe 0, ``pbo_cscv``'s convention, but is never the
    selection (no moments to deflate): when every trading column lost, the least bad one is
    deflated, and a matrix in which nothing traded is refused, as is an explicit idle ``selected``.
    A constant non-zero column is not an account that exists and is refused. PBO needs
    two columns; with one it is ``None``. The returned dict is strict-JSON-ready and carries
    :data:`SELECTION_CAVEATS` under ``"caveats"``: with one exposed day these numbers describe
    in-sample evidence, not a fresh test.
    """
    x = _matrix(trial_matrix, name="trial_matrix")
    t, k = x.shape
    trials = _integer("n_trials", n_trials, minimum=1)
    if trials < k:
        raise InvalidConfigError(
            "n_trials counts every trial in the ledger and cannot be below the columns supplied",
            context={"n_trials": trials, "columns": k},
        )
    if metric not in PBO_METRICS:
        raise InvalidConfigError("unknown CSCV metric", context={"got": metric, "allowed": PBO_METRICS})
    sharpes = np.empty(k, np.float64)
    idle = np.zeros(k, np.bool_)
    for j in range(k):
        column = x[:, j]
        if float(column.min()) == float(column.max()):
            if column[0] != 0.0:
                raise InvalidDataError(
                    "a constant non-zero increment column is not an account",
                    context={"column": j, "level": float(column[0])},
                )
            idle[j] = True
            sharpes[j] = 0.0
        else:
            sharpes[j] = return_moments(column).sharpe
    if selected is None:
        if idle.all():
            raise InvalidDataError(
                "every column is idle: no trial traded, so there is no selection to deflate", context={"columns": k}
            )
        chosen = int(np.argmax(np.where(idle, -np.inf, sharpes)))  # an idle trial is never the selection
    else:
        chosen = _integer("selected", selected, minimum=0)
        if chosen >= k:
            raise InvalidConfigError("selected column out of range", context={"selected": chosen, "columns": k})
        if idle[chosen]:
            raise InvalidDataError(
                "the selected column is idle: there is no Sharpe ratio to deflate", context={"selected": chosen}
            )
    moments = return_moments(x[:, chosen])
    sr_variance = float(np.var(sharpes, ddof=1)) if k >= 2 else 0.0
    pbo = pbo_cscv(x, n_splits, metric=metric) if k >= 2 else None
    return {
        "n_obs": t,
        "n_columns": k,
        "n_trials": trials,
        "idle_columns": int(np.count_nonzero(idle)),
        "selected": chosen,
        "mean": moments.mean,
        "std": moments.std,
        "sharpe": moments.sharpe,
        "skew": moments.skew,
        "kurt": moments.kurt,
        "sr_variance": sr_variance,
        "expected_max_sharpe": expected_max_sharpe(trials, sr_variance),
        "psr": probabilistic_sharpe_ratio(moments.sharpe, t, moments.skew, moments.kurt),
        "dsr": deflated_sharpe_ratio(moments.sharpe, trials, t, moments.skew, moments.kurt, sr_variance),
        "pbo": None if pbo is None else pbo.pbo,
        "prob_oos_loss": None if pbo is None else pbo.prob_oos_loss,
        "n_splits": None if pbo is None else pbo.n_splits,
        "n_combinations": None if pbo is None else pbo.n_combinations,
        "metric": metric,
        "caveats": list(SELECTION_CAVEATS),
    }


__all__ = [
    "DEFAULT_BLOCK_LENGTHS",
    "MAX_BOOTSTRAP_ELEMENTS",
    "MAX_GRID_INTERVALS",
    "SELECTION_CAVEATS",
    "VALUATIONS",
    "AccountIncrements",
    "HACMean",
    "PairedBootstrap",
    "account_increments",
    "common_bootstrap_indices",
    "hac_mean",
    "increment_matrix",
    "newey_west_lags",
    "paired_block_bootstrap",
    "selection_diagnostics",
]
