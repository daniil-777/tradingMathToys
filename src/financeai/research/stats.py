"""Research statistics for strategy evaluation (NumPy and the standard library only).

Conventions shared by every function:

* Inputs are finite ``float64`` series (``T x N`` matrices for CSCV). NaN/inf is refused,
  never dropped: a failed or invalid account must not silently vanish from a statistic.
* Out-of-range tuning parameters (lags, block lengths, counts, ``alpha``, seeds) raise
  :class:`InvalidConfigError`; impossible data or estimated moments raise
  :class:`InvalidDataError`.
* Sharpe ratios are *per observation* (not annualised): ``mean / std`` with ``ddof=1``.
  Skewness and kurtosis are the biased sample moments and kurtosis is Pearson's
  (non-excess, 3 for a normal law), as in Bailey & Lopez de Prado. ``kurt < 1 + skew^2``
  is impossible for any distribution and is refused; it is the usual symptom of an excess
  kurtosis passed by mistake.
* Normal quantiles come from :class:`statistics.NormalDist` (Wichura's AS241, ~1e-16), so
  no SciPy dependency is needed.

References: Politis & Romano (1994) stationary bootstrap; Newey & West (1987, 1994);
Bailey & Lopez de Prado (2012) probabilistic and (2014) deflated Sharpe ratio; Bailey,
Borwein, Lopez de Prado & Zhu (2017) probability of backtest overfitting (CSCV).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.serialization import canonical_json

EULER_GAMMA: Final = 0.5772156649015329
PBO_METRICS: Final = ("sharpe", "mean")
MAX_CSCV_SPLITS: Final = 24
"""CSCV evaluates C(S, S/2) splits: 2.7 million at S = 24; larger S is refused."""

_NORMAL: Final = NormalDist()
_CHUNK_ELEMENTS: Final = 1 << 20
"""Upper bound on the elements of one resample/combination block (bounds peak memory)."""
_PEARSON_TOLERANCE: Final = 1e-9


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _series(x: ArrayLike, *, name: str = "x", min_len: int = 2) -> NDArray[np.float64]:
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidDataError(f"{name} is not numeric", context={"error": str(exc)}) from exc
    if a.ndim != 1:
        raise InvalidDataError(f"{name} must be 1-D", context={"shape": a.shape})
    if a.shape[0] < min_len:
        raise InvalidDataError(f"{name} needs at least {min_len} observations", context={"n": a.shape[0]})
    if not bool(np.all(np.isfinite(a))):
        raise InvalidDataError(f"{name} contains non-finite values")
    return a


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise InvalidConfigError(f"{name} must be an integer >= {minimum}", context={"got": value})
    return int(value)


def _real(name: str, value: object, error: type[InvalidDataError | InvalidConfigError]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise error(f"{name} must be a real number", context={"got": value})
    out = float(value)
    if not math.isfinite(out):
        raise error(f"{name} must be finite", context={"got": value})
    return out


def _seed(seed: object) -> int:
    return _integer("seed", seed, minimum=0)


# ---------------------------------------------------------------------------
# Long-run variance
# ---------------------------------------------------------------------------


def newey_west_se(x: ArrayLike, lags: int | None = None) -> float:
    """HAC (Bartlett-kernel) standard error of the sample mean of ``x``.

    ``S = gamma_0 + 2 * sum_{j=1..L} (1 - j / (L + 1)) * gamma_j`` with biased
    autocovariances ``gamma_j = sum_t d_t d_{t-j} / n``; the result is ``sqrt(S / n)``.
    ``lags=None`` uses the Newey-West (1994) rule ``floor(4 * (n / 100) ** (2 / 9))``.
    """
    a = _series(x)
    n = int(a.shape[0])
    if lags is None:
        lags = min(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)), n - 1)
    lag = _integer("lags", lags, minimum=0)
    if lag >= n:
        raise InvalidConfigError("lags must be smaller than the sample size", context={"lags": lag, "n": n})
    d = a - a.mean()
    total = float(d @ d)
    for j in range(1, lag + 1):
        total += 2.0 * (1.0 - j / (lag + 1.0)) * float(d[j:] @ d[:-j])
    # The Bartlett kernel is positive semi-definite, so only rounding can push S below 0.
    return math.sqrt(max(total / n, 0.0) / n)


# ---------------------------------------------------------------------------
# Sharpe-ratio inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReturnMoments:
    """Per-observation moments feeding the PSR/DSR (Pearson kurtosis, biased moments)."""

    n_obs: int
    mean: float
    std: float
    """Sample standard deviation (``ddof=1``)."""
    sharpe: float
    skew: float
    kurt: float


def return_moments(x: ArrayLike) -> ReturnMoments:
    """Sharpe ratio, skewness and (non-excess) kurtosis of a return series.

    A constant series (e.g. a zero-trade account) has no Sharpe ratio; it is refused rather
    than reported as zero.
    """
    a = _series(x)
    if float(a.min()) == float(a.max()):
        raise InvalidDataError("constant series: Sharpe ratio and higher moments are undefined")
    n = int(a.shape[0])
    mean = float(a.mean())
    d = a - mean
    d2 = d * d
    m2 = float(d2.sum()) / n
    m3 = float(d2 @ d) / n
    m4 = float(d2 @ d2) / n
    std = math.sqrt(m2 * n / (n - 1))
    return ReturnMoments(n_obs=n, mean=mean, std=std, sharpe=mean / std, skew=m3 / m2**1.5, kurt=m4 / (m2 * m2))


def _variance_factor(sr: float, skew: object, kurt: object) -> float:
    g3 = _real("skew", skew, InvalidDataError)
    g4 = _real("kurt", kurt, InvalidDataError)
    if g4 < 1.0 + g3 * g3 - _PEARSON_TOLERANCE:
        raise InvalidDataError(
            "kurtosis below 1 + skew^2 is impossible; pass Pearson (non-excess) kurtosis, 3 for a normal law",
            context={"skew": g3, "kurt": g4},
        )
    factor = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if not factor > 0.0:
        raise InvalidDataError(
            "Sharpe-ratio variance factor is not positive for these moments",
            context={"sr_hat": sr, "skew": g3, "kurt": g4},
        )
    return factor


def probabilistic_sharpe_ratio(
    sr_hat: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0
) -> float:
    """``P[SR > sr_benchmark]`` given an estimate from ``n_obs`` observations (PSR, 2012).

    ``Phi((sr_hat - sr_benchmark) * sqrt(n_obs - 1) / sqrt(1 - skew * sr_hat
    + (kurt - 1) / 4 * sr_hat^2))``; all Sharpe ratios are per observation.
    """
    sr = _real("sr_hat", sr_hat, InvalidDataError)
    benchmark = _real("sr_benchmark", sr_benchmark, InvalidDataError)
    n = _integer("n_obs", n_obs, minimum=2)
    factor = _variance_factor(sr, skew, kurt)
    return _NORMAL.cdf((sr - benchmark) * math.sqrt(n - 1) / math.sqrt(factor))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` independent null trials (DSR's ``SR0``).

    ``sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))`` with the
    Euler-Mascheroni constant ``g``. One trial involves no selection, so ``SR0 = 0`` (the
    asymptotic formula diverges there). Upper quantiles are evaluated as ``-Phi^-1(p)`` of
    the small tail probability, which keeps full precision for very large ``N``.
    """
    n = _integer("n_trials", n_trials, minimum=1)
    variance = _real("sr_variance", sr_variance, InvalidDataError)
    if variance < 0.0:
        raise InvalidDataError("sr_variance must be non-negative", context={"got": variance})
    if n == 1 or variance == 0.0:
        return 0.0
    q_n = -_NORMAL.inv_cdf(1.0 / n)
    q_ne = -_NORMAL.inv_cdf(1.0 / (n * math.e))
    return math.sqrt(variance) * ((1.0 - EULER_GAMMA) * q_n + EULER_GAMMA * q_ne)


def deflated_sharpe_ratio(
    sr_hat: float, n_trials: int, n_obs: int, skew: float, kurt: float, sr_variance: float
) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014), a probability in ``[0, 1]``.

    The PSR of ``sr_hat`` against the expected maximum of ``n_trials`` null Sharpe ratios
    whose cross-trial variance is ``sr_variance`` (same per-observation units as
    ``sr_hat``). ``n_trials`` counts *independent* trials; correlated configurations should
    be clustered first, and every trial in the ledger must be counted, including failed and
    invalid ones.
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(sr_hat, n_obs, skew, kurt, sr_benchmark=sr0)


# ---------------------------------------------------------------------------
# Stationary bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class BootstrapCI:
    """Percentile interval from a stationary bootstrap (``replicates`` is read-only)."""

    estimate: float
    lower: float
    upper: float
    std_error: float
    """Standard deviation (``ddof=1``) of the bootstrap replicates."""
    alpha: float
    block_mean: float
    n_boot: int
    replicates: NDArray[np.float64]


def _block_mean(block_mean: object) -> float:
    mean = _real("block_mean", block_mean, InvalidConfigError)
    if mean < 1.0:
        raise InvalidConfigError("block_mean must be >= 1 observation", context={"got": mean})
    return mean


def _index_blocks(n: int, p: float, n_boot: int, rng: np.random.Generator) -> Iterator[NDArray[np.int64]]:
    """Yield resample index rows in fixed-size chunks (a pure function of n, n_boot, seed).

    Each row starts at a uniform position; every later position starts a new block with
    probability ``p`` (geometric block lengths with mean ``1 / p``) at a fresh uniform
    position and otherwise continues the current block circularly (Politis & Romano 1994).
    """
    rows = max(1, _CHUNK_ELEMENTS // n)
    col = np.arange(n, dtype=np.int64)
    for first in range(0, n_boot, rows):
        k = min(rows, n_boot - first)
        restart = (rng.random((k, n)) < p).ravel()
        restart[::n] = True
        where = np.flatnonzero(restart)
        starts = rng.integers(0, n, size=where.shape[0], dtype=np.int64)
        # Block b covers positions where[b]..; index = start_b + (col - col_b) mod n. Every row
        # opens with a restart, so the running block id never spans two resamples.
        base = starts - where % n
        base[base < 0] += n
        idx = base[np.cumsum(restart, dtype=np.int64) - 1].reshape(k, n)
        idx += col
        np.subtract(idx, n, out=idx, where=idx >= n)
        yield idx


def stationary_bootstrap_indices(n: int, *, block_mean: float, n_boot: int, seed: int) -> NDArray[np.int64]:
    """``(n_boot, n)`` read-only resample indices; resample paired series with the same rows.

    :func:`stationary_bootstrap_ci` draws exactly these indices for the same arguments.
    """
    size = _integer("n", n, minimum=1)
    boots = _integer("n_boot", n_boot, minimum=1)
    p = 1.0 / _block_mean(block_mean)
    rng = np.random.default_rng(_seed(seed))
    out = np.concatenate(list(_index_blocks(size, p, boots, rng)), axis=0)
    out.setflags(write=False)
    return out


def _reduce(stat: Callable[..., Any], sample: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(stat(sample, axis=1), dtype=np.float64)
    if values.shape != (sample.shape[0],):
        raise InvalidConfigError("stat must reduce axis=1 to one value per resample", context={"shape": values.shape})
    if not bool(np.all(np.isfinite(values))):
        raise InvalidDataError("bootstrap statistic is not finite on a resample")
    return values


def stationary_bootstrap_ci(
    x: ArrayLike,
    stat: Callable[..., Any] = np.mean,
    *,
    block_mean: float,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    seed: int,
) -> BootstrapCI:
    """Politis-Romano stationary-bootstrap percentile interval for ``stat(x)``.

    ``stat`` is a NumPy-style reducer called as ``stat(samples, axis=1)`` on a
    ``(resamples, n)`` block (``np.mean``, ``np.median``, or a custom function honouring
    ``axis``); vectorising over resamples keeps the bootstrap fast. ``block_mean=1`` is the
    iid bootstrap; choose it near the dependence horizon of ``x`` otherwise. A statistic
    that is not finite on any resample is refused rather than filtered.
    """
    a = _series(x)
    mean_block = _block_mean(block_mean)
    boots = _integer("n_boot", n_boot, minimum=2)
    level = _real("alpha", alpha, InvalidConfigError)
    if not 0.0 < level < 1.0:
        raise InvalidConfigError("alpha must lie in (0, 1)", context={"got": level})
    rng = np.random.default_rng(_seed(seed))
    estimate = float(_reduce(stat, a[None, :])[0])
    replicates = np.empty(boots, dtype=np.float64)
    done = 0
    for idx in _index_blocks(int(a.shape[0]), 1.0 / mean_block, boots, rng):
        replicates[done : done + idx.shape[0]] = _reduce(stat, a[idx])
        done += idx.shape[0]
    lower, upper = np.quantile(replicates, [level / 2.0, 1.0 - level / 2.0])
    std_error = float(np.std(replicates, ddof=1))
    replicates.setflags(write=False)
    return BootstrapCI(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        std_error=std_error,
        alpha=level,
        block_mean=mean_block,
        n_boot=boots,
        replicates=replicates,
    )


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class PBOResult:
    """CSCV outcome; per-split arrays follow ``itertools.combinations`` order (read-only)."""

    pbo: float
    """Share of splits whose in-sample winner ranks at or below the out-of-sample median."""
    prob_oos_loss: float
    """Share of splits whose in-sample winner has negative out-of-sample performance."""
    logits: NDArray[np.float64]
    oos_relative_rank: NDArray[np.float64]
    is_performance: NDArray[np.float64]
    oos_performance: NDArray[np.float64]
    selected: NDArray[np.int64]
    n_splits: int
    metric: str

    @property
    def n_combinations(self) -> int:
        return int(self.logits.shape[0])


def _performance(
    metric: str,
    count: NDArray[np.float64],
    total: NDArray[np.float64],
    s1: NDArray[np.float64],
    s2: NDArray[np.float64],
    flat: NDArray[np.bool_],
    level: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-strategy mean or Sharpe ratio of one half of each split.

    ``total`` sums the raw rows. ``s1``/``s2`` are sums of ``x - center`` and its square (the
    shifted-data algorithm, which avoids catastrophic cancellation in the variance).
    ``flat`` marks halves whose raw rows all equal ``level`` (:class:`_ConstantBlocks`).
    """
    k = count[:, None]
    # A flat half's mean is its exact level, never a re-summed one: a matrix product need not
    # add every column in the same order, so equal levels could differ in the last bit.
    mean = np.where(flat, level, total / k)
    if metric == "mean":
        return mean
    var = np.maximum((s2 - s1 * s1 / k) / (k - 1.0), 0.0)
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = mean / std
    # Zero dispersion, decided on the raw rows: an idle (all-zero) half scores 0 and a constant
    # non-zero return ranks at +/- infinity instead of being dropped as NaN. Rounded moments
    # cannot decide it (centred sums of an idle half leave residues around 0); a varying half
    # whose variance still rounds to 0 is treated alike rather than becoming NaN.
    degenerate = np.where(mean == 0.0, 0.0, np.copysign(np.inf, mean))
    out: NDArray[np.float64] = np.where(flat | (std == 0.0), degenerate, sharpe)
    return out


def _dense_codes(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-column dense ranks of ``values`` as small exact floats (equal values share a code)."""
    order = np.argsort(values, axis=0, kind="stable")
    ranked = np.take_along_axis(values, order, axis=0)
    fresh = np.zeros(values.shape, dtype=np.float64)
    fresh[1:] = ranked[1:] != ranked[:-1]
    codes = np.empty_like(fresh)
    np.put_along_axis(codes, order, np.cumsum(fresh, axis=0), axis=0)
    return codes


@dataclass(frozen=True, slots=True)
class _ConstantBlocks:
    """Exact zero-dispersion test for CSCV halves, from per-block minima and maxima.

    A half is flat iff each of its blocks is constant and all share one level. Levels are
    replaced by per-strategy dense integer codes, so "one level" is tested exactly with
    integer-valued sums: for the ``n`` blocks of a half ``(sum c)^2 == n * sum c^2`` iff their
    codes are all equal (Cauchy-Schwarz), and such small integer sums are exact in float64 in
    any summation order. The level is then looked up, not re-summed. Only strategies constant
    in at least half of the blocks (``columns``) can have a flat half.
    """

    n_strategies: int
    columns: NDArray[np.intp]
    varying: NDArray[np.float64]
    """1.0 where a block of a candidate strategy is not constant."""
    codes: NDArray[np.float64]
    levels: NDArray[np.float64]
    """``levels[code, i]`` is the level that ``code`` stands for in candidate column ``i``."""

    @classmethod
    def of(cls, lo: NDArray[np.float64], hi: NDArray[np.float64], half_blocks: int) -> _ConstantBlocks:
        columns = np.flatnonzero((lo == hi).sum(axis=0) >= half_blocks)
        low = lo[:, columns]
        codes = _dense_codes(low)
        levels = np.zeros_like(low)
        np.put_along_axis(levels, codes.astype(np.intp), low, axis=0)
        return cls(
            n_strategies=int(lo.shape[1]),
            columns=columns,
            varying=(low != hi[:, columns]).astype(np.float64),
            codes=codes,
            levels=levels,
        )

    def halves(self, member: NDArray[np.float64]) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """``(flat, level)`` of each half; ``member`` is a 0/1 mask of its blocks per row."""
        flat = np.zeros((member.shape[0], self.n_strategies), dtype=np.bool_)
        level = np.zeros((member.shape[0], self.n_strategies), dtype=np.float64)
        if self.columns.size:
            n = member.sum(axis=1, keepdims=True)
            c1 = member @ self.codes
            same = (member @ self.varying == 0.0) & (c1 * c1 == n * (member @ (self.codes * self.codes)))
            # Every code of a flat half equals c1 / n exactly; elsewhere the lookup is unused.
            found = np.take_along_axis(self.levels, (c1 / n).astype(np.intp), axis=0)
            flat[:, self.columns] = same
            level[:, self.columns] = np.where(same, found, 0.0)
        return flat, level


def _matrix(perf: ArrayLike) -> NDArray[np.float64]:
    try:
        m = np.asarray(perf, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidDataError("performance matrix is not numeric", context={"error": str(exc)}) from exc
    if m.ndim != 2:
        raise InvalidDataError("performance matrix must be 2-D (T x N)", context={"shape": m.shape})
    if m.shape[1] < 2:
        raise InvalidDataError("CSCV needs at least two strategies", context={"shape": m.shape})
    if not bool(np.all(np.isfinite(m))):
        raise InvalidDataError("performance matrix contains non-finite values")
    return m


def pbo_cscv(perf_matrix: ArrayLike, n_splits: int, *, metric: str = "sharpe") -> PBOResult:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    The ``T x N`` matrix (rows: time-ordered observations, columns: the N configurations
    tried, *all* of them) is cut into ``n_splits`` contiguous blocks
    (``np.array_split``: sizes differ by at most one, no row is dropped). For each of the
    ``C(S, S/2)`` ways to choose half the blocks as in-sample, the in-sample best
    configuration (first on ties) is ranked out of sample; tied out-of-sample values
    share their average rank. With relative rank ``w = rank / (N + 1)`` the split's logit
    is ``ln(w / (1 - w))``; PBO is the share of logits ``<= 0`` (a median out-of-sample
    rank counts as overfit). Block sufficient statistics are combined by matrix products,
    so the cost is ``O(C(S, S/2) * S * N)`` rather than ``O(C(S, S/2) * T * N)``.

    A half without dispersion has no Sharpe ratio. By convention an idle half (all rows
    exactly 0) scores 0 and a constant non-zero return scores +/- infinity. This is decided
    exactly on the raw rows (block minima/maxima), never from rounded moments, so a strategy
    idle in one half and trading in the other still scores exactly 0 there, and strategies
    held at one constant level tie exactly under both metrics.
    """
    m = _matrix(perf_matrix)
    splits = _integer("n_splits", n_splits, minimum=2)
    if splits % 2:
        raise InvalidConfigError("n_splits must be even", context={"got": splits})
    if splits > MAX_CSCV_SPLITS:
        raise InvalidConfigError(
            "n_splits is too large for exhaustive CSCV", context={"got": splits, "max": MAX_CSCV_SPLITS}
        )
    t, n_strategies = (int(v) for v in m.shape)
    if splits > t:
        raise InvalidConfigError("n_splits exceeds the number of observations", context={"n_splits": splits, "rows": t})
    if metric not in PBO_METRICS:
        raise InvalidConfigError("unknown CSCV metric", context={"got": metric, "allowed": PBO_METRICS})
    blocks = np.array_split(np.arange(t), splits)
    sizes = np.array([b.shape[0] for b in blocks], dtype=np.float64)
    if metric == "sharpe" and float(np.sort(sizes)[: splits // 2].sum()) < 2.0:
        raise InvalidConfigError(
            "every half needs >= 2 observations for a Sharpe ratio", context={"rows": t, "n_splits": splits}
        )
    center = m.mean(axis=0)
    centred = m - center
    total = np.stack([m[b].sum(axis=0) for b in blocks])
    s1 = np.stack([centred[b].sum(axis=0) for b in blocks])
    s2 = np.stack([(centred[b] * centred[b]).sum(axis=0) for b in blocks])
    constant = _ConstantBlocks.of(
        np.stack([m[b].min(axis=0) for b in blocks]), np.stack([m[b].max(axis=0) for b in blocks]), splits // 2
    )

    n_comb = math.comb(splits, splits // 2)
    logits = np.empty(n_comb, dtype=np.float64)
    rel_rank = np.empty(n_comb, dtype=np.float64)
    is_perf = np.empty(n_comb, dtype=np.float64)
    oos_perf = np.empty(n_comb, dtype=np.float64)
    selected = np.empty(n_comb, dtype=np.int64)
    combos = itertools.combinations(range(splits), splits // 2)
    rows = max(1, _CHUNK_ELEMENTS // max(n_strategies, splits))
    first = 0
    while first < n_comb:
        chunk = np.array(list(itertools.islice(combos, rows)), dtype=np.intp)
        k = chunk.shape[0]
        member = np.zeros((k, splits), dtype=np.float64)
        member[np.arange(k)[:, None], chunk] = 1.0
        other = 1.0 - member
        perf_is = _performance(
            metric, member @ sizes, member @ total, member @ s1, member @ s2, *constant.halves(member)
        )
        perf_oos = _performance(metric, other @ sizes, other @ total, other @ s1, other @ s2, *constant.halves(other))
        best = np.argmax(perf_is, axis=1)
        picked = perf_oos[np.arange(k), best]
        below = (perf_oos < picked[:, None]).sum(axis=1)
        ties = (perf_oos == picked[:, None]).sum(axis=1)
        w = (below + (ties + 1) / 2.0) / (n_strategies + 1.0)
        stop = first + k
        logits[first:stop] = np.log(w / (1.0 - w))
        rel_rank[first:stop] = w
        is_perf[first:stop] = perf_is[np.arange(k), best]
        oos_perf[first:stop] = picked
        selected[first:stop] = best
        first = stop
    for arr in (logits, rel_rank, is_perf, oos_perf, selected):
        arr.setflags(write=False)
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        prob_oos_loss=float(np.mean(oos_perf < 0.0)),
        logits=logits,
        oos_relative_rank=rel_rank,
        is_performance=is_perf,
        oos_performance=oos_perf,
        selected=selected,
        n_splits=splits,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def seed_from(campaign_id: str, trial_id: str) -> int:
    """Deterministic 64-bit trial seed: first 8 bytes (big-endian) of a SHA-256.

    The digest covers the canonical JSON array ``[campaign_id, trial_id]`` so that
    ``("ab", "c")`` and ``("a", "bc")`` never collide. The value is a non-negative int below
    ``2**64``, accepted by ``np.random.default_rng``/``SeedSequence``.
    """
    for name, value in (("campaign_id", campaign_id), ("trial_id", trial_id)):
        if not isinstance(value, str) or not value:
            raise InvalidConfigError(f"{name} must be a non-empty string", context={"got": value})
    digest = hashlib.sha256(canonical_json([campaign_id, trial_id]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


__all__ = [
    "EULER_GAMMA",
    "MAX_CSCV_SPLITS",
    "PBO_METRICS",
    "BootstrapCI",
    "PBOResult",
    "ReturnMoments",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "newey_west_se",
    "pbo_cscv",
    "probabilistic_sharpe_ratio",
    "return_moments",
    "seed_from",
    "stationary_bootstrap_ci",
    "stationary_bootstrap_indices",
]
