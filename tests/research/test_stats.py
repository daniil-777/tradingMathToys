"""Research statistics: hand-calculated and published known values, bootstrap coverage on
iid/AR(1) data with fixed seeds, CSCV/PBO against a brute-force reference and stable seeds."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.research.stats import (
    PBO_METRICS,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    newey_west_se,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    return_moments,
    seed_from,
    stationary_bootstrap_ci,
    stationary_bootstrap_indices,
)

PHI = NormalDist().cdf


def ar1(rng: np.random.Generator, n: int, phi: float) -> NDArray[np.float64]:
    """Stationary AR(1) with unit innovations (started from its stationary law)."""
    eps = rng.normal(size=n)
    out = np.empty(n)
    out[0] = eps[0] / math.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


# ---------------------------------------------------------------------------
# Newey-West
# ---------------------------------------------------------------------------


def test_newey_west_matches_hand_calculation_with_one_lag() -> None:
    # x = 1..5: mean 3, gamma0 = 10/5 = 2, gamma1 = 4/5, Bartlett weight 1 - 1/2:
    # S = 2 + 2 * (1/2) * (4/5) = 14/5 and SE = sqrt(S / n) = sqrt(14/25).
    got = newey_west_se([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    assert got == pytest.approx(math.sqrt(14 / 25), rel=1e-15)


def test_newey_west_matches_hand_calculation_with_two_lags() -> None:
    # Deviations from the mean 1/2 are (1/2, -3/2, 3/2, -1/2, 5/2, -5/2), so gamma0 = 35/12,
    # gamma1 = -15/8 and gamma2 = 13/12; Bartlett weights 2/3 and 1/3 give
    # S = 35/12 + 2 * (2/3 * -15/8 + 1/3 * 13/12) = 41/36 and SE^2 = S / 6 = 41/216.
    x = [1.0, -1.0, 2.0, 0.0, 3.0, -2.0]
    assert newey_west_se(x, 2) == pytest.approx(math.sqrt(41 / 216), rel=1e-14)


def test_newey_west_with_zero_lags_is_the_iid_standard_error() -> None:
    x = np.random.default_rng(1).normal(size=257)
    assert newey_west_se(x, 0) == pytest.approx(float(np.std(x)) / math.sqrt(x.size), rel=1e-12)


def test_newey_west_default_lag_follows_the_newey_west_1994_rule() -> None:
    x = np.random.default_rng(2).normal(size=1_000)
    lags = math.floor(4.0 * (1_000 / 100.0) ** (2.0 / 9.0))
    assert newey_west_se(x) == newey_west_se(x, lags)


def test_newey_west_tracks_the_long_run_variance_of_an_ar1_process() -> None:
    # AR(1) with phi = 0.5 and unit innovations has long-run variance 1 / (1 - phi)^2 = 4.
    n = 50_000
    x = ar1(np.random.default_rng(7), n, 0.5)
    se = newey_west_se(x, 40)
    assert se == pytest.approx(math.sqrt(4.0 / n), rel=0.1)
    assert newey_west_se(x, 0) < 0.7 * se  # ignoring the autocorrelation understates the error


@pytest.mark.parametrize("lags", [-1, 5, True])
def test_newey_west_rejects_lags_outside_the_sample(lags: Any) -> None:
    with pytest.raises(InvalidConfigError):
        newey_west_se([1.0, 2.0, 3.0, 4.0, 5.0], lags)


def test_statistics_reject_non_finite_short_or_multidimensional_series() -> None:
    with pytest.raises(InvalidDataError):
        newey_west_se([1.0, float("nan"), 2.0], 1)
    with pytest.raises(InvalidDataError):
        newey_west_se([1.0], 0)
    with pytest.raises(InvalidDataError):
        stationary_bootstrap_ci(np.ones((3, 2)), block_mean=2.0, n_boot=10, seed=0)
    with pytest.raises(InvalidDataError):
        return_moments([1.0, float("inf"), 2.0])


# ---------------------------------------------------------------------------
# Sharpe-ratio inference
# ---------------------------------------------------------------------------


def test_return_moments_match_hand_calculation() -> None:
    # x = (1, 2, 3, 4, 10): mean 4, sample std sqrt(50/4); central moments m2 = 10, m3 = 36,
    # m4 = 278.8, so skew = 36 / 10^1.5 and (non-excess) kurtosis = 278.8 / 100.
    m = return_moments([1.0, 2.0, 3.0, 4.0, 10.0])
    assert m.n_obs == 5
    assert m.mean == 4.0
    assert m.std == pytest.approx(math.sqrt(12.5), rel=1e-15)
    assert m.sharpe == pytest.approx(4.0 / math.sqrt(12.5), rel=1e-15)
    assert m.skew == pytest.approx(36.0 / 10.0**1.5, rel=1e-14)
    assert m.kurt == pytest.approx(2.788, rel=1e-14)


def test_return_moments_refuse_an_undefined_sharpe_ratio() -> None:
    # A zero-trade strategy has no Sharpe ratio; it is not silently zero.
    with pytest.raises(InvalidDataError):
        return_moments(np.zeros(20))


def test_probabilistic_sharpe_ratio_matches_the_closed_form() -> None:
    # Normal returns (skew 0, kurtosis 3): z = 0.1 * sqrt(100) / sqrt(1 + 0.01 / 2).
    normal = probabilistic_sharpe_ratio(0.1, 101)
    assert normal == pytest.approx(PHI(1.0 / math.sqrt(1.005)), rel=1e-15)
    # Skewed, fat-tailed returns against a benchmark: the variance factor is
    # 1 - (-0.5)(0.2) + (6 - 1)/4 * 0.2^2 = 1.15 and z = (0.2 - 0.05) * sqrt(49) / sqrt(1.15).
    expected = PHI(0.15 * 7.0 / math.sqrt(1.15))
    got = probabilistic_sharpe_ratio(0.2, 50, skew=-0.5, kurt=6.0, sr_benchmark=0.05)
    assert got == pytest.approx(expected, rel=1e-14)
    assert probabilistic_sharpe_ratio(0.3, 10, sr_benchmark=0.3) == 0.5


def test_expected_max_sharpe_matches_the_closed_form() -> None:
    gamma = 0.5772156649015329
    inv = NormalDist().inv_cdf
    expected = 0.2 * ((1 - gamma) * inv(1 - 1 / 10) + gamma * inv(1 - 1 / (10 * math.e)))
    assert expected_max_sharpe(10, 0.04) == pytest.approx(expected, rel=1e-15)
    assert expected_max_sharpe(1, 0.04) == 0.0  # no selection among a single trial
    assert expected_max_sharpe(10, 0.0) == 0.0


def test_deflated_sharpe_ratio_reproduces_the_published_worked_example() -> None:
    # Bailey & Lopez de Prado (2014): 100 trials whose annualised Sharpe ratios have variance
    # 0.5, best annualised SR 2.5, 1,250 daily observations, skewness -3, kurtosis 10.
    # In daily units: SR0 ~ 0.1132 and DSR ~ 0.9004 (hand-recomputed: 0.113172 and 0.900397).
    sr0 = expected_max_sharpe(100, 0.5 / 250)
    dsr = deflated_sharpe_ratio(2.5 / math.sqrt(250), 100, 1250, -3.0, 10.0, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)
    assert dsr == pytest.approx(0.9004, abs=5e-5)


def test_one_trial_deflates_to_the_probabilistic_sharpe_ratio() -> None:
    dsr = deflated_sharpe_ratio(0.1, 1, 500, -1.0, 5.0, 0.3)
    assert dsr == probabilistic_sharpe_ratio(0.1, 500, -1.0, 5.0)


@given(n=st.integers(1, 10_000), extra=st.integers(1, 1_000), sr=st.floats(-0.5, 0.5), var=st.floats(0.0, 0.1))
@settings(max_examples=100, deadline=None)
def test_more_trials_never_raise_the_deflated_sharpe_ratio(n: int, extra: int, sr: float, var: float) -> None:
    fewer = deflated_sharpe_ratio(sr, n, 1_000, 0.0, 3.0, var)
    more = deflated_sharpe_ratio(sr, n + extra, 1_000, 0.0, 3.0, var)
    assert 0.0 <= more <= fewer <= 1.0


def test_sharpe_inferences_reject_impossible_moments_and_sizes() -> None:
    # With kurt >= 1 + skew^2 the variance factor is >= (1 - skew * SR / 2)^2; it vanishes only
    # for a two-point law (kurt = 1 + skew^2) with skew * SR = 2, where no inference is possible.
    with pytest.raises(InvalidDataError):
        probabilistic_sharpe_ratio(1.0, 100, skew=2.0, kurt=5.0)
    with pytest.raises(InvalidDataError):  # excess kurtosis passed by mistake: kurt < 1 + skew^2
        probabilistic_sharpe_ratio(0.1, 100, skew=0.0, kurt=0.0)
    with pytest.raises(InvalidConfigError):
        probabilistic_sharpe_ratio(0.1, 1)
    with pytest.raises(InvalidConfigError):
        deflated_sharpe_ratio(0.1, 0, 100, 0.0, 3.0, 0.01)
    with pytest.raises(InvalidDataError):
        deflated_sharpe_ratio(0.1, 10, 100, 0.0, 3.0, -0.01)
    with pytest.raises(InvalidDataError):
        deflated_sharpe_ratio(float("nan"), 10, 100, 0.0, 3.0, 0.01)


# ---------------------------------------------------------------------------
# Stationary bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_of_a_constant_series_is_degenerate() -> None:
    ci = stationary_bootstrap_ci(np.full(50, 0.25), block_mean=5.0, n_boot=200, alpha=0.1, seed=3)
    assert (ci.estimate, ci.lower, ci.upper, ci.std_error) == (0.25, 0.25, 0.25, 0.0)


def test_bootstrap_without_restarts_resamples_circular_shifts() -> None:
    # With an astronomically long mean block every resample is a rotation of the series, so
    # every replicate of the mean equals the sample mean up to summation order.
    x = np.random.default_rng(4).normal(size=97)
    ci = stationary_bootstrap_ci(x, block_mean=1e15, n_boot=300, alpha=0.05, seed=9)
    assert ci.lower == pytest.approx(float(x.mean()), abs=1e-14)
    assert ci.upper == pytest.approx(float(x.mean()), abs=1e-14)


def test_stationary_bootstrap_indices_follow_geometric_blocks() -> None:
    n, block = 1_000, 8.0
    idx = stationary_bootstrap_indices(n, block_mean=block, n_boot=400, seed=11)
    assert idx.shape == (400, n) and idx.dtype == np.int64
    assert int(idx.min()) >= 0 and int(idx.max()) < n
    assert not idx.flags.writeable
    continues = idx[:, 1:] == (idx[:, :-1] + 1) % n
    # A step continues its block unless it restarts (probability 1/block); a restart lands on
    # the successor with probability 1/n.
    assert float(continues.mean()) == pytest.approx((1 - 1 / block) + (1 / block) / n, abs=0.005)


def _scalar_indices(n: int, block_mean: float, n_boot: int, seed: int) -> NDArray[np.int64]:
    """Plain-loop oracle drawing the same random numbers in the same order (one chunk)."""
    rng = np.random.default_rng(seed)
    restart = rng.random((n_boot, n)) < 1.0 / block_mean
    restart[:, 0] = True
    starts = iter(rng.integers(0, n, size=int(restart.sum()), dtype=np.int64).tolist())
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        position = 0
        for t in range(n):
            position = next(starts) if restart[b, t] else (position + 1) % n
            out[b, t] = position
    return out


@given(n=st.integers(1, 40), block=st.floats(1.0, 30.0), n_boot=st.integers(1, 12), seed=st.integers(0, 2**32 - 1))
@settings(max_examples=60, deadline=None)
def test_stationary_bootstrap_indices_match_a_scalar_reference(n: int, block: float, n_boot: int, seed: int) -> None:
    fast = stationary_bootstrap_indices(n, block_mean=block, n_boot=n_boot, seed=seed)
    np.testing.assert_array_equal(fast, _scalar_indices(n, block, n_boot, seed))


def test_bootstrap_replicates_are_the_statistic_of_the_published_indices() -> None:
    x = np.random.default_rng(5).normal(size=64)
    ci = stationary_bootstrap_ci(x, np.median, block_mean=4.0, n_boot=150, alpha=0.1, seed=21)
    idx = stationary_bootstrap_indices(64, block_mean=4.0, n_boot=150, seed=21)
    np.testing.assert_array_equal(ci.replicates, np.median(x[idx], axis=1))
    lo, hi = np.quantile(ci.replicates, [0.05, 0.95])
    assert (ci.lower, ci.upper) == (float(lo), float(hi))
    assert ci.estimate == float(np.median(x))
    assert (ci.alpha, ci.block_mean, ci.n_boot) == (0.1, 4.0, 150)
    assert not ci.replicates.flags.writeable


def test_bootstrap_is_reproducible_from_its_seed() -> None:
    x = np.random.default_rng(6).normal(size=80)
    a = stationary_bootstrap_ci(x, block_mean=3.0, n_boot=100, seed=1)
    b = stationary_bootstrap_ci(x, block_mean=3.0, n_boot=100, seed=1)
    c = stationary_bootstrap_ci(x, block_mean=3.0, n_boot=100, seed=2)
    np.testing.assert_array_equal(a.replicates, b.replicates)
    assert not np.array_equal(a.replicates, c.replicates)


def test_iid_bootstrap_reproduces_the_moments_of_the_mean() -> None:
    # For the iid bootstrap (mean block 1) E*[mean*] = mean and Var*[mean*] = var_biased / n.
    x = np.random.default_rng(8).normal(size=200)
    ci = stationary_bootstrap_ci(x, block_mean=1.0, n_boot=20_000, alpha=0.05, seed=8)
    se = float(np.std(x)) / math.sqrt(200)
    tolerance = 4 * se / math.sqrt(20_000)
    assert float(ci.replicates.mean()) == pytest.approx(float(x.mean()), abs=tolerance)
    assert ci.std_error == pytest.approx(se, rel=0.03)


def test_bootstrap_interval_covers_the_mean_of_iid_data_at_the_nominal_rate() -> None:
    rng = np.random.default_rng(2024)
    reps, hits = 200, 0
    for rep in range(reps):
        x = rng.normal(0.0, 1.0, size=120)
        ci = stationary_bootstrap_ci(x, block_mean=1.0, n_boot=400, alpha=0.1, seed=rep)
        hits += ci.lower <= 0.0 <= ci.upper
    assert 0.84 <= hits / reps <= 0.96


def test_stationary_blocks_restore_coverage_under_autocorrelation() -> None:
    # AR(1), phi = 0.6: an iid resample understates the variance of the mean by a factor of
    # (1 + phi) / (1 - phi) = 4, so its nominal 90% interval covers only ~59% of the time.
    rng = np.random.default_rng(99)
    reps, iid_hits, block_hits = 150, 0, 0
    for rep in range(reps):
        x = ar1(rng, 400, 0.6)
        iid = stationary_bootstrap_ci(x, block_mean=1.0, n_boot=400, alpha=0.1, seed=rep)
        blocks = stationary_bootstrap_ci(x, block_mean=20.0, n_boot=400, alpha=0.1, seed=rep)
        iid_hits += iid.lower <= 0.0 <= iid.upper
        block_hits += blocks.lower <= 0.0 <= blocks.upper
    assert iid_hits / reps <= 0.72
    assert block_hits / reps >= 0.75
    assert block_hits - iid_hits >= 0.12 * reps


@pytest.mark.parametrize(
    "override",
    [
        {"block_mean": 0.5},
        {"block_mean": float("nan")},
        {"n_boot": 0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_bootstrap_rejects_invalid_parameters(override: dict[str, Any]) -> None:
    params: dict[str, Any] = {"block_mean": 2.0, "n_boot": 10, "alpha": 0.1, "seed": 0} | override
    with pytest.raises(InvalidConfigError):
        stationary_bootstrap_ci(np.arange(10.0), **params)


def test_bootstrap_refuses_a_statistic_that_is_not_finite_on_a_resample() -> None:
    def sharpe(a: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
        result: NDArray[np.float64] = a.mean(axis=axis) / a.std(axis=axis)
        return result

    x = np.zeros(31)
    x[-1] = 1.0  # most resamples are constant: 0 / 0
    with np.errstate(all="ignore"), pytest.raises(InvalidDataError):
        stationary_bootstrap_ci(x, sharpe, block_mean=2.0, n_boot=200, seed=0)


def test_bootstrap_rejects_a_statistic_that_does_not_reduce_the_resample_axis() -> None:
    identity: Callable[..., Any] = lambda a, axis: a  # noqa: E731
    with pytest.raises(InvalidConfigError):
        stationary_bootstrap_ci(np.arange(10.0), identity, block_mean=2.0, n_boot=10, seed=0)


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------

HAND_ROWS = np.array([[5, 2, 5], [5, 9, 4], [4, 6, 1], [0, 9, 2]], dtype=np.float64)


def test_pbo_matches_the_hand_worked_example() -> None:
    # Four one-row blocks, three strategies (A, B, C), metric = mean; C(4, 2) = 6 splits:
    #   IS {0,1} picks B (5.5); OOS {2,3} = (2, 7.5, 1.5): B ranks 3/3 -> logit ln(3/4 / 1/4)
    #   IS {0,2} picks A (4.5); OOS {1,3} = (2.5, 9, 3):   A ranks 1/3 -> ln(1/3)
    #   IS {0,3} picks B (5.5); OOS {1,2} = (4.5, 7.5, 2.5): rank 3 -> ln(3)
    #   IS {1,2} picks B (7.5); OOS {0,3} = (2.5, 5.5, 3.5): rank 3 -> ln(3)
    #   IS {1,3} picks B (9);   OOS {0,2} = (4.5, 4, 3):     rank 2 -> ln(1) = 0 (overfit)
    #   IS {2,3} picks B (7.5); OOS {0,1} = (5, 5.5, 4.5):   rank 3 -> ln(3)
    res = pbo_cscv(HAND_ROWS, 4, metric="mean")
    ln3 = math.log(3.0)
    np.testing.assert_allclose(res.logits, [ln3, -ln3, ln3, ln3, 0.0, ln3], rtol=0, atol=1e-15)
    assert res.selected.tolist() == [1, 0, 1, 1, 1, 1]
    np.testing.assert_allclose(res.oos_relative_rank, [0.75, 0.25, 0.75, 0.75, 0.5, 0.75])
    np.testing.assert_allclose(res.is_performance, [5.5, 4.5, 5.5, 7.5, 9.0, 7.5])
    np.testing.assert_allclose(res.oos_performance, [7.5, 2.5, 7.5, 5.5, 4.0, 5.5])
    assert res.pbo == pytest.approx(2 / 6, rel=1e-15)
    assert res.prob_oos_loss == 0.0
    assert res.n_combinations == 6 and res.n_splits == 4 and res.metric == "mean"
    assert not res.logits.flags.writeable


def test_pbo_is_one_when_in_sample_winners_always_lose_out_of_sample() -> None:
    a = np.array([2.0, 1.0, -1.0, -2.0])
    res = pbo_cscv(np.column_stack([a, -a]), 2)
    assert res.pbo == 1.0
    np.testing.assert_allclose(res.logits, [math.log(0.5)] * 2)
    assert res.prob_oos_loss == 1.0


def test_pbo_is_zero_for_a_strategy_that_dominates_every_split() -> None:
    perf = np.random.default_rng(3).normal(size=(240, 6))
    perf[:, 2] += 3.0
    res = pbo_cscv(perf, 8)
    assert res.pbo == 0.0
    assert np.all(res.selected == 2)


def test_pbo_of_pure_noise_strategies_is_near_one_half() -> None:
    perf = np.random.default_rng(12).normal(size=(800, 40))
    res = pbo_cscv(perf, 10)
    assert res.n_combinations == 252
    assert 0.25 <= res.pbo <= 0.75


def test_pbo_gives_a_zero_trade_strategy_a_zero_sharpe_ratio() -> None:
    # An idle configuration is a legitimate candidate: Sharpe 0 by convention, not NaN.
    rng = np.random.default_rng(6)
    losers = rng.normal(-1.0, 1.0, size=(64, 3))
    res = pbo_cscv(np.column_stack([np.zeros(64), losers]), 4)
    assert np.all(res.selected == 0)
    assert np.all(np.isfinite(res.logits))


def _reference_scores(sub: NDArray[np.float64], metric: str) -> NDArray[np.float64]:
    """Scores of raw rows, applying the zero-dispersion convention literally."""
    mean = sub.mean(axis=0)
    if metric == "mean":
        return np.asarray(mean)
    flat = sub.max(axis=0) == sub.min(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = mean / sub.std(axis=0, ddof=1)
    level = sub[0]
    return np.asarray(np.where(flat, np.where(level == 0.0, 0.0, np.copysign(np.inf, level)), sharpe))


def _reference_cscv(perf: NDArray[np.float64], splits: int, metric: str) -> tuple[NDArray[Any], ...]:
    """Brute-force CSCV on raw rows: logits, selected, IS and OOS performance per split."""
    blocks = np.array_split(np.arange(perf.shape[0]), splits)
    out: list[tuple[float, int, float, float]] = []
    for combo in itertools.combinations(range(splits), splits // 2):
        r_is = _reference_scores(perf[np.concatenate([blocks[i] for i in combo])], metric)
        r_oos = _reference_scores(perf[np.concatenate([blocks[i] for i in range(splits) if i not in combo])], metric)
        best = int(np.argmax(r_is))
        rank = (r_oos < r_oos[best]).sum() + ((r_oos == r_oos[best]).sum() + 1) / 2
        w = rank / (perf.shape[1] + 1)
        out.append((math.log(w / (1 - w)), best, float(r_is[best]), float(r_oos[best])))
    return tuple(np.array(column) for column in zip(*out, strict=True))


@given(
    seed=st.integers(0, 2**32 - 1),
    n_strategies=st.integers(2, 7),
    half=st.integers(1, 3),
    rows_per_block=st.integers(2, 6),
    extra=st.integers(0, 3),
    metric=st.sampled_from(["sharpe", "mean"]),
)
@settings(max_examples=60, deadline=None)
def test_pbo_matches_a_brute_force_reference(
    seed: int, n_strategies: int, half: int, rows_per_block: int, extra: int, metric: str
) -> None:
    splits = 2 * half
    rows = splits * rows_per_block + extra
    perf = np.random.default_rng(seed).normal(0.05, 1.0, size=(rows, n_strategies))
    res = pbo_cscv(perf, splits, metric=metric)
    expected = _reference_cscv(perf, splits, metric)[0]
    np.testing.assert_allclose(res.logits, expected, rtol=0, atol=1e-12)
    assert res.pbo == float(np.mean(expected <= 0.0))


@given(
    seed=st.integers(0, 2**32 - 1),
    n_strategies=st.integers(2, 6),
    half=st.integers(1, 3),
    rows_per_block=st.integers(2, 5),
    extra=st.integers(0, 3),
    metric=st.sampled_from(["sharpe", "mean"]),
)
@settings(max_examples=80, deadline=None)
def test_pbo_applies_the_idle_convention_to_partially_idle_strategies(
    seed: int, n_strategies: int, half: int, rows_per_block: int, extra: int, metric: str
) -> None:
    # Each strategy sits at one constant level (idle 0.0, or a constant non-zero return) in a
    # random subset of blocks and trades elsewhere, so its full-sample mean is not that level.
    splits = 2 * half
    rows = splits * rows_per_block + extra
    rng = np.random.default_rng(seed)
    perf = rng.normal(0.05, 1.0, size=(rows, n_strategies))
    blocks = np.array_split(np.arange(rows), splits)
    for j in range(n_strategies):
        level = float(rng.choice([0.0, 0.0, 0.1, 0.25, -0.5]))
        for b in np.flatnonzero(rng.random(splits) < 0.5):
            perf[blocks[b], j] = level
    res = pbo_cscv(perf, splits, metric=metric)
    logits, selected, is_perf, oos_perf = _reference_cscv(perf, splits, metric)
    np.testing.assert_array_equal(res.selected, selected)
    np.testing.assert_allclose(res.logits, logits, rtol=0, atol=1e-12)
    # atol=0: an idle half must score exactly 0 (infinities must match exactly as well).
    np.testing.assert_allclose(res.is_performance, is_perf, rtol=1e-9, atol=0)
    np.testing.assert_allclose(res.oos_performance, oos_perf, rtol=1e-9, atol=0)
    assert res.pbo == float(np.mean(logits <= 0.0))
    assert res.prob_oos_loss == float(np.mean(oos_perf < 0.0))


def test_pbo_scores_an_idle_half_zero_even_when_the_strategy_trades_elsewhere() -> None:
    # Strategy 0 is idle (exactly 0.0) in the first half and trades in the second, so its
    # full-sample mean is not 0. Its idle half must still score exactly 0 under both metrics:
    # never a rounding residue of centred sums, nor +/- infinity derived from one.
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 60))
        active = np.concatenate([np.zeros(n), rng.normal(0.3, 1.0, n)])
        perf = np.column_stack([active, rng.normal(-3.0, 0.5, size=(2 * n, 3))])
        for metric in PBO_METRICS:
            res = pbo_cscv(perf, 2, metric=metric)
            assert res.selected.tolist() == [0, 0]
            assert res.is_performance[0] == 0.0  # split 0 picks the idle half in sample
            assert res.oos_performance[1] == 0.0  # split 1 meets it out of sample
            # Out of sample strategy 0 ranks 4 of 4 above the losers: w = 4 / 5, logit ln 4.
            np.testing.assert_allclose(res.logits, [math.log(4.0)] * 2, rtol=1e-12)


def test_idle_strategies_tie_exactly_out_of_sample() -> None:
    rng = np.random.default_rng(7)
    a = np.concatenate([np.zeros(25), rng.normal(1.0, 1.0, 25)])
    b = np.concatenate([np.zeros(25), rng.normal(0.5, 1.0, 25)])
    loser = rng.normal(-3.0, 0.5, 50)
    for metric in PBO_METRICS:
        res = pbo_cscv(np.column_stack([a, b, loser]), 2, metric=metric)
        # Split 1 picks a or b in sample; out of sample (the idle half) both score exactly 0 and
        # tie above the loser: rank 1 + (2 + 1) / 2 = 2.5 of 3 strategies, so w = 2.5 / 4.
        assert int(res.selected[1]) in (0, 1) and res.oos_performance[1] == 0.0
        assert res.oos_relative_rank[1] == 2.5 / 4.0


def test_strategies_held_at_one_constant_level_tie_exactly() -> None:
    # 0.1 is not a dyadic fraction, so re-summing it rounds, and a matrix product need not add
    # every column in the same order: the level of a flat half must be taken exactly, so that
    # identical strategies tie (and the first wins in sample) under both metrics.
    rng = np.random.default_rng(3)
    perf = rng.normal(-1.0, 1.0, size=(36, 6))
    perf[:, 3] = 0.1
    perf[:, 5] = 0.1
    for metric in PBO_METRICS:
        res = pbo_cscv(perf, 8, metric=metric)
        logits, selected, is_perf, oos_perf = _reference_cscv(perf, 8, metric)
        np.testing.assert_array_equal(res.selected, selected)
        np.testing.assert_allclose(res.logits, logits, rtol=0, atol=1e-12)
        np.testing.assert_allclose(res.is_performance, is_perf, rtol=1e-12, atol=0)
        np.testing.assert_allclose(res.oos_performance, oos_perf, rtol=1e-12, atol=0)


def test_pbo_rejects_invalid_layouts() -> None:
    perf = np.random.default_rng(0).normal(size=(12, 3))
    for splits in (3, 0, 14, True):
        with pytest.raises(InvalidConfigError):
            pbo_cscv(perf, splits)
    with pytest.raises(InvalidConfigError):
        pbo_cscv(perf, 4, metric="sortino")
    with pytest.raises(InvalidConfigError):  # two one-row halves cannot estimate a Sharpe ratio
        pbo_cscv(perf[:2], 2)
    with pytest.raises(InvalidDataError):
        pbo_cscv(perf[:, :1], 4)
    with pytest.raises(InvalidDataError):
        pbo_cscv(np.full((8, 2), np.nan), 2)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def test_seed_from_is_pinned_to_the_first_eight_bytes_of_sha256() -> None:
    assert seed_from("campaign-1", "trial-1") == 10_901_091_246_030_655_521
    digest = hashlib.sha256(b'["campaign-1","trial-1"]').digest()
    assert seed_from("campaign-1", "trial-1") == int.from_bytes(digest[:8], "big")


@given(a=st.text(min_size=1), b=st.text(min_size=1), c=st.text(min_size=1), d=st.text(min_size=1))
@settings(max_examples=200, deadline=None)
def test_seed_from_separates_its_arguments(a: str, b: str, c: str, d: str) -> None:
    seed = seed_from(a, b)
    assert 0 <= seed < 2**64
    if (a, b) != (c, d):
        assert seed != seed_from(c, d)


def test_seed_from_does_not_confuse_concatenations() -> None:
    assert seed_from("ab", "c") != seed_from("a", "bc")


def test_seed_from_feeds_numpy_generators_reproducibly() -> None:
    seed = seed_from("campaign-1", "trial-2")
    first = np.random.default_rng(seed).integers(1 << 30, size=4)
    np.testing.assert_array_equal(first, np.random.default_rng(seed).integers(1 << 30, size=4))


def test_seed_from_rejects_empty_or_non_text_identifiers() -> None:
    with pytest.raises(InvalidConfigError):
        seed_from("", "trial")
    with pytest.raises(InvalidConfigError):
        seed_from("campaign", 3)  # type: ignore[arg-type]
