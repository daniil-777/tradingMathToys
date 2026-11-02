"""Cost-aware linear forecasts of a follower's short-horizon return.

Features are causal per-event columns aligned with the event stream (each built lazily so
a full event-by-feature matrix never has to exist). Labels use *world truth* (the follower
book in effect at exchange-clock arrival ``t + L`` and at ``t + L + h``) and are therefore
only used for fitting on earlier blocks, never as decision inputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.jit import njit
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.stream import TRADE_EVENT
from financeai.models.ridge import RidgeModel
from financeai.strategies.leadlag import (
    depth_weighted_mid,
    event_values,
    ewma_deviation_signal,
    follower_spread_ticks,
    lead_gap_signal,
    observed_rows,
    window_change_bps,
)

_MS: Final = 1_000_000
_S: Final = 1_000_000_000
DEFAULT_FLOW_SCALE: Final = 250_000.0

Column = tuple[str, Callable[[], NDArray[np.float64]]]


@njit()
def _decayed_flow(
    recv: Any, venue_ev: Any, kind_ev: Any, row_ev: Any, side: Any, size: Any, target: int, tau: float, out: Any
) -> None:
    level = 0.0
    last_t = recv[0] if recv.shape[0] else 0
    for i in range(recv.shape[0]):
        t = recv[i]
        level *= math.exp(-(t - last_t) / tau)
        last_t = t
        if venue_ev[i] == target and kind_ev[i] == TRADE_EVENT:
            r = row_ev[i]
            level += float(side[r]) * float(size[r])
        out[i] = level


def decayed_trade_flow(prep: PreparedMarket, *, venue: int, tau_ns: int, measure: str = "units") -> NDArray[np.float64]:
    """Causal signed aggressor flow in stored units, native quantity, or quote notional.

    ``measure="notional"`` uses contract multipliers and trade prices (inverse face amounts
    are already quote notional). A positive value denotes net aggressive buying.
    """
    if isinstance(venue, bool) or not isinstance(venue, int) or not 0 <= venue < len(prep.data):
        raise InvalidConfigError("venue must index the prepared market")
    if isinstance(tau_ns, bool) or not isinstance(tau_ns, int) or tau_ns <= 0:
        raise InvalidConfigError("tau_ns must be a positive integer")
    if measure not in ("units", "native", "notional"):
        raise InvalidConfigError("flow measure must be units, native, or notional")
    s = prep.stream
    tr = prep.data.venues[venue].trades
    out = np.empty(len(s), np.float64)
    instrument = prep.data.instruments[venue]
    size = tr.sz.astype(np.float64)
    if measure == "native":
        size *= float(instrument.quantity_step)
    elif measure == "notional":
        size *= instrument.cash_factor
        if not instrument.inverse:
            size *= tr.px * instrument.tick_float
    _decayed_flow(s.recv_ns, s.venue, s.kind, s.row, tr.side_label, size, venue, float(tau_ns), out)
    return out


def depth_imbalance(prep: PreparedMarket, *, venue: int, levels: int = 5) -> NDArray[np.float64]:
    """(bid quantity - ask quantity) / total over the top ``levels`` observed levels."""
    obs = observed_rows(prep.stream, venue)
    have = obs >= 0
    out = np.full(len(prep.stream), np.nan)
    r = obs[have]
    bid = prep.bsz[venue][r, :levels].astype(np.float64).sum(axis=1)
    ask = prep.asz[venue][r, :levels].astype(np.float64).sum(axis=1)
    tot = bid + ask
    vals = np.where((tot > 0) & prep.valid[venue][r], (bid - ask) / np.where(tot > 0, tot, 1.0), np.nan)
    out[have] = vals
    return out


def feature_columns(
    prep: PreparedMarket, follower: int, leader: int, depth_units: int = 50_000, flow_scale: float = DEFAULT_FLOW_SCALE
) -> list[Column]:
    """Eighteen causal features; choose depth and flow scales in the instrument's stored units."""
    if not np.isfinite(flow_scale) or flow_scale <= 0:
        raise InvalidConfigError("flow_scale must be finite and positive")

    def _lead_gap(w: int) -> Callable[[], NDArray[np.float64]]:
        return lambda: lead_gap_signal(
            prep, leader=leader, follower=follower, window_ns=w * _MS, depth_units=depth_units, gate_ticks=0
        )

    def _ewma_dev(tau: int) -> Callable[[], NDArray[np.float64]]:
        return lambda: ewma_deviation_signal(
            prep, leader=leader, follower=follower, tau_ns=tau * _S, depth_units=depth_units, gate_ticks=0
        )

    def _own_ret(w: int) -> Callable[[], NDArray[np.float64]]:
        def own() -> NDArray[np.float64]:
            x = event_values(prep.stream, follower, depth_weighted_mid(prep.data.venues[follower], depth_units))
            return window_change_bps(x, prep.stream.recv_ns, w * _MS)

        return own

    def _flow(v: int, tau_ms: int) -> Callable[[], NDArray[np.float64]]:
        return lambda: np.tanh(decayed_trade_flow(prep, venue=v, tau_ns=tau_ms * _MS) / flow_scale)

    def _imb5(v: int) -> Callable[[], NDArray[np.float64]]:
        return lambda: depth_imbalance(prep, venue=v, levels=5)

    cols: list[Column] = []
    cols.extend((f"lead_gap_{w}ms", _lead_gap(w)) for w in (50, 100, 250, 1000))
    cols.extend((f"ewma_dev_{tau}s", _ewma_dev(tau)) for tau in (5, 20, 60))
    cols.extend((f"own_ret_{w}ms", _own_ret(w)) for w in (250, 1000, 5000))
    for v, tag in ((follower, "fol"), (leader, "lead")):
        cols.extend((f"flow_{tag}_{tau_ms}ms", _flow(v, tau_ms)) for tau_ms in (250, 1000, 5000))
    cols.extend((f"imbalance5_{tag}", _imb5(v)) for v, tag in ((follower, "fol"), (leader, "lead")))
    return cols


def follower_labels(
    prep: PreparedMarket, *, follower: int, event_idx: NDArray[np.int64], latency_ns: int, horizon_ns: int
) -> NDArray[np.float64]:
    """Follower top-of-book mid return (bps) from exchange-clock arrival to arrival + horizon."""
    env = prep.exch_env[follower]
    arrival = prep.stream.recv_ns[event_idx] + latency_ns
    r0 = np.searchsorted(env, arrival, side="right") - 1
    r1 = np.searchsorted(env, arrival + horizon_ns, side="right") - 1
    ok = (r0 >= 0) & (r1 >= 0)
    r0c, r1c = np.maximum(r0, 0), np.maximum(r1, 0)
    valid = prep.valid[follower]
    ok &= valid[r0c] & valid[r1c]
    mid0 = 0.5 * (prep.bpx[follower][r0c, 0].astype(np.float64) + prep.apx[follower][r0c, 0])
    mid1 = 0.5 * (prep.bpx[follower][r1c, 0].astype(np.float64) + prep.apx[follower][r1c, 0])
    return np.where(ok, (mid1 / mid0 - 1.0) * 1e4, np.nan)


def training_rows(
    prep: PreparedMarket,
    *,
    follower: int,
    start_ns: int,
    end_ns: int,
    horizon_ns: int,
    latency_ns: int,
    every: int = 10,
) -> NDArray[np.int64]:
    """Every ``every``-th event in [start, end) whose label also ends before ``end`` (purge)
    and where the follower quotes a one-tick spread."""
    recv = prep.stream.recv_ns
    idx = np.flatnonzero((recv >= start_ns) & (recv + latency_ns + horizon_ns < end_ns))[::every]
    tight = follower_spread_ticks(prep, follower)[idx] <= 1
    return idx[tight].astype(np.int64)


def design_matrix(columns: Sequence[Column], rows: NDArray[np.int64]) -> NDArray[np.float64]:
    return np.column_stack([fn()[rows] for _, fn in columns])


def prediction_signal(
    prep: PreparedMarket, model: RidgeModel, columns: Sequence[Column], *, follower: int, gate_ticks: int = 1
) -> NDArray[np.float64]:
    """Per-event prediction accumulated one column at a time; 0 where any feature is
    missing or (when gating) the follower spread exceeds ``gate_ticks``."""
    assert model.coef_ is not None
    pred = np.full(len(prep.stream), model.intercept_)
    ok = np.ones(len(prep.stream), bool)
    for j, (_, fn) in enumerate(columns):
        col = fn()
        finite = np.isfinite(col)
        ok &= finite
        pred += model.coef_[j] * np.where(finite, col, 0.0)
    if gate_ticks > 0:
        ok &= follower_spread_ticks(prep, follower) <= gate_ticks
    return np.ascontiguousarray(np.where(ok, pred, 0.0))
