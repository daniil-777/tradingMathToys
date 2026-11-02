"""Causal cross-venue signal recipes using depth-weighted prices and explicit spread gates.

Depth-weighted midpoints average the bid and ask VWAPs over a configurable number
of stored quantity units. Every signal uses only the receipt-time observations
available at that event. Quote prices across venues must have compatible units.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.core.jit import njit
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.stream import BOOK_EVENT, TRADE_EVENT, EventStream
from financeai.domains.market.tape import BOOK_SNAPSHOT, MarketTapes

_NS_PER_MS: Final = 1_000_000


@njit()
def _depth_weighted_mid(
    bpx: NDArray[np.int32],
    bsz: NDArray[np.int32],
    apx: NDArray[np.int32],
    asz: NDArray[np.int32],
    nb: NDArray[np.int8],
    na: NDArray[np.int8],
    kind: NDArray[np.int8],
    tick: float,
    depth: int,
    out: NDArray[np.float64],
) -> None:
    """Row usability (:func:`book_validity`: snapshot, both sides, not crossed) is evaluated
    inline rather than passed as an array: it saves four full-tape passes per call."""
    for i in range(bpx.shape[0]):
        out[i] = np.nan
        if kind[i] != BOOK_SNAPSHOT or nb[i] <= 0 or na[i] <= 0 or bpx[i, 0] >= apx[i, 0]:
            continue
        rem = depth
        acc = 0.0
        for k in range(nb[i]):
            take = min(rem, int(bsz[i, k]))
            acc += take * float(bpx[i, k])
            rem -= take
            if rem <= 0:
                break
        if rem > 0:
            continue
        bid_vwap = acc / depth
        rem = depth
        acc = 0.0
        for k in range(na[i]):
            take = min(rem, int(asz[i, k]))
            acc += take * float(apx[i, k])
            rem -= take
            if rem <= 0:
                break
        if rem > 0:
            continue
        out[i] = 0.5 * (bid_vwap + acc / depth) * tick


def depth_weighted_mid(tapes: MarketTapes, depth_units: int) -> NDArray[np.float64]:
    """Per book row: mean of bid/ask VWAPs over the first ``depth_units`` USD (NaN if thinner
    or the row is unusable, see :func:`~financeai.domains.market.stream.book_validity`)."""
    if depth_units <= 0:
        raise InvalidConfigError("depth_units must be positive")
    b = tapes.book
    out = np.empty(len(b), np.float64)
    _depth_weighted_mid(
        b.bid_px,
        b.bid_sz,
        b.ask_px,
        b.ask_sz,
        b.n_bid,
        b.n_ask,
        b.kind,
        tapes.instrument.tick_float,
        int(depth_units),
        out,
    )
    return out


@njit()
def _observed_rows(
    venue_ev: NDArray[np.int8], kind_ev: NDArray[np.int8], row_ev: NDArray[np.int64], venue: int, out: NDArray[np.int64]
) -> None:
    """Running maximum of the venue's book rows (the running maximum, not merely the latest
    row, keeps the former ``np.maximum.accumulate`` semantics for any stream)."""
    current = -1
    for e in range(venue_ev.shape[0]):
        if venue_ev[e] == venue and kind_ev[e] == BOOK_EVENT:
            r = row_ev[e]
            if r > current:
                current = r
        out[e] = current


@njit()
def _event_values(
    venue_ev: NDArray[np.int8],
    kind_ev: NDArray[np.int8],
    row_ev: NDArray[np.int64],
    venue: int,
    row_values: NDArray[np.float64],
    out: NDArray[np.float64],
) -> int:
    """``out[e] = row_values[observed row]`` (NaN before the first). Returns the first row
    that ``row_values`` cannot index, else -1 (the kernel has no bounds checks)."""
    current = -1
    n_rows = row_values.shape[0]
    for e in range(venue_ev.shape[0]):
        if venue_ev[e] == venue and kind_ev[e] == BOOK_EVENT:
            r = row_ev[e]
            if r > current:
                if r >= n_rows:
                    return int(r)
                current = r
        out[e] = row_values[current] if current >= 0 else np.nan
    return -1


@njit(error_model="numpy")
def _window_change_bps(
    values: NDArray[np.float64], recv: NDArray[np.int64], window: int, out: NDArray[np.float64]
) -> None:
    """``j`` tracks ``searchsorted(recv, recv[e] - window, side="right")``; with ``recv``
    non-decreasing it only ever moves forward, so one pass replaces 16M binary searches.
    ``error_model="numpy"`` keeps ``x / 0`` as inf/NaN like the array expression it replaced;
    the arithmetic is written in the same order so every result is bit-identical."""
    n = values.shape[0]
    j = 0
    for e in range(n):
        q = recv[e] - window
        while j < n and recv[j] <= q:
            j += 1
        past = values[j - 1] if j > 0 else np.nan
        out[e] = (values[e] / past - 1.0) * 1e4


@njit()
def _lagged_values(values: NDArray[np.float64], recv: NDArray[np.int64], lag: int, out: NDArray[np.float64]) -> None:
    """``out[e] = values`` as of ``recv[e] - lag`` (the same forward-only scan as above)."""
    n = values.shape[0]
    j = 0
    for e in range(n):
        q = recv[e] - lag
        while j < n and recv[j] <= q:
            j += 1
        out[e] = values[j - 1] if j > 0 else np.nan


def observed_rows(stream: EventStream, venue: int) -> NDArray[np.int64]:
    """Last book row of ``venue`` delivered at or before each event (-1 before the first)."""
    out = np.empty(len(stream), np.int64)
    _observed_rows(stream.venue, stream.kind, stream.row, venue, out)
    return out


def event_values(stream: EventStream, venue: int, row_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``row_values`` (one per book row of ``venue``) at the row observed by each event (NaN
    before the first); the observed-row scan and the gather are one fused pass."""
    values = np.asarray(row_values, dtype=np.float64)
    if values.ndim != 1:
        raise InvalidDataError("row_values must be one value per book row", context={"shape": values.shape})
    out = np.empty(len(stream), np.float64)
    bad = _event_values(stream.venue, stream.kind, stream.row, venue, values, out)
    if bad >= 0:
        raise IndexError(f"book row {bad} is out of bounds for {values.shape[0]} row values")
    return out


def _aligned(values: NDArray[np.float64], recv_ns: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Event-aligned float64 values and the int64 receipt clock (no copies when already so;
    narrower floats are promoted to float64, which is the contract of every caller). The
    clock must be non-decreasing (``EventStream.recv_ns`` is by construction)."""
    x = np.asarray(values, dtype=np.float64)
    t = np.asarray(recv_ns, dtype=np.int64)
    if x.shape != t.shape or x.ndim != 1:
        raise InvalidDataError(
            "values must be aligned with the receipt clock", context={"values": x.shape, "recv_ns": t.shape}
        )
    return x, t


def window_change_bps(values: NDArray[np.float64], recv_ns: NDArray[np.int64], window_ns: int) -> NDArray[np.float64]:
    """``(x_e / x(as of t_e - window) - 1) * 1e4``; NaN when no earlier value exists.

    ``recv_ns`` is the stream's (non-decreasing) receipt clock aligned with ``values``; the
    result is float64 whatever the width of ``values``.
    """
    x, t = _aligned(values, recv_ns)
    out = np.empty(x.shape[0], np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):  # the pure-Python fallback divides NumPy scalars
        _window_change_bps(x, t, int(window_ns), out)
    return out


def follower_spread_ticks(prep: PreparedMarket, venue: int) -> NDArray[np.float64]:
    obs = observed_rows(prep.stream, venue)
    have = obs >= 0
    out = np.full(len(prep.stream), np.inf)
    r = obs[have]
    valid = prep.valid[venue][r]
    spread = prep.apx[venue][r, 0].astype(np.float64) - prep.bpx[venue][r, 0].astype(np.float64)
    out[have] = np.where(valid, spread, np.inf)
    return out


def _gate(signal: NDArray[np.float64], prep: PreparedMarket, follower: int, gate_ticks: int) -> NDArray[np.float64]:
    out = np.where(np.isfinite(signal), signal, 0.0)
    if gate_ticks > 0:
        out = np.where(follower_spread_ticks(prep, follower) <= gate_ticks, out, 0.0)
    return np.ascontiguousarray(out)


def delayed(values: NDArray[np.float64], recv_ns: NDArray[np.int64], delay_ns: int) -> NDArray[np.float64]:
    """``values`` as of ``t - delay`` (NaN before any earlier value): a placebo/stress shift.

    ``recv_ns`` is the stream's (non-decreasing) receipt clock aligned with ``values``; a
    non-positive delay returns ``values`` itself, otherwise the result is float64 whatever
    the width of ``values``.
    """
    if delay_ns <= 0:
        return values
    x, t = _aligned(values, recv_ns)
    out = np.empty(x.shape[0], np.float64)
    _lagged_values(x, t, int(delay_ns), out)
    return out


def lead_gap_signal(
    prep: PreparedMarket,
    *,
    leader: int,
    follower: int,
    window_ns: int,
    depth_units: int,
    gate_ticks: int = 1,
    leader_delay_ns: int = 0,
) -> NDArray[np.float64]:
    """Leader's depth-weighted-mid change minus the follower's over the same window (bps).

    Positive values mean the leader rose relative to the follower: buy the follower.
    ``leader_delay_ns > 0`` builds the *leader-delay placebo*: the leader's history is seen
    ``delay`` late, which should destroy a genuine cross-venue lead-lag edge.
    """
    venues = prep.data.venues
    recv = prep.stream.recv_ns
    x_l = delayed(
        event_values(prep.stream, leader, depth_weighted_mid(venues[leader], depth_units)), recv, leader_delay_ns
    )
    x_f = event_values(prep.stream, follower, depth_weighted_mid(venues[follower], depth_units))
    gap = window_change_bps(x_l, recv, window_ns) - window_change_bps(x_f, recv, window_ns)
    return _gate(gap, prep, follower, gate_ticks)


@njit()
def _ewma_deviation(basis: Any, recv: Any, tau: float, out: Any) -> None:
    level = np.nan
    last_t = 0
    for i in range(basis.shape[0]):
        x = basis[i]
        out[i] = x - level if (x == x and level == level) else np.nan
        if x == x:
            if level == level:
                w = np.exp(-(recv[i] - last_t) / tau)
                level = w * level + (1.0 - w) * x
            else:
                level = x
            last_t = recv[i]


def ewma_deviation_signal(
    prep: PreparedMarket, *, leader: int, follower: int, tau_ns: int, depth_units: int, gate_ticks: int = 1
) -> NDArray[np.float64]:
    """Deviation of the cross-venue basis (bps) from its EWMA of *previous* values.

    Positive values mean the leader is rich versus the recent relation: buy the follower.
    """
    if tau_ns <= 0:
        raise InvalidConfigError("tau must be positive")
    venues = prep.data.venues
    x_l = event_values(prep.stream, leader, depth_weighted_mid(venues[leader], depth_units))
    x_f = event_values(prep.stream, follower, depth_weighted_mid(venues[follower], depth_units))
    with np.errstate(invalid="ignore", divide="ignore"):
        basis = np.ascontiguousarray((x_l / x_f - 1.0) * 1e4)
    out = np.empty(len(prep.stream), np.float64)
    _ewma_deviation(basis, prep.stream.recv_ns, float(tau_ns), out)
    return _gate(out, prep, follower, gate_ticks)


def sweep_signal(prep: PreparedMarket, *, source: int, min_units: int) -> NDArray[np.float64]:
    """+1/-1 on every event of a receipt batch whose signed aggressor USD on ``source``
    reaches ``min_units`` (Bid label = aggressive buy), else 0."""
    trades = prep.data.venues[source].trades
    out = np.zeros(len(prep.stream), np.float64)
    if len(trades) == 0:
        return out
    times, first = np.unique(trades.recv_ns, return_index=True)
    signed = np.add.reduceat(trades.side_label.astype(np.int64) * trades.sz.astype(np.int64), first)
    keep = np.abs(signed) >= min_units
    times, signs = times[keep], np.sign(signed[keep]).astype(np.float64)
    if times.size == 0:
        return out
    recv = prep.stream.recv_ns
    idx = np.searchsorted(times, recv, side="left")
    hit = (idx < times.size) & (times[np.minimum(idx, times.size - 1)] == recv)
    out[hit] = signs[idx[hit]]
    _ = TRADE_EVENT  # trades are delivered in the same receipt cohort as the decision
    return out


_SPEC_KEYS: Final = {
    "flow": {"kind", "venue", "tau_ms", "measure"},
    "lead_gap": {"kind", "leader", "follower", "window_ms", "depth_units", "gate_ticks"},
    "ewma_dev": {"kind", "leader", "follower", "tau_ms", "depth_units", "gate_ticks"},
    "sweep": {"kind", "source", "min_units"},
    "move": {"kind", "venue", "window_ms", "depth_units", "gate_ticks"},
    "quote_confirm": {"kind", "venue", "window_ms", "ticks"},
    "ofi": {"kind", "venue", "levels", "window_ms", "normalize"},
}
_OFI_NORMALISE: Final = ("none", "depth", "sign")


def quote_confirm_signal(prep: PreparedMarket, *, venue: int, window_ns: int, ticks: int) -> NDArray[np.float64]:
    """Two-sided quote confirmation: +1 when the venue's best bid AND best ask (last received book)
    both rose by at least ``ticks`` against the book as of ``t - window``, -1 when both fell by at
    least ``ticks``, 0 otherwise; NaN when either book is missing or invalid. One side moving alone
    (a spread blow-out) never confirms."""
    if ticks < 1:
        raise InvalidConfigError("quote_confirm ticks must be >= 1", context={"got": ticks})
    recv = prep.stream.recv_ns
    now = observed_rows(prep.stream, venue)
    then = np.searchsorted(prep.book_recv[venue], recv - window_ns, side="right") - 1
    ok = (now >= 0) & (then >= 0)
    ok[ok] &= prep.valid[venue][now[ok]] & prep.valid[venue][then[ok]]
    out = np.full(len(recv), np.nan)
    i, j = now[ok], then[ok]
    d_bid = prep.bpx[venue][i, 0].astype(np.int64) - prep.bpx[venue][j, 0]
    d_ask = prep.apx[venue][i, 0].astype(np.int64) - prep.apx[venue][j, 0]
    out[ok] = np.where(
        (d_bid >= ticks) & (d_ask >= ticks),
        1.0,
        np.where((d_bid <= -ticks) & (d_ask <= -ticks), -1.0, 0.0),
    )
    return np.ascontiguousarray(out)


def ofi_signal(prep: PreparedMarket, *, venue: int, levels: int, window_ns: int, normalize: str) -> NDArray[np.float64]:
    """Multi-level order-flow imbalance of ``venue`` summed over the snapshots received in
    ``(t - window, t]`` (see :mod:`financeai.strategies.flow`). ``normalize``: ``none`` (USD),
    ``depth`` (divided by the mean displayed depth over levels ``1..levels`` at the counted
    snapshots; 0 when nothing was counted) or ``sign`` (-1/0/+1). NaN before the venue's first
    observed book."""
    from financeai.strategies import flow

    if not 1 <= levels <= prep.bpx[venue].shape[1]:
        raise InvalidConfigError(
            "ofi levels must be between 1 and the recorded depth",
            context={"got": levels, "depth": int(prep.bpx[venue].shape[1])},
        )
    if normalize not in _OFI_NORMALISE:
        raise InvalidConfigError("ofi normalize must be one of none, depth, sign", context={"got": normalize})
    e, depth = flow.ofi_per_update(
        prep.bpx[venue],
        prep.bsz[venue],
        prep.apx[venue],
        prep.asz[venue],
        prep.nb[venue],
        prep.na[venue],
        prep.valid[venue],
        levels,
    )
    obs = observed_rows(prep.stream, venue)
    total, count = flow.window_sum_per_event(e, prep.book_recv[venue], obs, prep.stream.recv_ns, window_ns)
    if normalize == "sign":
        return np.ascontiguousarray(np.where(np.isnan(total), np.nan, np.sign(total)))
    if normalize == "depth":
        depth_sum, _ = flow.window_sum_per_event(depth, prep.book_recv[venue], obs, prep.stream.recv_ns, window_ns)
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = np.where(count > 0, total / (depth_sum / np.maximum(count, 1)), 0.0)
        return np.ascontiguousarray(np.where(np.isnan(total), np.nan, np.where(np.isfinite(scaled), scaled, 0.0)))
    return total


def composite_branch(prep: PreparedMarket, spec: Mapping[str, Any]) -> NDArray[np.int8]:
    """For an ``any`` composite: per event, the index of the component the composite selected
    (its largest normalised magnitude), or -1 where no component is past its threshold
    (``|value / scale| >= 1``) or the value is NaN. Attribution diagnostic; never a signal."""
    if spec.get("kind") != "any":
        raise InvalidConfigError("composite_branch needs an any-composite", context={"kind": spec.get("kind")})
    if set(spec) != {"kind", "components"} or not spec["components"]:
        raise InvalidConfigError("any-signal needs a non-empty components list")
    parts = [np.abs(build_signal(prep, c["spec"]) / float(c["scale"])) for c in spec["components"]]
    stack = np.nan_to_num(np.stack(parts), nan=0.0)
    best = stack.argmax(axis=0)
    passed = stack[best, np.arange(stack.shape[1])] >= 1.0
    return np.ascontiguousarray(np.where(passed, best, -1).astype(np.int8))


def _venue(prep: PreparedMarket, key: object) -> int:
    keys = [i.key for i in prep.data.instruments]
    if not isinstance(key, str) or key not in keys:
        raise InvalidConfigError("unknown instrument in signal spec", context={"got": key, "known": keys})
    return keys.index(key)


def _int(spec: Mapping[str, Any], name: str) -> int:
    value = spec[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidConfigError(f"{name} must be an integer", context={"got": value})
    return value


def _composite(prep: PreparedMarket, spec: Mapping[str, Any], mode: str) -> NDArray[np.float64]:
    """``any``: per event the component with the largest normalised magnitude (sign kept).
    ``all``: sign * smallest normalised magnitude when every component agrees in sign, else 0.
    Components are divided by their own ``scale`` so ``enter = 1`` means "past its threshold"."""
    if set(spec) != {"kind", "components"} or not spec["components"]:
        raise InvalidConfigError(f"{mode}-signal needs a non-empty components list")
    parts = []
    for comp in spec["components"]:
        if set(comp) != {"spec", "scale"} or not float(comp["scale"]) > 0:
            raise InvalidConfigError("each component needs spec and a positive scale")
        parts.append(build_signal(prep, comp["spec"]) / float(comp["scale"]))
    if mode == "any":
        best = parts[0]
        for x in parts[1:]:
            best = np.where(np.abs(x) > np.abs(best), x, best)
        return np.ascontiguousarray(best)
    sign = np.sign(parts[0])
    mag = np.abs(parts[0])
    for x in parts[1:]:
        sign = np.where(np.sign(x) == sign, sign, 0.0)
        mag = np.minimum(mag, np.abs(x))
    return np.ascontiguousarray(sign * mag)


_OPTIONAL: Final = {"gate", "fresh", "normalize_tau_ms"}


def build_signal(prep: PreparedMarket, spec: Mapping[str, Any]) -> NDArray[np.float64]:
    """Build a registered signal recipe from a strict spec mapping.

    Pipeline: base recipe (``leader_delay_ms`` for lead gaps) -> optional ``normalize_tau_ms``
    (divide by the causal EWMA RMS of previous values) -> optional volume ``gate`` -> optional
    ``fresh`` book-age gate. Composite kinds ``any``/``all`` combine scaled components.
    """
    kind = spec.get("kind")
    if kind in ("any", "all"):
        return _composite(prep, spec, str(kind))
    if kind not in _SPEC_KEYS:
        raise InvalidConfigError("unknown signal kind", context={"kind": kind})
    allowed = _SPEC_KEYS[kind] | _OPTIONAL | ({"leader_delay_ms"} if kind == "lead_gap" else set())
    if not (_SPEC_KEYS[kind] <= set(spec) <= allowed):
        raise InvalidConfigError(
            "signal spec keys differ", context={"expected": sorted(_SPEC_KEYS[kind]), "got": sorted(spec)}
        )
    if kind == "lead_gap":
        delay_ms = _int(spec, "leader_delay_ms") if "leader_delay_ms" in spec else 0
        out = lead_gap_signal(
            prep,
            leader=_venue(prep, spec["leader"]),
            follower=_venue(prep, spec["follower"]),
            window_ns=_int(spec, "window_ms") * _NS_PER_MS,
            depth_units=_int(spec, "depth_units"),
            gate_ticks=_int(spec, "gate_ticks"),
            leader_delay_ns=delay_ms * _NS_PER_MS,
        )
    elif kind == "ewma_dev":
        out = ewma_deviation_signal(
            prep,
            leader=_venue(prep, spec["leader"]),
            follower=_venue(prep, spec["follower"]),
            tau_ns=_int(spec, "tau_ms") * _NS_PER_MS,
            depth_units=_int(spec, "depth_units"),
            gate_ticks=_int(spec, "gate_ticks"),
        )
    elif kind == "flow":
        from financeai.strategies.linear import decayed_trade_flow

        out = decayed_trade_flow(
            prep, venue=_venue(prep, spec["venue"]), tau_ns=_int(spec, "tau_ms") * _NS_PER_MS, measure=spec["measure"]
        )
    elif kind == "move":
        v = _venue(prep, spec["venue"])
        x = event_values(prep.stream, v, depth_weighted_mid(prep.data.venues[v], _int(spec, "depth_units")))
        move = window_change_bps(x, prep.stream.recv_ns, _int(spec, "window_ms") * _NS_PER_MS)
        out = _gate(move, prep, v, _int(spec, "gate_ticks"))
    elif kind == "quote_confirm":
        out = quote_confirm_signal(
            prep,
            venue=_venue(prep, spec["venue"]),
            window_ns=_int(spec, "window_ms") * _NS_PER_MS,
            ticks=_int(spec, "ticks"),
        )
    elif kind == "ofi":
        normalize = spec["normalize"]
        if not isinstance(normalize, str):
            raise InvalidConfigError("ofi normalize must be a string", context={"got": normalize})
        out = ofi_signal(
            prep,
            venue=_venue(prep, spec["venue"]),
            levels=_int(spec, "levels"),
            window_ns=_int(spec, "window_ms") * _NS_PER_MS,
            normalize=normalize,
        )
    else:
        out = sweep_signal(prep, source=_venue(prep, spec["source"]), min_units=_int(spec, "min_units"))
    from financeai.strategies import regime

    if "normalize_tau_ms" in spec:
        out = regime.normalise(out, prep, _int(spec, "normalize_tau_ms") * _NS_PER_MS)
    if "gate" in spec:
        out = regime.apply_gate(out, prep, spec["gate"])
    if "fresh" in spec:
        fresh = spec["fresh"]
        if set(fresh) != {"max_age_ms"}:
            raise InvalidConfigError("fresh gate needs exactly max_age_ms")
        mask = regime.freshness_mask(
            prep, venues=tuple(range(len(prep.data))), max_age_ns=_int(fresh, "max_age_ms") * _NS_PER_MS
        )
        out = np.ascontiguousarray(np.where(mask, out, 0.0))
    return out
