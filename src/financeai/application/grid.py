"""Parallel evaluation of registered policy grids over session windows and scenarios.

Each experiment = (signal recipe, threshold-policy parameters, order latency). For every session
window (development blocks and the full day) the experiment is replayed once; because fees never
influence decisions, the resulting fills are re-scored *exactly* under every fee schedule
instead of replaying again. Workers load the memory-mapped tape caches once, compute each
signal recipe once, and return plain records for the trial ledger.
"""

from __future__ import annotations

import dataclasses
import math
import multiprocessing as mp
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from financeai.application.strategy import policy_config_id
from financeai.core.hashing import sha256_json
from financeai.domains.market.cache import load_tapes
from financeai.domains.market.engine import REASON_ENTRY, PreparedMarket, ThresholdParams, run_replay
from financeai.domains.market.instruments import InstrumentSpec
from financeai.domains.market.ledger import Fills, Settlement, settle
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan
from financeai.domains.market.stream import MarketData
from financeai.strategies.leadlag import build_signal

NS_PER_MS: Final = 1_000_000


@dataclass(frozen=True)
class GridExperiment:
    family: str
    signal: Mapping[str, Any]
    params: ThresholdParams
    latency_ms: int
    exit_signal: Mapping[str, Any] | None = None
    """Optional separate exit-signal recipe (e.g. gap closure); part of the configuration."""

    @property
    def signal_id(self) -> str:
        return sha256_json(dict(self.signal))[:16]

    @property
    def config_id(self) -> str:
        """Policy configuration identity (latency is a scenario dimension, not a config). Policy
        fields added after earlier rounds were registered are left out at their default value,
        so earlier identities (and registered experiment hashes) stay valid."""
        return policy_config_id(self.signal, self.params, self.exit_signal)


@dataclass(frozen=True)
class Window:
    name: str
    plan: SessionPlan


def with_fees(
    fills: Fills,
    instruments: Sequence[InstrumentSpec],
    fees: Sequence[Decimal],
    fees_per_unit: Sequence[Decimal] | None = None,
) -> Fills:
    """The same fills with cash recomputed under another fee schedule."""
    fixed = tuple(fees_per_unit) if fees_per_unit is not None else (Decimal(0),) * len(instruments)
    if len(fees) != len(instruments) or len(fixed) != len(instruments):
        raise ValueError("fee arrays must match the instruments")
    tick = np.array([float(i.tick_size) for i in instruments])[fills.venue]
    fee = np.array([float(f) for f in fees])[fills.venue]
    side = fills.side.astype(np.float64)
    factor = np.array([i.cash_factor for i in instruments])[fills.venue]
    inverse = np.array([i.inverse for i in instruments])[fills.venue]
    price = fills.px_ticks * tick
    exposure = fills.qty_units * factor
    cash = np.where(inverse, (side - fee) * exposure / price, (-side - fee) * exposure * price)
    fixed_unit = np.array([float(f) * float(i.quantity_step) for f, i in zip(fixed, instruments, strict=True)])
    cash -= fills.qty_units * fixed_unit[fills.venue]
    return dataclasses.replace(fills, cash=np.ascontiguousarray(cash))


def rescore(
    fills: Fills,
    instruments: Sequence[InstrumentSpec],
    fees: Sequence[Decimal],
    cap_units: int,
    fees_per_unit: Sequence[Decimal] | None = None,
) -> Settlement:
    """Settle the same fills under another fee schedule (exact: decisions ignore fees)."""
    return settle(
        with_fees(fills, instruments, fees, fees_per_unit),
        instruments,
        fees,
        cap_units=cap_units,
        fees_per_unit=fees_per_unit,
    )


def _episodes(fills: Fills, n_venues: int) -> list[tuple[int, float, bool]]:
    """Per-venue round trips (flat -> flat) as (first fill's submit time, net settlement cash, closed)."""
    out: list[tuple[int, float, bool]] = []
    for v in range(n_venues):
        m = fills.venue == v
        pos, acc, start = 0, 0.0, 0
        for side, qty, cash, sub in zip(
            fills.side[m].tolist(),
            fills.qty_units[m].tolist(),
            fills.cash[m].tolist(),
            fills.submit_ns[m].tolist(),
            strict=True,
        ):
            if pos == 0:
                start = sub
            pos += side * qty
            acc += cash
            if pos == 0:
                out.append((start, acc, True))
                acc = 0.0
        if pos != 0:
            out.append((start, acc, False))
    return out


def episode_nets(fills: Fills, n_venues: int) -> list[tuple[int, float]]:
    """(start time, net settlement cash) of every per-venue round trip, venue by venue; an episode still
    open at the end is included (its account is invalid anyway), so the nets sum to the account."""
    return [(start, net) for start, net, _ in _episodes(fills, n_venues)]


def concentration(nets: Sequence[float]) -> dict[str, Any]:
    """How much of an account one or two episodes carry. ``net_ex_top1_pnl`` removes the best
    episode only when it made money, so a losing account is never flattered by trimming."""
    ordered = sorted(nets, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    return {
        "episodes": len(ordered),
        "top1_pnl": top1,
        "top2_pnl": math.fsum(ordered[:2]),
        "net_ex_top1_pnl": math.fsum(ordered) - max(top1, 0.0),
    }


def attribute_episodes(
    fills: Fills, n_venues: int, blocks: Sequence[tuple[str, int, int]]
) -> tuple[list[float], list[int]]:
    """Split a full-day account into blocks: each per-venue round trip (flat -> flat) goes to
    the block containing its first fill's submit time. Returns (net settlement cash, episodes) per block;
    an episode still open at the end is attributed too (its account is invalid anyway)."""
    nets = [0.0] * len(blocks)
    counts = [0] * len(blocks)

    def block_of(t: int) -> int:
        for k, (_, lo, hi) in enumerate(blocks):
            if lo <= t < hi:
                return k
        return len(blocks) - 1 if t >= blocks[-1][2] else 0

    for start, net, closed in _episodes(fills, n_venues):
        k = block_of(start)
        nets[k] += net
        counts[k] += closed
    return nets, counts


def block_windows(
    day_start_ns: int,
    *,
    hours: int = 6,
    warmup_ns: int = 300 * 10**9,
    entry_stop_ns: int = 900 * 10**9,
    flatten_ns: int = 600 * 10**9,
    exec_stop_ns: int = 5 * 10**9,
) -> list[Window]:
    """Four 6-hour development blocks plus the full day, each a flat-start account."""
    hour = 3_600 * 10**9
    spans = [
        (f"B{k + 1}", day_start_ns + k * hours * hour, day_start_ns + (k + 1) * hours * hour)
        for k in range(24 // hours)
    ]
    spans.append(("FULL", day_start_ns, day_start_ns + 24 * hour))
    return [
        Window(
            name,
            SessionPlan.for_window(
                lo,
                hi,
                warmup_ns=warmup_ns,
                entry_stop_ns=entry_stop_ns,
                flatten_ns=flatten_ns,
                exec_stop_ns=exec_stop_ns,
            ),
        )
        for name, lo, hi in spans
    ]


_WORKER: dict[str, Any] = {}


def _init_worker(cache_dirs: Sequence[str]) -> None:
    data = MarketData(tuple(load_tapes(d) for d in cache_dirs))
    _WORKER["prep"] = PreparedMarket.build(data)
    _WORKER["signals"] = {}


def _signal(spec: Mapping[str, Any]) -> NDArray[np.float64]:
    key = sha256_json(dict(spec))
    cache: dict[str, NDArray[np.float64]] = _WORKER["signals"]
    if key not in cache:
        if len(cache) >= 4:
            cache.pop(next(iter(cache)))
        cache[key] = build_signal(_WORKER["prep"], spec)
    return cache[key]


def evaluate_experiment(
    prep: PreparedMarket,
    signal: np.ndarray[Any, Any],
    experiment: GridExperiment,
    windows: Sequence[Window],
    fee_sets: Mapping[str, Mapping[str, Decimal]],
    cap_units: int,
    *,
    attribution: Sequence[tuple[str, int, int]] | None = None,
    exit_signal: np.ndarray[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    """Replay ``experiment`` per window; with ``attribution`` also emit one row per block derived
    from each window's account by episode attribution (no extra replays)."""
    instruments = prep.data.instruments
    if len({i.settlement_currency for i in instruments}) > 1:
        raise ValueError("grid comparisons require a common settlement currency")
    first_fees = next(iter(fee_sets.values()))
    scenario = Scenario(f"lat{experiment.latency_ms}", first_fees, experiment.latency_ms * NS_PER_MS)
    rows: list[dict[str, Any]] = []
    for window in windows:
        res = run_replay(
            prep, signal, experiment.params, scenario, window.plan, RiskLimits(cap_units), exit_signal=exit_signal
        )
        entries = (res.orders.reason == REASON_ENTRY) & (res.orders.filled_units > 0)
        for fee_name, fees in fee_sets.items():
            fee_list = [fees[i.key] for i in instruments]
            rescored = with_fees(res.fills, instruments, fee_list)
            st = settle(rescored, instruments, fee_list, cap_units=cap_units)
            conc = concentration([n for _, n in episode_nets(rescored, len(instruments))])
            if not st.valid:
                conc["net_ex_top1_pnl"] = None
            base = {
                "config_id": experiment.config_id,
                "family": experiment.family,
                "signal_id": experiment.signal_id,
                "signal": dict(experiment.signal),
                "params": dataclasses.asdict(experiment.params),
                "latency_ms": experiment.latency_ms,
                "fees": fee_name,
                "valid": st.valid,
                "reason": st.reason,
            }
            if experiment.exit_signal is not None:
                base["exit_signal"] = dict(experiment.exit_signal)
            rows.append(
                {
                    **base,
                    "window": window.name,
                    "net_pnl": st.net_pnl if st.valid else None,
                    "fees_paid": math.fsum(st.fees),
                    "orders": len(res.orders),
                    "fills": len(res.fills),
                    "round_trips": int(entries.sum()),
                    "traded_units": int(sum(st.traded_units)),
                    "max_abs_position_units": list(st.max_abs_position_units),
                    **conc,
                }
            )
            if attribution:
                nets, counts = attribute_episodes(rescored, len(instruments), attribution)
                for (name, _, _), net, count in zip(attribution, nets, counts, strict=True):
                    rows.append(
                        {
                            **base,
                            "window": name,
                            "net_pnl": net if st.valid else None,
                            "round_trips": count,
                            "attributed_from": window.name,
                        }
                    )
    return rows


def _run_group(
    tasks: Sequence[GridExperiment],
    windows: Sequence[Window],
    fee_sets: Mapping[str, Mapping[str, Decimal]],
    cap_units: int,
    attribution: Sequence[tuple[str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    prep: PreparedMarket = _WORKER["prep"]
    out: list[dict[str, Any]] = []
    for experiment in tasks:
        xsig = _signal(experiment.exit_signal) if experiment.exit_signal is not None else None
        out.extend(
            evaluate_experiment(
                prep,
                _signal(experiment.signal),
                experiment,
                windows,
                fee_sets,
                cap_units,
                attribution=attribution,
                exit_signal=xsig,
            )
        )
    return out


def day_blocks(day_start_ns: int, hours: int = 6) -> list[tuple[str, int, int]]:
    hour = 3_600 * 10**9
    return [
        (f"B{k + 1}", day_start_ns + k * hours * hour, day_start_ns + (k + 1) * hours * hour)
        for k in range(24 // hours)
    ]


def run_grid(
    tasks: Iterable[GridExperiment],
    windows: Sequence[Window],
    fee_sets: Mapping[str, Mapping[str, Decimal]],
    *,
    cache_dirs: Sequence[str],
    workers: int = 8,
    cap_units: int = 1_000_000,
    chunk: int = 24,
    attribution: Sequence[tuple[str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate tasks in worker processes, grouped by signal recipe (deterministic output order)."""
    groups: dict[str, list[GridExperiment]] = defaultdict(list)
    for experiment in tasks:
        groups[experiment.signal_id].append(experiment)
    jobs: list[list[GridExperiment]] = []
    for sid in sorted(groups):
        items = groups[sid]
        jobs.extend(items[i : i + chunk] for i in range(0, len(items), chunk))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(list(cache_dirs),)
    ) as pool:
        results = list(
            pool.map(
                _run_group,
                jobs,
                [windows] * len(jobs),
                [fee_sets] * len(jobs),
                [cap_units] * len(jobs),
                [attribution] * len(jobs),
            )
        )
    return [row for group in results for row in group]
