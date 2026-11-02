"""Configurable robustness filters and selection rules for grid-search results.

A configuration is *eligible* for one (latency, fee schedule) scenario when every block
account and the full-day account are valid, it completes at least ``min_round_trips``
round trips over the day, at least ``min_positive_blocks`` block accounts make money, and
at least ``min_positive_neighbour_fraction`` of its one-step grid neighbours are profitable
over the full day (a plateau, not an isolated spike). Eligible configurations are ranked by
full-day net settlement cash; with none eligible, nothing is selected (the policy stays in cash).

Amendment v5 (optional fields, defaults keep the v4 rule): ``min_net_ex_top1`` additionally
requires the full-day net *without its single best episode* to exceed that floor (reason
``concentrated``), and ``rank_by="net_ex_top1"`` ranks eligible configurations by that trimmed
net, so one lucky round trip cannot decide the selection.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Row = Mapping[str, Any]


@dataclass(frozen=True)
class SelectionRule:
    blocks: tuple[str, ...] = ("B1", "B2", "B3", "B4")
    full: str = "FULL"
    min_round_trips: int = 30
    min_positive_blocks: int = 3
    min_positive_neighbour_fraction: float = 0.5
    min_net_ex_top1: float | None = None
    rank_by: str = "full"  # "full" (v4) or "net_ex_top1" (v5)


@dataclass(frozen=True)
class Candidate:
    config_id: str
    full_net_pnl: float | None
    block_net_pnl: tuple[float | None, ...]
    round_trips: int
    positive_neighbour_fraction: float | None
    eligible: bool
    reasons: tuple[str, ...]
    full_row: Mapping[str, Any]
    net_ex_top1_pnl: float | None = None


def _neighbours(point: Mapping[str, Any], axes: Mapping[str, Sequence[Any]]) -> list[tuple[tuple[str, Any], ...]]:
    out = []
    for name, values in axes.items():
        idx = list(values).index(point[name])
        for step in (-1, 1):
            j = idx + step
            if 0 <= j < len(values):
                moved = dict(point)
                moved[name] = values[j]
                out.append(tuple(sorted(moved.items())))
    return out


def select_configs(
    rows: Iterable[Row],
    *,
    latency_ms: int,
    fees: str,
    axes: Mapping[str, Sequence[Any]],
    key: Callable[[Row], Mapping[str, Any]],
    rule: SelectionRule,
    group: Callable[[Row], Any] | None = None,
) -> tuple[Candidate | None, list[Candidate]]:
    """Apply the rule; ``key`` maps a row to its steppable axis values, ``group`` (optional)
    to categorical dimensions that neighbours must share (e.g. leader/follower pair)."""
    by_config: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in rows:
        if row["latency_ms"] == latency_ms and row["fees"] == fees:
            by_config[row["config_id"]][row["window"]] = row
    point_of: dict[str, tuple[Any, tuple[tuple[str, Any], ...]]] = {}
    full_net_at: dict[tuple[Any, tuple[tuple[str, Any], ...]], float | None] = {}
    for cid, windows in by_config.items():
        full = windows.get(rule.full)
        if full is None:
            continue
        point = tuple(sorted(key(full).items()))
        gkey = group(full) if group else None
        point_of[cid] = (gkey, point)
        full_net_at[(gkey, point)] = full["net_pnl"] if full["valid"] else None

    table: list[Candidate] = []
    for cid, windows in by_config.items():
        full = windows.get(rule.full)
        if full is None:
            continue
        reasons: list[str] = []
        accounts = [windows.get(b) for b in rule.blocks] + [full]
        if any(a is None or not a["valid"] for a in accounts):
            reasons.append("invalid_account")
        block_nets = tuple(
            (w["net_pnl"] if w is not None and w["valid"] else None) for w in (windows.get(b) for b in rule.blocks)
        )
        if sum(1 for v in block_nets if v is not None and v > 0) < rule.min_positive_blocks:
            reasons.append("positive_blocks")
        trips = int(full["round_trips"])
        if trips < rule.min_round_trips:
            reasons.append("round_trips")
        full_net = full["net_pnl"] if full["valid"] else None
        if full_net is None or full_net <= 0:
            reasons.append("full_day_not_positive")
        gkey, point = point_of[cid]
        # Neighbours missing from the results (or invalid) count as not profitable.
        neigh = [full_net_at.get((gkey, p)) for p in _neighbours(dict(point), axes)]
        frac = (sum(1 for v in neigh if v is not None and v > 0) / len(neigh)) if neigh else None
        if frac is None or frac < rule.min_positive_neighbour_fraction:
            reasons.append("neighbour_plateau")
        ex_top1 = full.get("net_ex_top1_pnl") if full["valid"] else None
        if rule.min_net_ex_top1 is not None and (ex_top1 is None or ex_top1 <= rule.min_net_ex_top1):
            reasons.append("concentrated")
        table.append(Candidate(cid, full_net, block_nets, trips, frac, not reasons, tuple(reasons), full, ex_top1))
    if rule.rank_by == "net_ex_top1":
        table.sort(
            key=lambda c: (c.net_ex_top1_pnl is not None, c.net_ex_top1_pnl or 0.0, c.full_net_pnl or 0.0), reverse=True
        )
    elif rule.rank_by == "full":
        table.sort(key=lambda c: (c.full_net_pnl is not None, c.full_net_pnl or 0.0), reverse=True)
    else:
        raise ValueError(f"unknown rank_by {rule.rank_by!r}")
    eligible = [c for c in table if c.eligible]
    return (eligible[0] if eligible else None), table
