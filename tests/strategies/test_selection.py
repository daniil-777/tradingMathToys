"""Registered selection rule: eligibility, neighbour plateau and ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from financeai.application.selection import SelectionRule, select_configs

AXES: Mapping[str, Sequence[Any]] = {"enter": [1.0, 2.0, 3.0], "hold": [10, 30]}


def rows_for(
    cid: str,
    enter: float,
    hold: int,
    blocks: list[float | None],
    full: float | None,
    trips: int = 50,
    valid: bool = True,
) -> list[dict[str, Any]]:
    params = {"enter": enter, "max_hold_ns": hold}
    out = [
        {
            "config_id": cid,
            "window": f"B{k + 1}",
            "net_pnl": v,
            "valid": v is not None and valid,
            "round_trips": trips // 4,
            "params": params,
            "latency_ms": 5,
            "fees": "base",
        }
        for k, v in enumerate(blocks)
    ]
    out.append(
        {
            "config_id": cid,
            "window": "FULL",
            "net_pnl": full,
            "valid": full is not None and valid,
            "round_trips": trips,
            "params": params,
            "latency_ms": 5,
            "fees": "base",
        }
    )
    return out


def key(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"enter": row["params"]["enter"], "hold": row["params"]["max_hold_ns"]}


def grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += rows_for("a", 1.0, 10, [1, 1, 1, -1], 2.0)
    rows += rows_for("b", 2.0, 10, [1, 1, 1, 1], 5.0)  # best, neighbours a, c, e
    rows += rows_for("c", 3.0, 10, [1, -1, -1, 1], 0.5)  # only 2 positive blocks
    rows += rows_for("d", 1.0, 30, [1, 1, 1, 1], 9.0, trips=10)  # too few round trips
    rows += rows_for("e", 2.0, 30, [1, 1, 1, None], 3.0)  # invalid block account
    rows += rows_for("f", 3.0, 30, [-1, -1, -1, -1], -4.0)
    return rows


def test_selects_the_best_eligible_configuration() -> None:
    chosen, table = select_configs(grid(), latency_ms=5, fees="base", axes=AXES, key=key, rule=SelectionRule())
    assert chosen is not None and chosen.config_id == "b"
    by_id = {c.config_id: c for c in table}
    assert not by_id["c"].eligible and "positive_blocks" in by_id["c"].reasons
    assert not by_id["d"].eligible and "round_trips" in by_id["d"].reasons
    assert not by_id["e"].eligible and "invalid_account" in by_id["e"].reasons
    # b's one-step neighbours are a (+2), c (+0.5) and e (+3): all positive
    assert by_id["b"].positive_neighbour_fraction == 1.0
    # f's neighbours are e (+3) and c (+0.5): positive, but f itself loses everywhere
    assert not by_id["f"].eligible


def test_no_eligible_configuration_selects_nothing() -> None:
    rows = rows_for("x", 1.0, 10, [-1, -1, 1, 1], 0.1)
    chosen, _ = select_configs(rows, latency_ms=5, fees="base", axes=AXES, key=key, rule=SelectionRule())
    assert chosen is None


def with_trim(rows: list[dict[str, Any]], ex_top1: dict[str, float]) -> list[dict[str, Any]]:
    for r in rows:
        if r["window"] == "FULL":
            r["net_ex_top1_pnl"] = ex_top1[r["config_id"]]
    return rows


def test_trimmed_rule_ranks_by_net_without_the_best_episode() -> None:
    rows = rows_for("p", 1.0, 10, [1, 1, 1, 9], 12.0) + rows_for("q", 2.0, 10, [1, 1, 1, 2], 5.0)
    rows += rows_for("r", 3.0, 10, [1, 1, 1, 1], 4.0)
    rows = with_trim(rows, {"p": 1.0, "q": 3.0, "r": -0.5})
    rule = SelectionRule(rank_by="net_ex_top1", min_net_ex_top1=0.0)
    chosen, table = select_configs(rows, latency_ms=5, fees="base", axes=AXES, key=key, rule=rule)
    assert chosen is not None and chosen.config_id == "q"  # p has the best day but one episode carries it
    by_id = {c.config_id: c for c in table}
    assert not by_id["r"].eligible and "concentrated" in by_id["r"].reasons
    assert [c.config_id for c in table][:2] == ["q", "p"]
    assert by_id["p"].net_ex_top1_pnl == 1.0
    # the registered v4 rule (default) still ranks by the full-day net
    chosen_v4, _ = select_configs(rows, latency_ms=5, fees="base", axes=AXES, key=key, rule=SelectionRule())
    assert chosen_v4 is not None and chosen_v4.config_id == "p"
