"""Compare a hindsight schedule with execution replay on synthetic or cached inverse markets.

This benchmark sees future prices. It is not a causal trading strategy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from financeai.domains.market.cache import load_tapes
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.ledger import exact_score
from financeai.domains.market.scenario import RiskLimits, SessionPlan, demo_scenarios
from financeai.domains.market.stream import MarketData
from financeai.domains.market.synthetic import SyntheticMarketSpec, synthetic_market
from financeai.research.adaptive_hindsight import adaptive_depth_oracle
from financeai.research.hindsight import replay_depth_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", help="inverse-market cache; repeat per instrument")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap-units", type=int, default=10_000)
    parser.add_argument("--step-units", type=int, default=1_000)
    args = parser.parse_args()
    data = (
        MarketData(tuple(load_tapes(path) for path in args.cache))
        if args.cache
        else synthetic_market(SyntheticMarketSpec(seed=7, rows_per_venue=500))
    )
    prep = PreparedMarket.build(data)
    scenario = demo_scenarios(data.instruments)["base"]
    roundtrip = scenario.order_latency_ns + scenario.ack_ns
    session = SessionPlan.for_window(
        int(prep.stream.recv_ns[0]),
        int(prep.stream.recv_ns[-1]),
        warmup_ns=0,
        entry_stop_ns=6 * roundtrip,
        flatten_ns=4 * roundtrip,
        exec_stop_ns=2 * roundtrip,
    )
    limits = RiskLimits(args.cap_units)
    plans = tuple(
        adaptive_depth_oracle(
            tape, scenario, session, limits, step_units=args.step_units, candidate_spacing_ns=50_000_000
        )
        for tape in data.venues
    )
    replay = replay_depth_oracle(data, plans, scenario, session, limits, full_stream=True)
    exact = exact_score(replay.fills, data.instruments, [scenario.fee_for(i) for i in data.instruments])
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("orders", "fills"):
        table = getattr(replay, name)
        np.savez_compressed(
            args.out / f"{name}.npz", **{field.name: getattr(table, field.name) for field in dataclasses.fields(table)}
        )
    summary = {
        "uses_future_data": True,
        "valid": replay.settlement.valid,
        "pnl_by_currency": replay.settlement.pnl_by_currency,
        "exact_cash": str(exact),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
