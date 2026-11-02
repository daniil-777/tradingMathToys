"""Create an offline interactive report on invented spot, linear, and inverse instruments.

    python examples/replay_report.py --out runs/replay.html

Install .[interactive] first. Open the resulting HTML in a browser; no server is required.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

from financeai.application.strategy import StrategyConfig, run_strategy
from financeai.domains.market import ContractType, InstrumentLatency, InstrumentSpec, MarketData, RiskLimits
from financeai.domains.market.engine import PreparedMarket, ReplayResult, ThresholdParams
from financeai.domains.market.scenario import Scenario
from financeai.domains.market.synthetic import SyntheticMarketSpec, synthetic_tapes
from financeai.reporting import ReplayReportConfig, write_replay_report


def example_replay() -> tuple[ReplayResult, PreparedMarket, Scenario]:
    instruments = (
        InstrumentSpec("demo", "SHARE-USD", Decimal("0.01"), contract_type=ContractType.SPOT, base_currency="SHARE"),
        InstrumentSpec("demo", "INDEX-USD", Decimal("0.01"), contract_multiplier=Decimal("5")),
        InstrumentSpec(
            "demo", "COIN-USD", Decimal("0.5"), lot_size=10, contract_type=ContractType.INVERSE, base_currency="COIN"
        ),
    )
    keys = tuple(i.key for i in instruments)
    tapes = synthetic_tapes(
        SyntheticMarketSpec(
            seed=7, venues=keys, instruments=instruments, rows_per_venue=800, start_ns=1_735_689_600_000_000_001
        )
    )
    market = PreparedMarket.build(MarketData(tuple(tapes[key] for key in keys)))
    scenario = Scenario(
        "Teaching costs",
        {key: Decimal("0.0001") for key in keys},
        2_000_000,
        latency_by_instrument={keys[0]: InstrumentLatency(2_000_000, 7_000_000)},
    )
    config = StrategyConfig.from_threshold_params(
        name="Synthetic lead-gap illustration",
        instruments=keys,
        signal={
            "kind": "lead_gap",
            "leader": keys[1],
            "follower": keys[0],
            "window_ms": 20,
            "depth_units": 10,
            "gate_ticks": 2,
        },
        params=ThresholdParams(
            (1, 0, 1), (10, 0, 100), 0.5, -1e9, max_hold_ns=50_000_000, entry_slip_ticks=1, exit_slip_ticks=20
        ),
        session={"warmup_s": 0, "entry_stop_s": 0.4, "flatten_s": 0.3, "exec_stop_s": 0.06},
    )
    limits = RiskLimits(1000, per_instrument={keys[0]: 10, keys[2]: 100})
    return run_strategy(market, config, scenario, limits), market, scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/replay.html"))
    parser.add_argument("--theme", choices=("light", "dark"), default="dark")
    args = parser.parse_args()
    result, market, scenario = example_replay()
    artifact = write_replay_report(
        result,
        market,
        args.out,
        scenario=scenario,
        config=ReplayReportConfig(title="From market observation to trading result", theme=args.theme),
    )
    print(artifact.html_path)


if __name__ == "__main__":
    main()
