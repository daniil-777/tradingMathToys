"""Run a complete synthetic market replay without downloads or credentials.

    python examples/market_replay.py --out runs/quickstart

Writes two reusable tape caches and a replay's configuration, summary, orders and fills.
The random-walk books demonstrate the workflow; they provide no evidence of trading alpha.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from financeai.application.strategy import StrategyConfig, load_market, run_strategy, write_run_outputs
from financeai.core.errors import FinanceAIError
from financeai.domains.market.cache import save_tapes
from financeai.domains.market.instruments import INSTRUMENTS, ContractType
from financeai.domains.market.scenario import demo_scenarios
from financeai.domains.market.synthetic import GENERATOR, SyntheticMarketSpec, synthetic_source, synthetic_tapes

STRATEGY = Path(__file__).with_name("strategies") / "lead_gap.json"


def run_example(out: Path, *, seed: int = 7, market: str = "linear") -> dict[str, Any]:
    """Create a new output directory and replay the bundled strategy on synthetic books."""
    cfg = StrategyConfig.load(STRATEGY)
    kind = ContractType(market)
    instruments = tuple(
        INSTRUMENTS[key]
        if kind == ContractType.INVERSE
        else dataclasses.replace(
            INSTRUMENTS[key],
            contract_type=kind,
            tick_size=Decimal("0.01"),
            lot_size=1,
            settlement_currency=None,
            shortable=kind != ContractType.SPOT,
        )
        for key in cfg.instruments
    )
    # The generator sizes depth in lots, so match the teaching policy to those units.
    config = cfg.to_mapping()
    config["signal"]["depth_units"] = 10 * instruments[0].lot_size
    sizes = config["policy"]["size_units"]
    config["policy"]["size_units"] = {i.key: 10 * i.lot_size if sizes[i.key] else 0 for i in instruments}
    config["policy"]["allow_short"] = kind != ContractType.SPOT
    cfg = StrategyConfig.from_mapping(config)
    spec = SyntheticMarketSpec(seed=seed, venues=cfg.instruments, instruments=instruments)
    out.mkdir(parents=True, exist_ok=False)
    caches = []
    for key, tapes in synthetic_tapes(spec).items():
        target = out / "cache" / key.replace("/", "_")
        save_tapes(tapes, target, source=synthetic_source(spec, key), parser=GENERATOR)
        caches.append(target)
    prep = load_market(caches, cfg.instruments)
    scenario = demo_scenarios(prep.data.instruments)["base"]
    result = run_strategy(prep, cfg, scenario)
    return write_run_outputs(result, prep, cfg, scenario, out / "base")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/quickstart"), help="new directory for generated artifacts"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--market", choices=[k.value for k in ContractType], default="linear")
    args = parser.parse_args(argv)
    try:
        summary = run_example(args.out, seed=args.seed, market=args.market)
    except FinanceAIError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps({"error": "io_error", "message": str(exc)}), file=sys.stderr)
        return 2
    fields = ("scenario", "valid", "net_pnl", "settlement_currencies", "fills", "final_position_units")
    print(json.dumps({"out": str(args.out), **{key: summary[key] for key in fields}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
