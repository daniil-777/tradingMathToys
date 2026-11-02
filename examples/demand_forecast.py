"""Nonfinancial end-to-end example: delayed demand forecasting on the generic ticket and maturity machinery.

Synthetic hourly sensor demand (daily and weekly seasonality, intermittent zero periods, a level shift) is reported
three hours after each period ends. Three baseline adapters (seasonal naive, per-phase seasonal EWMA, ridge on causal
lags) issue an immutable forecast ticket at every cutoff, learn online only from labels that have matured, and are
scored exactly once per ticket over chronological folds. No market data, fees or BTC ledger are involved.

    python examples/demand_forecast.py                       # pooled metrics per adapter
    python examples/demand_forecast.py --json                # the full report as JSON
    python examples/demand_forecast.py --protocol p.json     # a strict-JSON DemandProtocol document

Exit codes: 0 success, 2 invalid configuration/data (typed FinanceAI error as one JSON line on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from financeai.core.errors import FinanceAIError
from financeai.domains.demand import DemandProtocol, DemandReport, default_protocol, evaluate


def run_example(
    *, weeks: int = 12, streams: int = 2, folds: int = 2, seed: int = 7, protocol_path: str | Path | None = None
) -> DemandReport:
    """Evaluate the default synthetic protocol (or a protocol file) and return the report."""
    if protocol_path is not None:
        protocol = DemandProtocol.from_json(Path(protocol_path).read_text())
    else:
        protocol = default_protocol(weeks=weeks, streams=streams, n_folds=folds, seed=seed)
    return evaluate(protocol)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_report(report: DemandReport) -> str:
    lines = [
        f"protocol {report.protocol_name}  id {report.protocol_id[:12]}  data {report.dataset_id[:12]}  "
        f"runs {len(report.runs)}",
        f"{'adapter':<16}{'scored/issued':>15}{'MAE':>9}{'RMSE':>9}{'bias':>9}{'pinball':>9}",
    ]
    for name, m in report.pooled().items():
        lines.append(
            f"{name:<16}{f'{m.n_scored}/{m.n_issued}':>15}{_fmt(m.mae):>9}{_fmt(m.rmse):>9}"
            f"{_fmt(m.bias):>9}{_fmt(m.pinball):>9}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Delayed demand forecasting example (synthetic data).")
    parser.add_argument("--weeks", type=int, default=12, help="weeks of hourly data (>= 5)")
    parser.add_argument("--streams", type=int, default=2, help="number of sensor streams")
    parser.add_argument("--folds", type=int, default=2, help="chronological test folds")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protocol", help="strict-JSON DemandProtocol file (overrides the synthetic defaults)")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)
    try:
        report = run_example(
            weeks=args.weeks, streams=args.streams, folds=args.folds, seed=args.seed, protocol_path=args.protocol
        )
    except FinanceAIError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report.to_mapping(), indent=2, sort_keys=True) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
