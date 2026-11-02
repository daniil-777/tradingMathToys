"""Command-line interface: ``financeai ingest | run | suite``.

Exit codes: 0 success, 2 invalid configuration/data or file access errors (reported as
one JSON line on stderr). Unexpected exceptions propagate with a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import financeai
from financeai.core.errors import FinanceAIError, InvalidConfigError


def _json(path: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise InvalidConfigError("JSON configuration must be an object")
        return value
    except json.JSONDecodeError as exc:
        raise InvalidConfigError("invalid JSON configuration", context={"file": path}) from exc


def _cmd_ingest(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from financeai.core.hashing import sha256_json
    from financeai.domains.market.cache import SourceInfo, save_tapes
    from financeai.domains.market.instruments import INSTRUMENTS, InstrumentSpec
    from financeai.domains.market.recorder import read_recorder_file

    if args.instrument_spec:
        instrument = InstrumentSpec.from_mapping(_json(args.instrument_spec))
    elif args.instrument not in INSTRUMENTS:
        raise InvalidConfigError("unknown instrument", context={"known": sorted(INSTRUMENTS)})
    else:
        instrument = INSTRUMENTS[args.instrument]
    if args.format == "tardis":
        from financeai.domains.market.tardis import read_tardis_snapshot_trades

        if not args.trades:
            raise InvalidConfigError("Tardis snapshots require --trades CSV")
        tapes = read_tardis_snapshot_trades(args.raw, args.trades, instrument)
        book, trades = SourceInfo.from_file(args.raw), SourceInfo.from_file(args.trades)
        source = SourceInfo(
            name=f"{book.name}+{trades.name}",
            sha256=sha256_json({"book": asdict(book), "trades": asdict(trades)}),
            bytes=book.bytes + trades.bytes,
        )
        parser = {"name": "tardis-normalized-csv", "version": "1.0"}
    else:
        if args.trades:
            raise InvalidConfigError("--trades is only valid for --format tardis")
        tapes = read_recorder_file(args.raw, instrument)
        source = SourceInfo.from_file(args.raw)
        parser = None
    manifest = save_tapes(tapes, args.out, source=source, parser=parser)
    print(json.dumps({"out": str(args.out), "report": manifest["report"]}))
    return 0


def _run_one(
    cfg_path: str, caches: Sequence[str], scenario_name: str, out: Path, scenario_file: str | None = None
) -> dict[str, object]:
    from financeai.application.strategy import StrategyConfig, load_market, run_strategy, write_run_outputs
    from financeai.domains.market.scenario import Scenario, demo_scenarios

    cfg = StrategyConfig.load(cfg_path)
    prep = load_market(caches, cfg.instruments)
    scenarios = demo_scenarios(prep.data.instruments)
    if scenario_file:
        scenario = Scenario.from_mapping(_json(scenario_file))
    elif scenario_name not in scenarios:
        raise InvalidConfigError("unknown scenario", context={"known": sorted(scenarios)})
    else:
        scenario = scenarios[scenario_name]
    result = run_strategy(prep, cfg, scenario)
    return write_run_outputs(result, prep, cfg, scenario, out)


def _cmd_run(args: argparse.Namespace) -> int:
    summary = _run_one(args.config, args.cache, args.scenario, Path(args.out), args.scenario_file)
    print(json.dumps({k: summary[k] for k in ("scenario", "valid", "net_pnl", "round_trips")}))
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    from financeai.application.strategy import StrategyConfig, load_market, run_strategy, write_run_outputs
    from financeai.domains.market.scenario import demo_scenarios

    cfg = StrategyConfig.load(args.config)
    prep = load_market(args.cache, cfg.instruments)  # one prepared market for all scenarios
    out = Path(args.out)
    results = []
    for name, scenario in demo_scenarios(prep.data.instruments).items():
        result = run_strategy(prep, cfg, scenario)
        results.append(write_run_outputs(result, prep, cfg, scenario, out / name))
    out.mkdir(parents=True, exist_ok=True)
    (out / "suite.json").write_text(json.dumps({"strategy": cfg.name, "results": results}, indent=2) + "\n")
    print(json.dumps([{k: r[k] for k in ("scenario", "valid", "net_pnl", "round_trips")} for r in results]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="financeai", description=__doc__)
    parser.add_argument("--version", action="version", version=f"financeai {financeai.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="parse a recorder .txt[.gz] file into an immutable tape cache")
    p.add_argument("raw")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--instrument", help="built-in fictitious instrument key")
    group.add_argument("--instrument-spec", help="JSON InstrumentSpec for your market")
    p.add_argument("--format", choices=("recorder", "tardis"), default="recorder")
    p.add_argument("--trades", help="normalized Tardis trades CSV accompanying snapshots")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_ingest)
    for name, func, helptext in (
        ("run", _cmd_run, "replay one strategy under one scenario"),
        ("suite", _cmd_suite, "replay one strategy under all four example scenarios"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("config")
        p.add_argument("--cache", action="append", required=True, help="tape cache directory (repeat per venue)")
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--scenario", default="base")
            p.add_argument("--scenario-file", help="JSON execution costs and timing, overriding the demo scenario")
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FinanceAIError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps({"error": "io_error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
