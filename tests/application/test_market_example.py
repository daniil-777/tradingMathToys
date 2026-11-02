"""The documented quickstart produces real ledger activity for every market convention."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("market", ["spot", "linear", "inverse"])
def test_market_quickstart_trades_and_preserves_its_configuration(tmp_path: Path, market: str) -> None:
    path = Path(__file__).resolve().parents[2] / "examples" / "market_replay.py"
    spec = importlib.util.spec_from_file_location("market_replay_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / market
    summary = module.run_example(out, market=market)
    assert summary["valid"]
    assert summary["fills"] > 0
    assert all(position == 0 for position in summary["final_position_units"])
    recorded = json.loads((out / "base" / "summary.json").read_text())
    assert all(instrument["contract_type"] == market for instrument in recorded["instrument_specs"])
    assert recorded["scenario_config"]["name"] == "base"
    for filename in ("strategy.json", "orders.csv", "fills.csv"):
        assert (out / "base" / filename).stat().st_size > 0
