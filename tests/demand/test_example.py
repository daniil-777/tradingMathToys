"""The runnable nonfinancial demand example: callable from tests and as a script."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from financeai.domains.demand import default_protocol

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "demand_forecast.py"
ADAPTERS = ("seasonal_naive", "seasonal_ewma", "ridge_lags")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demand_forecast_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_prints_pooled_metrics_per_adapter(capsys: pytest.CaptureFixture[str]) -> None:
    assert _load().main(["--weeks", "6", "--streams", "1", "--folds", "1"]) == 0
    out = capsys.readouterr().out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.split()[:1] in [[a] for a in ADAPTERS]}
    assert set(rows) == set(ADAPTERS)
    assert rows["ridge_lags"][-1] == "n/a"  # point-only adapter: no pinball loss
    assert all(row[1] == "504/504" for row in rows.values())  # every issued ticket was scored once
    assert "MAE" in out and "pinball" in out


def test_example_report_is_strict_json_with_every_run(capsys: pytest.CaptureFixture[str]) -> None:
    module = _load()
    report = module.run_example(weeks=6, streams=2, folds=2)
    assert len(report.runs) == len(ADAPTERS) * 2 * 2
    assert module.main(["--weeks", "6", "--streams", "2", "--folds", "2", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc == json.loads(json.dumps(report.to_mapping()))
    assert set(doc["pooled"]) == set(ADAPTERS)
    assert all(m["n_unscored"] == 0 and m["mae"] > 0 for m in doc["pooled"].values())


def test_example_reads_a_protocol_file_and_reports_typed_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    text = default_protocol(weeks=6, streams=1, n_folds=1).to_json()
    good = tmp_path / "protocol.json"
    good.write_text(text)
    assert module.main(["--protocol", str(good)]) == 0
    capsys.readouterr()
    bad = tmp_path / "bad.json"
    bad.write_text(text.replace('"n_folds":1', '"n_folds":0'))
    assert module.main(["--protocol", str(bad)]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_config"
    assert module.main(["--weeks", "3"]) == 2  # too short: the first fit must see every hour-of-week phase
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_config"


def test_example_runs_as_a_script() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "FINANCEAI_DISABLE_JIT": "1"}
    done = subprocess.run(
        [sys.executable, str(EXAMPLE), "--weeks", "6", "--streams", "1", "--folds", "1"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert all(name in done.stdout for name in ADAPTERS)
