"""Interactive report semantics: complete ledgers, exact clocks, currency isolation, safe export."""

from __future__ import annotations

import dataclasses
import importlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError, InvalidDataError, OptionalDependencyError
from financeai.core.hashing import sha256_file, sha256_json
from financeai.domains.market import InstrumentLatency, RiskLimits
from financeai.reporting import ReplayReportConfig, build_replay_report_data, write_replay_report
from financeai.reporting.replay import _series

from ..market.test_heterogeneous_markets import MS, compare, portfolio
from .qa_fixture import QA_LIMITS, QA_SCENARIO, qa_market, qa_result


@pytest.fixture
def mixed():
    data = portfolio()
    return compare(data, RiskLimits(100), {data.instruments[0].key: InstrumentLatency(MS, 9 * MS)})


def test_mixed_currencies_are_separate_and_full_record_statistics_are_unchanged_by_reduction(mixed) -> None:
    result, prep, scenario = mixed
    sparse = build_replay_report_data(result, prep, scenario=scenario, config=ReplayReportConfig(display_buckets=1))
    dense = build_replay_report_data(result, prep, scenario=scenario)
    assert {a["currency"] for a in sparse["accounts"]} == {"CHF", "COIN"}
    for a, b in zip(sparse["accounts"], dense["accounts"], strict=True):
        indices = [v for v, i in enumerate(prep.data.instruments) if i.settlement_currency == a["currency"]]
        assert a["net_pnl"] == sum(result.settlement.cash[v] for v in indices)
        for key in ("net_pnl", "commissions", "max_drawdown", "fills", "unmarked_points"):
            assert a[key] == b[key]
    assert sparse["counts"]["displayed_events"] == len(result.fills)
    assert sparse["source_sha256"] == sha256_json(sparse["sources"])


def test_actual_and_acknowledged_positions_use_different_exact_clocks(mixed) -> None:
    result, prep, scenario = mixed
    data = build_replay_report_data(result, prep, scenario=scenario)
    asset = data["assets"][0]
    assert asset["series"]["position"]["ns"] != asset["series"]["known_position"]["ns"]
    assert asset["events"][0]["ack_latency_ns"] == str(9 * MS)
    assert all(isinstance(t, str) for t in asset["series"]["position"]["ns"])
    assert asset["instrument"] == prep.data.instruments[0].to_mapping()


def test_instrument_selection_filters_execution_panels_but_preserves_full_currency_account(mixed) -> None:
    result, prep, scenario = mixed
    all_data = build_replay_report_data(result, prep, scenario=scenario)
    key = prep.data.instruments[0].key
    selected = build_replay_report_data(result, prep, scenario=scenario, config=ReplayReportConfig(instruments=(key,)))
    assert [a["key"] for a in selected["assets"]] == [key]
    assert selected["accounts"] == all_data["accounts"]


def test_invalid_accounts_and_missing_valuations_stay_visible() -> None:
    data = build_replay_report_data(
        qa_result(), qa_market(), scenario=QA_SCENARIO, limits=QA_LIMITS, config=ReplayReportConfig(display_buckets=1)
    )
    assert not data["account_valid"] and data["invalid_reason"] == "cap_exceeded"
    assert all(a["net_pnl"] is None for a in data["accounts"])
    assert any(v is None for a in data["accounts"] for v in a["series"]["marked"]["y"])
    assert data["accounts"][0]["unmarked_points"] > 0


def test_unknown_limits_are_not_invented_and_stricter_audit_caps_invalidate_the_report(mixed) -> None:
    result, prep, scenario = mixed
    unknown = build_replay_report_data(dataclasses.replace(result, limits=None), prep, scenario=scenario)
    assert unknown["risk_limits"] is None
    assert all(a["cap_units"] is None for a in unknown["assets"])
    strict = build_replay_report_data(result, prep, scenario=scenario, limits=RiskLimits(1))
    assert not strict["account_valid"] and all(a["net_pnl"] is None for a in strict["accounts"])


def test_large_epoch_nanoseconds_and_integer_positions_have_exact_hover_values() -> None:
    start = 1_700_000_000_000_000_001
    value = 2**53 + 1
    series = _series(np.array([start + 1, start + 2], np.int64), np.array([value, 0], np.int64), start, start + 3, 10)
    assert str(start + 1) in series["ns"] and str(value) in series["exact"]
    assert series["x"][1] == 1e-9
    assert series["utc"][1] == "2023-11-14T22:13:20.000000002Z"


def test_extrema_and_gaps_survive_the_smallest_display_budget() -> None:
    s = _series(np.arange(7, dtype=np.int64), np.array([0.0, 9.0, -8.0, np.nan, 2.0, 3.0, 0.0]), 0, 7, 1)
    assert 9.0 in s["y"] and -8.0 in s["y"] and None in s["y"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"display_buckets": 0},
        {"max_events": True},
        {"max_points": 1},
        {"theme": "unknown"},
        {"evidence_scope": "PROFITABLE"},
        {"instruments": ()},
        {"instruments": ("x", "x")},
        {"title": " "},
    ],
)
def test_bad_display_settings_are_rejected(kwargs) -> None:
    with pytest.raises(InvalidConfigError):
        ReplayReportConfig(**kwargs)


def test_display_limits_fail_instead_of_silently_dropping_trades(mixed) -> None:
    result, prep, _ = mixed
    with pytest.raises(InvalidConfigError, match="max_events"):
        build_replay_report_data(result, prep, config=ReplayReportConfig(max_events=1))
    with pytest.raises(InvalidConfigError, match=r"max_points|display limits"):
        build_replay_report_data(result, prep, config=ReplayReportConfig(max_points=20))
    with pytest.raises(InvalidConfigError, match="unknown"):
        build_replay_report_data(result, prep, config=ReplayReportConfig(instruments=("typo/ASSET",)))


def test_corrupt_settlement_is_not_presented_as_a_valid_chart(mixed) -> None:
    result, prep, _ = mixed
    corrupt = dataclasses.replace(result, settlement=dataclasses.replace(result.settlement, cash=(123.0, 0.0, 0.0)))
    with pytest.raises(InvalidDataError, match="reconcile"):
        build_replay_report_data(corrupt, prep)


def test_missing_plotly_is_an_optional_dependency_error(mixed, tmp_path: Path, monkeypatch) -> None:
    real = importlib.import_module

    def blocked(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("intentionally unavailable")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(OptionalDependencyError, match="interactive"):
        write_replay_report(mixed[0], mixed[1], tmp_path / "report.html")
    assert not list(tmp_path.iterdir())


def test_html_is_offline_deterministic_escaped_and_hash_verified(mixed, tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    result, prep, scenario = mixed
    config = ReplayReportConfig(title="Example </script><script>window.BAD=1</script>")
    a = write_replay_report(result, prep, tmp_path / "one.html", scenario=scenario, config=config)
    b = write_replay_report(result, prep, tmp_path / "two.html", scenario=scenario, config=config)
    assert a.sha256 == b.sha256 == sha256_file(a.html_path)
    text = a.html_path.read_text()
    assert "<script src=" not in text and "window.BAD=1</script>" not in text
    assert "\\u003c/script\\u003e" in text
    manifest = json.loads(a.manifest_path.read_text())
    assert manifest["output"]["sha256"] == a.sha256
    assert manifest["renderer"]["plotly"]
    with pytest.raises(InvalidConfigError, match="already exists"):
        write_replay_report(result, prep, a.html_path)
    with pytest.raises(InvalidConfigError, match="html"):
        write_replay_report(result, prep, tmp_path / "bad.svg")


def test_unfilled_and_partial_order_metadata_is_not_presented_as_additional_fills(mixed) -> None:
    result, prep, scenario = mixed
    orders = result.orders
    # Keep the executed ledger intact, add an unfilled submission, and mark an existing
    # order as partial by increasing only its requested quantity.
    columns = {
        f.name: np.append(getattr(orders, f.name), getattr(orders, f.name)[0]) for f in dataclasses.fields(orders)
    }
    columns["order_id"][-1] = int(orders.order_id.max()) + 1
    columns["filled_units"][-1] = 0
    columns["qty_units"][0] += 1
    data = build_replay_report_data(
        dataclasses.replace(result, orders=type(orders)(**columns)), prep, scenario=scenario
    )
    events = data["assets"][int(orders.venue[0])]["events"]
    unfilled = next(e for e in events if e["kind"] == "unfilled")
    assert unfilled["qty_units"] == "0" and unfilled["requested_units"] == str(orders.qty_units[0])
    assert unfilled["order_status"] == "unfilled"
    partial = next(e for e in events if e["kind"] == "fill")
    assert partial["order_status"] == "partial"
    assert int(partial["requested_units"]) > int(partial["order_filled_units"])
    assert partial["price_ticks"] == str(result.fills.px_ticks[0])
    assert partial["price_exact"] == format(Decimal(partial["price_ticks"]) * prep.data.instruments[0].tick_size, "f")
