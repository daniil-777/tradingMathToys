"""Position/PnL figures: numbers come from the full ledger records, the manifest pins inputs,
formulas and outputs by hash, and display reduction never hides a cap excursion or an
invalid ending."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from financeai.core.clock import NS_PER_MS, NS_PER_S
from financeai.core.errors import InvalidConfigError, OptionalDependencyError
from financeai.core.hashing import sha256_file
from financeai.core.serialization import loads_strict
from financeai.domains.market import metrics
from financeai.domains.market.engine import (
    PreparedMarket,
    ReplayResult,
    ThresholdParams,
    run_replay,
)
from financeai.domains.market.ledger import Fills
from financeai.domains.market.scenario import DEMO_BASE_FEES, RiskLimits, Scenario, SessionPlan
from financeai.reporting import FigureArtifact, charts, position_pnl_figure

from ..market.synthetic import random_market
from .qa_fixture import (
    BAD_BOOK_NS,
    EXCURSION_NS,
    QA_DPI,
    QA_LIMITS,
    QA_SCENARIO,
    QA_SESSION,
    qa_market,
    qa_result,
)

MS = 1_000_000
BASE = Scenario("base", DEMO_BASE_FEES, 5 * MS)
LIMITS = RiskLimits(5_000)


def replay(seed: int = 4) -> tuple[ReplayResult, PreparedMarket, SessionPlan]:
    prep = PreparedMarket.build(random_market(seed, n_rows=400))
    stream = prep.stream
    signal = np.random.default_rng(seed + 1).normal(0.0, 1.0, size=len(stream))
    first, last = int(stream.recv_ns[0]), int(stream.recv_ns[-1])
    session = SessionPlan(first + 20 * MS, last - 400 * MS, last - 300 * MS, last - 60 * MS, last)
    params = ThresholdParams(
        direction=(1, -1),
        size_units=(2_000, 1_000),
        enter=0.7,
        exit=0.0,
        max_hold_ns=150 * MS,
        entry_slip_ticks=2,
        exit_slip_ticks=6,
    )
    return run_replay(prep, signal, params, BASE, session, LIMITS), prep, session


def render(out: Path, **kwargs: Any) -> Any:
    res, prep, session = replay()
    options: dict[str, Any] = {
        "title": "Synthetic replay",
        "scenario_name": "base",
        "scenario": BASE,
        "session": session,
        "limits": LIMITS,
        "dpi": 80,
    }
    options.update(kwargs)
    return position_pnl_figure(res, prep, out_path=out, **options)


def series_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in manifest["series_definitions"]}


def test_figure_numbers_reconcile_with_the_settlement(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    res, prep, _ = replay()
    art = render(tmp_path / "pnl.png")
    assert art.image_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    manifest = loads_strict(art.manifest_path.read_text())
    assert manifest == art.manifest
    assert art.manifest_path.name == "pnl.png.manifest.json"
    st_ = res.settlement
    assert st_.valid and st_.net_pnl is not None
    assert manifest["metrics"]["net_pnl"] == st_.net_pnl
    assert manifest["reconciliation_record"]["reconciled"] is True
    series = series_by_id(manifest)
    assert series["cash.total"]["final_value"] == pytest.approx(st_.net_pnl, rel=1e-12, abs=1e-15)
    assert series["mark"]["final_value"] == pytest.approx(st_.net_pnl, rel=1e-12, abs=1e-15)
    for v, spec in enumerate(prep.data.instruments):
        pos = series[f"position.{spec.key}"]
        assert pos["final_value"] == 0
        assert pos["full_max_abs"] == st_.max_abs_position_units[v]
        assert series[f"cash.{spec.key}"]["final_value"] == pytest.approx(st_.cash[v], rel=1e-12, abs=1e-15)
    assert manifest["terminal_validity"]["valid"] is True
    assert manifest["terminal_validity"]["net_pnl"] == st_.net_pnl


def test_manifest_pins_inputs_formulas_and_output(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    res, _, session = replay()
    art = render(tmp_path / "pnl.png", run_id="run-7", evidence_scope="SYNTHETIC")
    m = art.manifest
    assert m["schema_version"] == 1 and m["kind"] == "position_pnl"
    assert m["run_id"] == "run-7" and m["scenario"] == "base"
    assert m["timing_mode"] == "EXCHANGE_CLOCK" and m["evidence_scope"] == "SYNTHETIC"
    assert m["metrics_source"] == "FULL_ORDERED_RECORDS" and m["timezone"] == "UTC"
    sources = {s["name"]: s for s in m["source_artifacts"]}
    for f in fields(Fills):
        arr = np.ascontiguousarray(getattr(res.fills, f.name))
        record = sources[f"fills.{f.name}"]
        assert record["sha256"] == hashlib.sha256(arr.tobytes()).hexdigest()
        assert record["dtype"] == arr.dtype.str and record["shape"] == [len(res.fills)]
    assert m["output_sha256"] == sha256_file(art.image_path) == art.sha256
    assert m["output"]["file"] == "pnl.png" and m["output"]["format"] == "png"
    assert m["full_interval"]["start_ns"] <= session.start_ns
    assert m["full_interval"]["end_ns"] >= session.end_ns
    assert m["session"]["exec_cutoff_ns"] == session.exec_cutoff_ns
    assert m["display_reduction"]["method"] == "extrema_envelope"
    assert m["display_reduction"]["metrics_computed_before_reduction"] is True
    assert "x_seconds" in m["transformation"]
    for s in m["series_definitions"]:
        assert s["unit"] and s["formula"] and s["time_basis"]
    for key in ("renderer_source_sha256", "environment_sha256", "output_sha256"):
        assert len(m[key]) == 64 and int(m[key], 16) >= 0


def render_qa(out: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ReplayResult, PreparedMarket, FigureArtifact, Any]:
    """Render the QA fixture; also return the Matplotlib figure that was saved (a spy on
    ``Figure.savefig``), so the checks read the points actually drawn, not a summary of them."""
    figure_mod = pytest.importorskip("matplotlib.figure")
    saved: list[Any] = []
    savefig = figure_mod.Figure.savefig

    def spy(fig: Any, *args: Any, **kwargs: Any) -> Any:
        saved.append(fig)
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(figure_mod.Figure, "savefig", spy)
    result, prep = qa_result(), qa_market()
    art = position_pnl_figure(
        result,
        prep,
        title="QA fixture",
        scenario_name="qa_base",
        scenario=QA_SCENARIO,
        session=QA_SESSION,
        limits=QA_LIMITS,
        out_path=out,
        dpi=QA_DPI,
    )
    assert len(saved) == 1
    return result, prep, art, saved[0]


C_TOTAL = "C total: example cash (not profit while holding)"
M_CASH_PANEL = "M: marked value C - sum q/mid (observed mid, zero fee)"
M_OWN_PANEL = "M (COIN)"
PEAK = "running peak max(0, M)"


def drawn(fig: Any, label: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """x (seconds after the origin) and y of the one line drawn with this legend label."""
    lines = [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_label() == label]
    assert len(lines) == 1, f"expected one line labelled {label!r}, found {len(lines)}"
    return (np.asarray(lines[0].get_xdata(), np.float64), np.asarray(lines[0].get_ydata(), np.float64))


def _same(a: float, b: float) -> bool:
    return bool(a == b or (np.isnan(a) and np.isnan(b)))


def assert_drawn_envelope(
    label: str,
    x_drawn: NDArray[np.float64],
    y_drawn: NDArray[np.float64],
    times: NDArray[np.int64],
    values: NDArray[Any],
    manifest: dict[str, Any],
) -> None:
    """Every display bucket of a drawn step line keeps the full path's first, last, minimum and
    maximum point, and its gap (NaN) when it has one.

    The full path is rebuilt from the ledger with the figure's boundary points (flat start,
    last value held to the end) and the bucket geometry the manifest declares; every drawn
    point must be a point of that path, in time order.
    """
    start, end = manifest["full_interval"]["start_ns"], manifest["full_interval"]["end_ns"]
    origin = manifest["transformation"]["origin_ns"]
    width = manifest["display_reduction"]["bucket_width_ns"]
    vals = np.asarray(values, np.float64)
    t = np.concatenate(([start], times, [end])).astype(np.int64)
    y = np.concatenate(([0.0], vals, vals[-1:] if vals.size else [0.0]))
    x = (t - origin) / NS_PER_S  # the figure's own x transformation
    assert x_drawn.size and bool(np.all(np.diff(x_drawn) >= 0)), f"{label}: not in time order"
    at, right = np.searchsorted(x, x_drawn), np.searchsorted(x, x_drawn, side="right")
    assert at[-1] < x.size and np.array_equal(x[at], x_drawn), f"{label}: drew an invented time"
    for i in np.flatnonzero([not _same(a, b) for a, b in zip(y[at], y_drawn, strict=True)]):
        assert any(_same(v, y_drawn[i]) for v in y[at[i] : right[i]]), f"{label}: invented value"
    bucket = (t - start) // width
    shown_bucket = bucket[at]
    for b in np.unique(bucket).tolist():
        lo, hi = np.searchsorted(bucket, [b, b + 1])
        s_lo, s_hi = np.searchsorted(shown_bucket, [b, b + 1])
        full, shown = y[lo:hi], y_drawn[s_lo:s_hi]
        where = f"{label}: bucket {b}"
        assert shown.size, f"{where} lost all its points"
        assert _same(shown[0], full[0]), f"{where} lost its first point"
        assert _same(shown[-1], full[-1]), f"{where} lost its last point"
        fin, fin_shown = full[np.isfinite(full)], shown[np.isfinite(shown)]
        if fin.size:
            assert fin_shown.size and fin_shown.max() == fin.max(), f"{where} lost its maximum"
            assert fin_shown.min() == fin.min(), f"{where} lost its minimum"
        if fin.size < full.size:
            assert fin_shown.size < shown.size, f"{where} lost its gap (NaN)"


def check_qa_accounting(art: FigureArtifact) -> None:
    """Validity and annotations: computed from the full records, whatever the display does."""
    m = art.manifest
    assert m["metrics"]["violations"] == ["cap_exceeded", "not_flat"]
    assert m["metrics"]["net_pnl"] is None
    assert m["metrics"]["max_abs_position_units"] == [1_000_010, 2_000]
    tv = m["terminal_validity"]
    assert tv["valid"] is False and tv["net_pnl"] is None and tv["reason"] == "cap_exceeded"
    assert tv["violations"] == ["cap_exceeded", "not_flat"]
    assert tv["final_position_units"] == [10, 0]
    notes = m["annotations"]
    breach = [a for a in notes if a["kind"] == "cap_breach"]
    assert breach == [
        {
            "kind": "cap_breach",
            "venue": "venue_a/ALPHA-USD",
            "points": 1,
            "first_ns": EXCURSION_NS,
            "max_abs_position_units": 1_000_010,
            "cap_units": 1_000_000,
        }
    ]
    ending = [a for a in notes if a["kind"] == "invalid_ending"]
    assert ending and ending[0]["final_position_units"] == {"venue_a/ALPHA-USD": 10}
    assert m["metrics"]["unmarked_points"] == 1  # the lost venue_a book


def check_qa_display(result: ReplayResult, prep: PreparedMarket, art: FigureArtifact, fig: Any) -> None:
    """What is drawn after display reduction, read from the saved figure.

    The per-bucket envelope of every line is checked first (the mutation canary relies on
    it); then the fixture's three hard cases: the intrabucket cap excursion on the reduced
    venue_a path, the invalid ending as its last drawn point, and M's interior one-point gap.
    """
    m = art.manifest
    keys = [spec.key for spec in prep.data.instruments]
    mark_t, mark = metrics.marked_value_path(result, prep)
    peak = np.maximum.accumulate(np.where(np.isfinite(mark), np.maximum(mark, 0.0), 0.0))
    paths: dict[str, tuple[NDArray[np.int64], NDArray[Any]]] = {
        **{f"{key} position": metrics.position_path(result, v) for v, key in enumerate(keys)},
        C_TOTAL: metrics.cash_path(result),
        **{f"C {key}": metrics.cash_path(result, v) for v, key in enumerate(keys)},
        M_CASH_PANEL: (mark_t, mark),
        M_OWN_PANEL: (mark_t, mark),
        PEAK: (mark_t, peak),
    }
    for label, (times, values) in paths.items():
        assert_drawn_envelope(label, *drawn(fig, label), times, values, m)

    series = series_by_id(m)
    start, end = m["full_interval"]["start_ns"], m["full_interval"]["end_ns"]
    width = m["display_reduction"]["bucket_width_ns"]
    assert m["display_reduction"]["applied"] is True

    def bucket(ns: int) -> int:
        return int((int(ns) - start) // width)

    def x_of(ns: int) -> float:
        return float((np.array([ns], np.int64) - m["transformation"]["origin_ns"])[0] / NS_PER_S)

    # The venue_a path is reduced, and its 1 ms cap excursion is intrabucket: the points just
    # before and after it share its display bucket, so only the bucket-maximum rule keeps it.
    venue_a = series["position.venue_a/ALPHA-USD"]
    xq, yq = drawn(fig, "venue_a/ALPHA-USD position")
    assert venue_a["points_plotted"] == yq.shape[0] < venue_a["points_full"]
    tq, q = metrics.position_path(result, 0)
    hits = np.flatnonzero(np.abs(q) > QA_LIMITS.max_abs_position_units)
    assert hits.shape == (1,)
    k = int(hits[0])
    assert tq[k] == EXCURSION_NS and tq[k + 1] - tq[k] == NS_PER_MS < width
    assert bucket(tq[k - 1]) == bucket(tq[k]) == bucket(tq[k + 1])
    assert venue_a["full_max"] == venue_a["plotted_max"] == 1_000_010
    assert yq[xq == x_of(EXCURSION_NS)].tolist() == [1_000_010.0]
    # The invalid ending: the last drawn point is the open +10 USD at the end of the interval.
    assert (xq[-1], yq[-1]) == (x_of(end), 10.0)
    # M's one-point gap (a lost venue_a book while venue_a is held) is interior to its bucket,
    # so only the gap rule keeps it; both M lines draw it and close it at the next book.
    gap = np.flatnonzero(np.isnan(mark))
    assert mark_t[gap].tolist() == [BAD_BOOK_NS] and series["mark"]["gap_points"] == 1
    g = int(gap[0])
    assert bucket(mark_t[g - 1]) == bucket(mark_t[g]) == bucket(mark_t[g + 1])
    for label in (M_CASH_PANEL, M_OWN_PANEL):
        xm, ym = drawn(fig, label)
        at = np.flatnonzero(xm == x_of(BAD_BOOK_NS))
        assert at.shape == (1,) and np.isnan(ym[at[0]]), f"{label}: the gap is not drawn"
        assert np.isfinite(ym[at[0] - 1]) and np.isfinite(ym[at[0] + 1])
    venue_b = series["position.venue_b/ALPHA-PERP"]  # a small series: drawn in full
    assert venue_b["points_plotted"] == venue_b["points_full"]


@pytest.mark.parametrize("suffix", ["png", "svg"])
def test_qa_figure_draws_the_intrabucket_excursion_the_gap_and_the_invalid_ending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    result, prep, art, fig = render_qa(tmp_path / f"qa.{suffix}", monkeypatch)
    check_qa_accounting(art)
    check_qa_display(result, prep, art, fig)
    if suffix == "svg":
        text = art.image_path.read_text()
        assert "INVALID" in text and "cap breach" in text and "not flat" in text
        assert "SYNTHETIC" in text


def plain_stride(times_ns: Any, values: Any, start_ns: int, end_ns: int, n_buckets: int) -> NDArray[np.int64]:
    """Mutant reducer: every k-th point plus the last one, no extremes kept."""
    n = int(np.asarray(values).shape[0])
    every = np.arange(0, n, -(-n // n_buckets))
    return np.unique(np.concatenate((every, [n - 1]))).astype(np.int64)


def stride_with_global_extremes(
    times_ns: Any, values: Any, start_ns: int, end_ns: int, n_buckets: int
) -> NDArray[np.int64]:
    """Mutant reducer: plain stride plus the global minimum and maximum. It keeps a lone spike,
    so the manifest's plotted extremes match, but not every bucket's extremes."""
    keep = plain_stride(times_ns, values, start_ns, end_ns, n_buckets)
    v = np.asarray(values, np.float64)
    finite = np.flatnonzero(np.isfinite(v))
    extremes = [finite[np.argmin(v[finite])], finite[np.argmax(v[finite])]] if finite.size else []
    return np.union1d(keep, np.asarray(extremes, np.int64)).astype(np.int64)


@pytest.mark.parametrize("reducer", [plain_stride, stride_with_global_extremes])
def test_the_qa_checks_catch_a_reducer_that_drops_bucket_extremes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reducer: Callable[..., NDArray[np.int64]]
) -> None:
    """Mutation canary: the QA display checks must fail on the reduced venue_a position line
    when the pixel envelope is replaced by a reducer that loses intrabucket extremes, while the
    accounting (computed from the full records) still holds."""
    monkeypatch.setattr(charts, "extrema_envelope", reducer)
    result, prep, art, fig = render_qa(tmp_path / "qa.png", monkeypatch)
    check_qa_accounting(art)
    with pytest.raises(AssertionError, match=r"venue_a/ALPHA-USD position: bucket \d+ lost"):
        check_qa_display(result, prep, art, fig)


@pytest.mark.parametrize("suffix", ["png", "svg"])
def test_rendering_is_byte_for_byte_reproducible(tmp_path: Path, suffix: str) -> None:
    pytest.importorskip("matplotlib")
    a = render(tmp_path / "a" / f"pnl.{suffix}")
    b = render(tmp_path / "b" / f"pnl.{suffix}")
    assert a.sha256 == b.sha256
    assert a.manifest["output_sha256"] == b.manifest["output_sha256"]


def test_existing_outputs_are_never_silently_replaced(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    first = render(tmp_path / "pnl.png")
    with pytest.raises(InvalidConfigError):
        render(tmp_path / "pnl.png")
    again = render(tmp_path / "pnl.png", overwrite=True)
    assert again.sha256 == first.sha256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"evidence_scope": "MARKET"},
        {"scenario_name": "higher_fees"},
        {"title": ""},
        {"dpi": 0},
    ],
)
def test_figure_inputs_are_validated(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    with pytest.raises(InvalidConfigError):
        render(tmp_path / "pnl.png", **kwargs)
    assert not (tmp_path / "pnl.png").exists()


def test_only_png_and_svg_outputs_are_supported(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigError):
        render(tmp_path / "pnl.jpg")


def test_missing_matplotlib_is_a_typed_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(OptionalDependencyError) as info:
        render(tmp_path / "pnl.png")
    assert info.value.code == "optional_dependency_missing"
    assert not list(tmp_path.iterdir())


def test_importing_reporting_does_not_import_matplotlib() -> None:
    import financeai

    src = str(Path(financeai.__file__).resolve().parents[1])
    code = (
        "import sys, financeai.reporting, financeai.reporting.charts, "
        "financeai.domains.market.metrics; print('matplotlib' in sys.modules)"
    )
    env = {**os.environ, "PYTHONPATH": src}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "False"


def test_an_idle_account_and_the_two_panel_variant_render(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    _, prep, session = replay()
    idle = run_replay(
        prep,
        np.zeros(len(prep.stream)),
        ThresholdParams(direction=(1, -1), size_units=(2_000, 1_000), enter=0.7, exit=0.0),
        BASE,
        session,
        LIMITS,
    )
    assert len(idle.fills) == 0 and len(idle.orders) == 0
    art = position_pnl_figure(
        idle,
        prep,
        title="Idle",
        scenario_name="base",
        scenario=BASE,
        session=session,
        limits=LIMITS,
        out_path=tmp_path / "idle.svg",
        drawdown=False,
        dpi=80,
    )
    m = art.manifest
    assert m["terminal_validity"]["valid"] is True and m["metrics"]["net_pnl"] == 0.0
    assert m["metrics"]["fill_ratio"] is None and m["metrics"]["max_drawdown"] == 0.0
    series = series_by_id(m)
    assert series["cash.total"]["events"] == 0 and series["cash.total"]["final_value"] == 0.0
    assert "VALID" in art.image_path.read_text()
