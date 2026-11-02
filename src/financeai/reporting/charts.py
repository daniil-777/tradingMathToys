"""Ledger-derived figures (Matplotlib, imported lazily) with a JSON manifest per image.

Every number on a figure is computed by :mod:`financeai.domains.market.metrics` from the
complete ordered fill records *before* any display reduction. The only reduction allowed is
the pixel envelope :func:`extrema_envelope`, which keeps each pixel-wide time bucket's first,
last, minimum and maximum point (and the start of a gap), so a cap excursion shorter than one
pixel, or an invalid ending, can never be thinned away. Cap breaches are additionally drawn as
markers straight from the full records, and an invalid account is stated in words.

Each image ``<name>.png|.svg`` gets ``<name>.<ext>.manifest.json`` (schema 1): run, scenario,
timing mode and evidence scope; SHA-256 of every source fill/order column, the settlement and
the mark inputs; series formulas, units and time basis; the display transformation and
reduction; terminal validity and the ledger reconciliation; renderer/environment hashes and
the output's SHA-256. Rendering is deterministic (fixed fonts, SVG hash salt, no dates), so
regenerating a figure from the same records reproduces its hash.

Matplotlib is only imported inside the rendering functions: importing this module, or the
numerical core, never loads it, and a missing install raises
:class:`~financeai.core.errors.OptionalDependencyError`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

import financeai
from financeai.core.clock import MISSING_NS, NS_PER_DAY, NS_PER_MS, NS_PER_S, format_ns
from financeai.core.errors import InvalidConfigError, InvalidDataError, OptionalDependencyError
from financeai.core.hashing import sha256_file, sha256_json
from financeai.core.jit import HAVE_NUMBA, njit
from financeai.core.serialization import loads_strict
from financeai.domains.market import metrics
from financeai.domains.market.engine import (
    REASON_ENTRY,
    REASON_EXIT,
    REASON_FLATTEN,
    Orders,
    PreparedMarket,
    ReplayResult,
)
from financeai.domains.market.ledger import Fills
from financeai.domains.market.scenario import RiskLimits, Scenario, SessionPlan

EVIDENCE_SCOPES: Final = ("SYNTHETIC", "DEVELOPMENT", "FRESH_TEST")
"""Allowed ``evidence_scope`` values; synthetic figures are labelled as such on the image."""
MANIFEST_SCHEMA: Final = "financeai.figure_manifest"
MANIFEST_VERSION: Final = 1
_FORMATS: Final = {".png": "png", ".svg": "svg"}
_EVIDENCE_LABEL: Final = {
    "SYNTHETIC": "SYNTHETIC DATA - not market evidence",
    "DEVELOPMENT": "DEVELOPMENT replay (in-sample development data)",
    "FRESH_TEST": "FRESH TEST replay",
}

# Colour-blind-safe light-mode palette (validated categorical slots in fixed order; the first
# three pass the all-pairs CVD checks), neutral inks and reserved status colours. Identity is
# never colour-only: each venue also has its own line style and marker.
SURFACE: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
INK_2: Final = "#52514e"
MUTED: Final = "#898781"
GRID: Final = "#e1e0d9"
AXIS: Final = "#c3c2b7"
BAND: Final = "#f0efec"
CRITICAL: Final = "#d03b3b"
GOOD: Final = "#0ca30c"
SERIES: Final = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
LINESTYLES: Final[tuple[Any, ...]] = (
    "solid",
    (0, (6.0, 2.5)),
    (0, (1.2, 1.6)),
    (0, (6.0, 2.0, 1.2, 2.0)),
    (0, (3.0, 1.5)),
    (0, (8.0, 2.0, 2.0, 2.0)),
    (0, (1.0, 3.0)),
)
MARKERS: Final = ("o", "s", "^", "D", "v", "P", "h")
_MARK_STYLE: Final = (0, (6.0, 2.0, 1.2, 2.0))
_WIDTH_IN: Final = 12.0
_FILL_MARKER_LIMIT: Final = 300
_CAP_VIEW_FRACTION: Final = 0.02
_CALLOUT_BOX: Final = {"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": CRITICAL, "lw": 0.8}
_CALLOUT_ARROW: Final = {"arrowstyle": "-", "color": CRITICAL, "lw": 0.8}
_RC: Final = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.linewidth": 0.6,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "path.simplify": False,
    "svg.fonttype": "none",
    "svg.hashsalt": "financeai-figure",
    "lines.solid_joinstyle": "round",
    "lines.dash_joinstyle": "round",
    "text.color": INK,
    "axes.titlecolor": INK,
}


@njit()
def _envelope_keep(
    times: NDArray[np.int64], values: NDArray[np.float64], t_lo: int, width: int, keep: NDArray[np.bool_]
) -> None:
    n = times.shape[0]
    i = 0
    while i < n:
        bucket = (times[i] - t_lo) // width
        lo = -1
        hi = -1
        gap = -1
        j = i
        while j < n and (times[j] - t_lo) // width == bucket:
            x = values[j]
            if x != x:
                if gap < 0:
                    gap = j
            else:
                if lo < 0 or x < values[lo]:
                    lo = j
                if hi < 0 or x > values[hi]:
                    hi = j
            j += 1
        keep[i] = True
        keep[j - 1] = True
        if lo >= 0:
            keep[lo] = True
        if hi >= 0:
            keep[hi] = True
        if gap >= 0:
            keep[gap] = True
            if gap + 1 < j:
                keep[gap + 1] = True
        i = j


def extrema_envelope(
    times_ns: ArrayLike, values: ArrayLike, start_ns: int, end_ns: int, n_buckets: int
) -> NDArray[np.int64]:
    """Indices of the points to draw for a time series shown ``n_buckets`` pixels wide.

    The axis ``[start_ns, end_ns)`` is cut into equal integer-nanosecond buckets; in each
    non-empty bucket the first, last, minimum and maximum finite point are kept, plus the
    first NaN (gap start) and the point after it. Drawn as steps, the kept points cover the
    same vertical extent in every pixel column as the full series, so no extreme is lost; the
    only distortion is the timing of interior points within one bucket. ``times_ns`` must be
    non-decreasing integer nanoseconds.
    """
    times = np.asarray(times_ns)
    if times.ndim != 1 or times.dtype.kind not in "iu" or not np.can_cast(times.dtype, np.int64, "safe"):
        raise InvalidDataError("times must be a 1-D integer array of nanoseconds", context={"dtype": str(times.dtype)})
    vals = np.ascontiguousarray(values, dtype=np.float64)
    if vals.shape != times.shape:
        raise InvalidDataError(
            "times and values must have the same 1-D shape", context={"times": times.shape, "values": vals.shape}
        )
    if times.shape[0] > 1 and bool(np.any(np.diff(times) < 0)):
        raise InvalidDataError("times must be non-decreasing")
    for name, value in (("start_ns", start_ns), ("end_ns", end_ns), ("n_buckets", n_buckets)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise InvalidConfigError(f"{name} must be an integer", context={"got": value})
    if n_buckets < 1 or end_ns <= start_ns:
        raise InvalidConfigError(
            "need end_ns > start_ns and at least one bucket",
            context={"start_ns": start_ns, "end_ns": end_ns, "n_buckets": n_buckets},
        )
    width = max(1, -(-(int(end_ns) - int(start_ns)) // int(n_buckets)))
    keep = np.zeros(times.shape[0], np.bool_)
    _envelope_keep(np.ascontiguousarray(times, dtype=np.int64), vals, int(start_ns), width, keep)
    out = np.flatnonzero(keep).astype(np.int64)
    out.setflags(write=False)
    return out


# ---------------------------------------------------------------------------
# Shared plumbing: validation, hashing, manifest and atomic output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    """A rendered figure: image, manifest (as written) and the image's SHA-256."""

    image_path: Path
    manifest_path: Path
    sha256: str
    manifest: dict[str, Any]


def _matplotlib() -> ModuleType:
    try:
        return importlib.import_module("matplotlib")
    except ImportError as exc:
        raise OptionalDependencyError(
            "matplotlib is required for figures; install the 'report' extra", context={"extra": "report"}
        ) from exc


def _check_output(
    out_path: str | os.PathLike[str], evidence_scope: str, dpi: int, overwrite: bool
) -> tuple[Path, str, Path]:
    path = Path(out_path)
    fmt = _FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise InvalidConfigError("figure output must be a .png or .svg path", context={"path": str(path)})
    if evidence_scope not in EVIDENCE_SCOPES:
        raise InvalidConfigError("unknown evidence scope", context={"got": evidence_scope, "allowed": EVIDENCE_SCOPES})
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 20 <= dpi <= 600:
        raise InvalidConfigError("dpi must be an integer in [20, 600]", context={"got": dpi})
    manifest_path = path.with_name(path.name + ".manifest.json")
    if not overwrite and (path.exists() or manifest_path.exists()):
        raise InvalidConfigError(
            "figure output already exists; pass overwrite=True to replace it", context={"path": str(path)}
        )
    return path, fmt, manifest_path


def _array_record(name: str, arr: NDArray[Any]) -> dict[str, Any]:
    data = np.ascontiguousarray(arr)
    digest = hashlib.sha256(memoryview(data).cast("B")).hexdigest()
    return {"name": name, "dtype": data.dtype.str, "shape": list(data.shape), "sha256": digest}


def _fill_records(fills: Fills) -> list[dict[str, Any]]:
    return [_array_record(f"fills.{f.name}", getattr(fills, f.name)) for f in fields(Fills)]


def _order_records(orders: Orders) -> list[dict[str, Any]]:
    return [_array_record(f"orders.{f.name}", getattr(orders, f.name)) for f in fields(Orders)]


def _instrument_records(prep: PreparedMarket) -> list[dict[str, Any]]:
    return [
        {"venue": i.venue, "symbol": i.symbol, "tick_size": str(i.tick_size), "lot_size": i.lot_size}
        for i in prep.data.instruments
    ]


def _environment(mpl: ModuleType) -> dict[str, Any]:
    try:
        numba_version: str | None = importlib.metadata.version("numba")
    except importlib.metadata.PackageNotFoundError:
        numba_version = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": str(mpl.__version__),
        "numba": numba_version,
        "jit_enabled": HAVE_NUMBA,
        "financeai": financeai.__version__,
    }


def _renderer_sources() -> dict[str, str]:
    return {
        "financeai/reporting/charts.py": sha256_file(__file__),
        "financeai/domains/market/metrics.py": sha256_file(metrics.__file__),
    }


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _publish(fig: Any, path: Path, fmt: str, manifest_path: Path, dpi: int, manifest: dict[str, Any]) -> FigureArtifact:
    """Write the image, then its manifest (which records the image's hash)."""
    metadata: dict[str, str | None] = {"Date": None} if fmt == "svg" else {}

    def save(tmp: Path) -> None:
        fig.savefig(tmp, format=fmt, dpi=dpi, facecolor=SURFACE, metadata=metadata)

    _atomic_write(path, save)
    digest = sha256_file(path)
    manifest["output"] = {"file": path.name, "format": fmt, "dpi": dpi, "bytes": path.stat().st_size, "sha256": digest}
    manifest["output_sha256"] = digest
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"

    def save_manifest(tmp: Path) -> None:
        tmp.write_text(text, encoding="utf-8")

    _atomic_write(manifest_path, save_manifest)
    return FigureArtifact(image_path=path, manifest_path=manifest_path, sha256=digest, manifest=loads_strict(text))


def _base_manifest(
    kind: str,
    path: Path,
    *,
    title: str,
    run_id: str,
    scenario_name: str | None,
    scenario: Scenario | None,
    evidence_scope: str,
    mpl: ModuleType,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    env = _environment(mpl)
    renderer = _renderer_sources()
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "figure_id": f"{kind}/{path.name}",
        "kind": kind,
        "title": title,
        "run_id": run_id,
        "scenario": scenario_name,
        "timing_mode": scenario.timing.name if scenario is not None else None,
        "scenario_parameters": None
        if scenario is None
        else {
            "fees": {k: str(v) for k, v in sorted(scenario.fees.items())},
            "order_latency_ns": scenario.order_latency_ns,
            "ack_latency_ns": scenario.ack_ns,
            "consumption_ttl_ns": scenario.consumption_ttl_ns,
        },
        "evidence_scope": evidence_scope,
        "evidence_label": _EVIDENCE_LABEL[evidence_scope],
        "source_artifacts": sources,
        "timezone": "UTC",
        "metrics_source": "FULL_ORDERED_RECORDS",
        "renderer": {"modules": renderer, "matplotlib": env["matplotlib"], "rc": _rc_record()},
        "renderer_source_sha256": sha256_json(renderer),
        "environment": env,
        "environment_sha256": sha256_json(env),
    }


def _rc_record() -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(_RC.items())}


@contextmanager
def _style(mpl: ModuleType) -> Iterator[None]:
    with mpl.rc_context(_RC):
        yield


def _check_run_inputs(title: object, run_id: str | None) -> str:
    if not isinstance(title, str) or not title.strip():
        raise InvalidConfigError("a figure title is required")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise InvalidConfigError("run_id must be a non-empty string when given")
    return title.strip()


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------

_TICK_STEPS_NS: Final = tuple(
    int(k * unit)
    for unit, ks in (
        (NS_PER_MS, (1, 2, 5, 10, 20, 50, 100, 200, 500)),
        (
            NS_PER_S,
            (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1_800, 3_600, 7_200, 10_800, 14_400, 21_600, 43_200, 86_400),
        ),
    )
    for k in ks
)


def _clock_label(t_ns: int, step_ns: int, with_date: bool) -> str:
    seconds, rem = divmod(t_ns, NS_PER_S)
    stamp = datetime.fromtimestamp(seconds, tz=UTC)
    if step_ns >= 60 * NS_PER_S:
        text = f"{stamp:%H:%M}"
    elif step_ns >= NS_PER_S:
        text = f"{stamp:%H:%M:%S}"
    else:
        text = f"{stamp:%H:%M:%S}.{rem // NS_PER_MS:03d}"
    return f"{stamp:%m-%d} {text}" if with_date else text


def _utc_ticks(start_ns: int, end_ns: int, origin_ns: int, target: int = 8) -> tuple[list[float], list[str], str]:
    """Round UTC wall-clock ticks (exact integer arithmetic) as seconds after ``origin_ns``."""
    span = max(end_ns - start_ns, 1)
    step = next((s for s in _TICK_STEPS_NS if span // s <= target), NS_PER_DAY * -(-span // (target * NS_PER_DAY)))
    first = -(-start_ns // step) * step
    ticks = list(range(first, end_ns + 1, step))
    day0 = datetime.fromtimestamp(start_ns // NS_PER_S, tz=UTC).date()
    day1 = datetime.fromtimestamp(end_ns // NS_PER_S, tz=UTC).date()
    multi_day = day0 != day1
    labels = [_clock_label(t, step, multi_day) for t in ticks]
    title = "time (UTC)" if multi_day else f"time (UTC), {day0.isoformat()}"
    return [(t - origin_ns) / NS_PER_S for t in ticks], labels, title


# ---------------------------------------------------------------------------
# Position / PnL figure
# ---------------------------------------------------------------------------


def _interval(result: ReplayResult, prep: PreparedMarket, session: SessionPlan | None) -> tuple[int, int]:
    """The full span shown: market data, every order/fill/ack clock and the session plan."""
    lo: list[int] = []
    hi: list[int] = []
    stream = prep.stream.recv_ns
    if stream.shape[0]:
        lo.append(int(stream[0]))
        hi.append(int(stream[-1]))
    orders = result.orders
    if len(orders):
        lo.append(int(orders.submit_ns.min()))
        hi.append(int(orders.ack_ns.max()))
    if len(result.fills):
        lo.append(int(result.fills.submit_ns.min()))
        hi.append(int(result.fills.ack_ns.max()))
    if session is not None:
        lo.append(session.start_ns)
        hi.append(session.end_ns)
    if not lo:
        raise InvalidDataError("nothing to plot: no market data, orders or session")
    start, end = min(lo), max(hi)
    return start, (end if end > start else start + NS_PER_S)


@dataclass(frozen=True)
class _Series:
    sid: str
    label: str
    unit: str
    formula: str
    time_basis: str
    times: NDArray[np.int64]
    values: NDArray[np.float64]
    events: int
    keep: NDArray[np.int64]
    integral: bool = False

    def x(self, origin_ns: int) -> NDArray[np.float64]:
        return (self.times[self.keep] - origin_ns) / NS_PER_S

    def y(self) -> NDArray[np.float64]:
        return self.values[self.keep]

    def record(self) -> dict[str, Any]:
        def num(x: float) -> float | int | None:
            if not np.isfinite(x):
                return None
            return int(x) if self.integral else float(x)

        full, shown = self.values, self.values[self.keep]
        finite_full, finite_shown = full[np.isfinite(full)], shown[np.isfinite(shown)]
        rec: dict[str, Any] = {
            "id": self.sid,
            "label": self.label,
            "unit": self.unit,
            "formula": self.formula,
            "time_basis": self.time_basis,
            "events": self.events,
            "points_full": int(full.shape[0]),
            "points_plotted": int(shown.shape[0]),
            "full_min": num(finite_full.min()) if finite_full.size else None,
            "full_max": num(finite_full.max()) if finite_full.size else None,
            "plotted_min": num(finite_shown.min()) if finite_shown.size else None,
            "plotted_max": num(finite_shown.max()) if finite_shown.size else None,
            "final_value": num(full[-1]),
            "gap_points": int(full.shape[0] - finite_full.shape[0]),
        }
        if self.integral and finite_full.size:
            rec["full_max_abs"] = int(np.abs(finite_full).max())
        return rec


def _step_series(
    sid: str,
    label: str,
    unit: str,
    formula: str,
    time_basis: str,
    times: NDArray[np.int64],
    values: NDArray[Any],
    start: int,
    end: int,
    buckets: int,
    *,
    integral: bool = False,
    companion: NDArray[np.float64] | None = None,
) -> _Series:
    """The full step path with a flat start at ``start`` and the last value held to ``end``.

    ``companion`` (same events) is a series drawn with the same kept points, e.g. the drawdown
    of M; its bucket extremes are kept too.
    """
    t = np.concatenate(([start], np.asarray(times, np.int64), [end])).astype(np.int64)
    y = _with_ends(values)
    keep = extrema_envelope(t, y, start, end + 1, buckets)
    if companion is not None:
        keep = np.union1d(keep, extrema_envelope(t, _with_ends(companion), start, end + 1, buckets))
    return _Series(
        sid, label, unit, formula, time_basis, t, y, int(y.shape[0] - 2), np.asarray(keep, np.int64), integral
    )


def _with_ends(values: ArrayLike) -> NDArray[np.float64]:
    """Values with the flat start (0) prepended and the final value repeated at the end."""
    vals = np.asarray(values, dtype=np.float64)
    return np.concatenate(([0.0], vals, [vals[-1] if vals.shape[0] else 0.0]))


def _close_pnl(a: float | None, b: float) -> bool:
    return a is not None and abs(a - b) <= (metrics.RECONCILE_ABS_TOL + metrics.RECONCILE_REL_TOL * max(abs(a), abs(b)))


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_pnl(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.8f}"


def _violation_text(violations: list[str]) -> str:
    words = {"cap_exceeded": "cap exceeded", "not_flat": "not flat at end"}
    return ", ".join(words.get(v, v) for v in violations)


def position_pnl_figure(
    result: ReplayResult,
    prep: PreparedMarket,
    *,
    title: str,
    scenario_name: str,
    out_path: str | os.PathLike[str],
    scenario: Scenario | None = None,
    session: SessionPlan | None = None,
    limits: RiskLimits | None = None,
    evidence_scope: str = "SYNTHETIC",
    run_id: str | None = None,
    drawdown: bool = True,
    dpi: int = 150,
    overwrite: bool = False,
) -> FigureArtifact:
    """Positions, example cash C, marked value M and its drawdown over the full session.

    Panels: (1) per-venue signed units position as post-event steps at fill arrival, with the
    +/-cap lines, cap-breach markers and the terminal status; (2) cumulative example cash C
    (total and per venue) and the marked value M on one settlement currency axis; (3, optional) M on its own
    scale with its running peak, the shaded drawdown and the maximum drawdown; then a table
    with the same numbers. ``limits`` is the replay's cap (default: the example cap);
    ``session`` adds warm-up, entry cutoff, flattening, T and D; ``scenario`` (whose name
    must equal ``scenario_name``) adds the timing mode and an independent fee check.
    """
    path, fmt, manifest_path = _check_output(out_path, evidence_scope, dpi, overwrite)
    title = _check_run_inputs(title, run_id)
    if not isinstance(scenario_name, str) or not scenario_name.strip():
        raise InvalidConfigError("scenario_name is required")
    if scenario is not None and scenario.name != scenario_name:
        raise InvalidConfigError(
            "scenario_name differs from the scenario",
            context={"scenario_name": scenario_name, "scenario": scenario.name},
        )
    nv = len(prep.data)
    if nv >= len(SERIES):
        raise InvalidConfigError("too many venues for one readable figure", context={"venues": nv})
    lim = limits if limits is not None else RiskLimits()
    cap = lim.max_abs_position_units

    # -- metrics from the complete ordered records (before any reduction) -----------------
    stats = metrics.summary(result, prep, scenario, limits=lim)
    keys: list[str] = stats["venues"]
    currency = str(stats["settlement_currency"])
    positions = [metrics.position_path(result, v) for v in range(nv)]
    cash_total = metrics.cash_path(result)
    cash_by_venue = [metrics.cash_path(result, v) for v in range(nv)]
    mark_t, mark = metrics.marked_value_path(result, prep)
    eps = metrics.episodes(result, prep)
    finite = np.isfinite(mark)
    peak = np.maximum.accumulate(np.where(finite, np.maximum(mark, 0.0), 0.0))
    fall = np.where(finite, mark - peak, np.nan)
    start, end = _interval(result, prep, session)
    origin = start
    buckets = int(_WIDTH_IN * dpi)
    fill_sources = _fill_records(result.fills)
    run = run_id.strip() if run_id is not None else ("fills-" + sha256_json([r["sha256"] for r in fill_sources])[:16])

    basis_fill = "fill arrival_ns (exchange arrival on the local clock; true inventory)"
    pos_series = [
        _step_series(
            f"position.{key}",
            f"{key} position",
            "quantity units",
            "q_v(t) = sum(side * qty_units) over the venue's fills with arrival_ns <= t",
            basis_fill,
            *positions[v],
            start,
            end,
            buckets,
            integral=True,
        )
        for v, key in enumerate(keys)
    ]
    c_total = _step_series(
        "cash.total",
        "C total",
        currency,
        "C(t) = sum(instrument-specific cash flows) over fills with arrival_ns <= t; equals net "
        "settlement currency only when every venue is flat",
        basis_fill,
        *cash_total,
        start,
        end,
        buckets,
    )
    c_venues = [
        _step_series(
            f"cash.{key}",
            f"C {key}",
            currency,
            "C_v(t): the venue's share of C(t)",
            basis_fill,
            *cash_by_venue[v],
            start,
            end,
            buckets,
        )
        for v, key in enumerate(keys)
    ]
    mark_formula = (
        "M(t) = C(t) + marked inventory (inverse: -q*factor/mid; linear: q*factor*mid); mid_v is the mid of "
        "the last book received at or before t; NaN while a held venue has no valid "
        "book; zero-fee valuation convention, not an executable liquidation"
    )
    basis_mark = "change points of M: fill arrivals and book receipts (recv_ns) on held venues"
    m_series = _step_series(
        "mark", "M marked value", currency, mark_formula, basis_mark, mark_t, mark, start, end, buckets, companion=fall
    )
    peak_series = _Series(
        "mark.peak",
        "running peak",
        currency,
        "max(0, max_{s<=t} M(s))",
        basis_mark,
        m_series.times,
        _with_ends(peak),
        int(peak.shape[0]),
        m_series.keep,
    )
    dd_series = _Series(
        "mark.drawdown",
        "drawdown of M",
        currency,
        "DD(t) = M(t) - max(0, max_{s<=t} M(s)) (<= 0), NaN where M is NaN",
        basis_mark,
        m_series.times,
        _with_ends(fall),
        int(fall.shape[0]),
        m_series.keep,
    )
    all_series = [*pos_series, c_total, *c_venues, m_series, peak_series, dd_series]
    records = {s.sid: s.record() for s in all_series}
    pos_records = [records[s.sid] for s in pos_series]

    st = result.settlement
    invalid = not st.valid or bool(stats["violations"])
    annotations: list[dict[str, Any]] = []
    for v, key in enumerate(keys):
        t, q = positions[v]
        hits = np.flatnonzero(np.abs(q) > cap)
        if hits.shape[0]:
            annotations.append(
                {
                    "kind": "cap_breach",
                    "venue": key,
                    "points": int(hits.shape[0]),
                    "first_ns": int(t[hits[0]]),
                    "max_abs_position_units": int(np.abs(q[hits]).max()),
                    "cap_units": cap,
                }
            )
    open_end = {key: q for key, q in zip(keys, stats["final_position_units"], strict=True) if q}
    if invalid:
        annotations.append(
            {
                "kind": "invalid_ending",
                "reason": st.reason,
                "violations": stats["violations"],
                "final_position_units": open_end,
                "net_pnl": None,
                "t_ns": end,
            }
        )

    rows = _account_table(stats, eps, nv, cap, open_end, st.reason)
    mpl = _matplotlib()
    figure_mod = importlib.import_module("matplotlib.figure")
    ticker = importlib.import_module("matplotlib.ticker")
    with _style(mpl):
        heights = [3.0, 2.6] + ([1.8] if drawdown else []) + [0.22 * (len(rows) + 1) + 0.25]
        fig = figure_mod.Figure(figsize=(_WIDTH_IN, sum(heights) + 0.7), dpi=dpi, layout="constrained")
        grid = fig.add_gridspec(len(heights), 1, height_ratios=heights)
        ax_pos = fig.add_subplot(grid[0])
        ax_cash = fig.add_subplot(grid[1], sharex=ax_pos)
        ax_dd = fig.add_subplot(grid[2], sharex=ax_pos) if drawdown else None
        ax_tab = fig.add_subplot(grid[-1])
        data_axes = [ax for ax in (ax_pos, ax_cash, ax_dd) if ax is not None]
        x_end = (end - origin) / NS_PER_S
        for ax in data_axes:
            ax.set_xlim(0.0, x_end)
            ax.grid(True, axis="y")
            ax.set_axisbelow(True)
            ax.tick_params(length=2.5, width=0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        # Session clock: warm-up, entry cutoff, forced flattening, after T, D.
        if session is not None:

            def xs(ns: int) -> float:
                return (ns - origin) / NS_PER_S

            for k, ax in enumerate(data_axes):
                first = k == 0
                ax.axvspan(
                    0.0,
                    xs(session.start_ns),
                    color=BAND,
                    lw=0,
                    zorder=0,
                    label="warm-up (before start)" if first else None,
                )
                ax.axvspan(
                    xs(session.flatten_start_ns),
                    xs(session.exec_cutoff_ns),
                    color="#e8e7e1",
                    lw=0,
                    zorder=0,
                    label="forced flattening" if first else None,
                )
                ax.axvspan(
                    xs(session.exec_cutoff_ns),
                    xs(session.end_ns),
                    color="#d9d8d2",
                    lw=0,
                    zorder=0,
                    label="after execution cutoff T, drain to D" if first else None,
                )
                ax.axvline(
                    xs(session.entry_cutoff_ns),
                    color=INK_2,
                    lw=0.7,
                    linestyle=(0, (1.0, 1.5)),
                    zorder=1,
                    label="entry cutoff" if first else None,
                )
                ax.axvline(
                    xs(session.end_ns), color=MUTED, lw=0.7, zorder=1, label="D: end of drain" if first else None
                )

        # (1) positions with cap lines, breach markers and the terminal status.
        max_q = max((r.get("full_max_abs", 0) for r in pos_records), default=0)
        cap_in_view = max_q >= _CAP_VIEW_FRACTION * cap
        cap_label = f"+/-cap {cap:,} units"
        if not cap_in_view:
            cap_label += f" (off-scale: max |q| {max_q:,} units)"
        for sign in (1, -1):
            ax_pos.axhline(
                sign * cap,
                color=INK_2,
                lw=0.8,
                linestyle=(0, (5.0, 3.0)),
                zorder=2,
                label=cap_label if sign > 0 else None,
            )
        ax_pos.axhline(0.0, color=AXIS, lw=0.6, zorder=1)
        for v, s in enumerate(pos_series):
            x, y = s.x(origin), s.y()
            ax_pos.plot(
                x,
                y,
                drawstyle="steps-post",
                color=SERIES[v],
                linestyle=LINESTYLES[v],
                lw=1.0,
                marker=MARKERS[v],
                markevery=[len(x) - 1],
                ms=4.0,
                mec=SURFACE,
                mew=0.6,
                label=s.label,
                zorder=3,
            )
            t, q = positions[v]
            if 0 < t.shape[0] <= _FILL_MARKER_LIMIT:
                ax_pos.plot(
                    (t - origin) / NS_PER_S,
                    q,
                    linestyle="none",
                    marker=MARKERS[v],
                    ms=3.2,
                    mfc=SERIES[v],
                    mec=SURFACE,
                    mew=0.5,
                    zorder=4,
                )
        breach_label = "cap breach (|q| > cap), from full records"
        for note in (a for a in annotations if a["kind"] == "cap_breach"):
            v = keys.index(note["venue"])
            t, q = positions[v]
            hits = np.flatnonzero(np.abs(q) > cap)
            bx = (t[hits] - origin) / NS_PER_S
            ax_pos.scatter(
                bx,
                q[hits],
                marker="X",
                s=46,
                color=CRITICAL,
                edgecolors=SURFACE,
                linewidths=0.6,
                zorder=6,
                label=breach_label,
            )
            breach_label = ""
            right = bx[0] > 0.5 * x_end
            callout = ax_pos.annotate(
                f"cap breach ({note['venue']})\n|q| = {note['max_abs_position_units']:,} units > cap {cap:,} units",
                xy=(bx[0], q[hits[0]]),
                xytext=(-12 if right else 12, -18 if q[hits[0]] > 0 else 18),
                textcoords="offset points",
                ha="right" if right else "left",
                va="center",
                fontsize=7.5,
                bbox=_CALLOUT_BOX,
                arrowprops=_CALLOUT_ARROW,
                zorder=7,
            )
            callout.set_in_layout(False)
        for k, (key, q_end) in enumerate(open_end.items()):
            ax_pos.plot(
                [x_end], [q_end], marker="o", ms=7.0, mfc="none", mec=CRITICAL, mew=1.2, linestyle="none", zorder=7
            )
            callout = ax_pos.annotate(
                f"not flat at end ({key})\nfinal q = {q_end:+,} units",
                xy=(x_end, q_end),
                xytext=(-14, 22 + 26 * k),
                textcoords="offset points",
                ha="right",
                fontsize=7.5,
                bbox=_CALLOUT_BOX,
                arrowprops=_CALLOUT_ARROW,
                zorder=7,
            )
            callout.set_in_layout(False)
        if cap_in_view:
            hi = max(cap, max((r["full_max"] or 0 for r in pos_records), default=0))
            lo = min(-cap, min((r["full_min"] or 0 for r in pos_records), default=0))
            ax_pos.set_ylim(lo * 1.12, hi * 1.12)
        else:
            span = max(max_q, 1) * 1.25
            ax_pos.set_ylim(-span, span)
        ax_pos.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))
        ax_pos.set_ylabel("position (quantity units)")
        if invalid:
            status = f"INVALID: {_violation_text(stats['violations'] or [str(st.reason)])} - no net settlement currency"
        else:
            status = f"VALID, flat at end - net settlement currency {stats['net_pnl']:+.8f}"
        ax_pos.set_title(
            status,
            loc="right",
            fontsize=8.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": CRITICAL if invalid else GOOD, "lw": 1.0},
        )
        ax_pos.set_title("(1) positions at fill arrival (true inventory)", loc="left")
        ax_pos.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)

        # (2) example cash C and marked value M (one settlement currency axis).
        ax_cash.axhline(0.0, color=AXIS, lw=0.6, zorder=1)
        ax_cash.plot(
            c_total.x(origin),
            c_total.y(),
            drawstyle="steps-post",
            color=INK,
            lw=1.2,
            label="C total: example cash (not profit while holding)",
            zorder=4,
        )
        for v, s in enumerate(c_venues):
            ax_cash.plot(
                s.x(origin),
                s.y(),
                drawstyle="steps-post",
                color=SERIES[v],
                linestyle=LINESTYLES[v],
                lw=0.9,
                label=s.label,
                zorder=3,
            )
        ax_cash.plot(
            m_series.x(origin),
            m_series.y(),
            drawstyle="steps-post",
            color=SERIES[nv],
            linestyle=_MARK_STYLE,
            lw=1.2,
            label="M: marked value C - sum q/mid (observed mid, zero fee)",
            zorder=5,
        )
        ax_cash.set_ylabel(currency)
        ax_cash.ticklabel_format(axis="y", useOffset=False)
        ax_cash.set_title("(2) example cash C and marked value M", loc="left")
        ax_cash.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)

        # (3) M on its own scale with the running peak and the drawdown.
        if ax_dd is not None:
            xm = m_series.x(origin)
            ym = m_series.y()
            yp = peak_series.y()
            shown = np.isfinite(ym)
            ax_dd.fill_between(
                xm,
                np.where(shown, ym, yp),
                yp,
                step="post",
                where=shown,
                color=SERIES[nv],
                alpha=0.18,
                lw=0.0,
                label="drawdown below peak",
            )
            ax_dd.plot(
                xm,
                yp,
                drawstyle="steps-post",
                color=INK_2,
                lw=0.8,
                linestyle=(0, (1.0, 1.5)),
                label="running peak max(0, M)",
            )
            ax_dd.plot(
                xm,
                ym,
                drawstyle="steps-post",
                color=SERIES[nv],
                linestyle=_MARK_STYLE,
                lw=1.1,
                label=f"M ({currency})",
            )
            trough = stats["drawdown_trough_ns"]
            if trough is not None:
                j = int(np.searchsorted(mark_t, trough))
                ax_dd.plot(
                    [(trough - origin) / NS_PER_S],
                    [mark[j]],
                    marker="v",
                    ms=6.0,
                    color=INK,
                    mec=SURFACE,
                    mew=0.6,
                    linestyle="none",
                    label=f"max drawdown {stats['max_drawdown']:.8f} settlement currency",
                )
            ax_dd.set_ylabel(f"M ({currency})")
            ax_dd.ticklabel_format(axis="y", useOffset=False)
            ax_dd.set_title("(3) marked value M on its own scale, with its drawdown", loc="left")
            ax_dd.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)

        tick_x, tick_labels, axis_title = _utc_ticks(start, end, origin)
        bottom = data_axes[-1]
        bottom.set_xticks(tick_x, tick_labels)
        bottom.set_xlabel(axis_title)
        for ax in data_axes[:-1]:
            ax.tick_params(labelbottom=False)

        # Table view: the same numbers as text.
        ax_tab.axis("off")
        table = ax_tab.table(
            cellText=rows,
            colLabels=["metric", *keys, "account"],
            loc="upper center",
            cellLoc="right",
            colLoc="right",
            edges="horizontal",
            colWidths=[0.26] + [0.74 / (nv + 1)] * (nv + 1),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.set_in_layout(False)  # fixed geometry; measuring 40+ cells per layout pass is slow
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.5)
            if c == 0:
                cell.get_text().set_horizontalalignment("left")
            if r == 0:
                cell.get_text().set_fontweight("bold")
        timing = scenario.timing.name if scenario is not None else "undeclared"
        ax_tab.set_title(
            f"table view of the full ordered records - run {run} - scenario "
            f"{scenario_name} - timing {timing} - evidence {evidence_scope}",
            loc="left",
            fontsize=8.0,
            color=INK_2,
        )
        fig.suptitle(
            f"{title}  [{_EVIDENCE_LABEL[evidence_scope]}]",
            x=0.005,
            horizontalalignment="left",
            fontsize=12,
            fontweight="bold",
        )

        series_records = [records[s.sid] for s in all_series]
        plotted_ok = all(
            r["final_value"] == q for r, q in zip(pos_records, st.final_position_units, strict=True)
        ) and all(_close_pnl(records[s.sid]["final_value"], c) for s, c in zip(c_venues, st.cash, strict=True))
        sources = [
            *fill_sources,
            *_order_records(result.orders),
            {"name": "settlement", "sha256": sha256_json(asdict(st))},
            {
                "name": "instruments",
                "items": _instrument_records(prep),
                "sha256": sha256_json(_instrument_records(prep)),
            },
        ]
        for v, key in enumerate(keys):
            sources += [
                _array_record(f"marks.{key}.book_recv_ns", prep.book_recv[v]),
                _array_record(f"marks.{key}.best_bid_ticks", prep.bpx[v][:, 0]),
                _array_record(f"marks.{key}.best_ask_ticks", prep.apx[v][:, 0]),
                _array_record(f"marks.{key}.book_valid", prep.valid[v]),
            ]
        manifest = _base_manifest(
            "position_pnl",
            path,
            title=title,
            run_id=run,
            scenario_name=scenario_name,
            scenario=scenario,
            evidence_scope=evidence_scope,
            mpl=mpl,
            sources=sources,
        )
        manifest.update(
            {
                "fills_sha256": sha256_json([r["sha256"] for r in fill_sources]),
                "full_interval": {
                    "start_ns": start,
                    "end_ns": end,
                    "start_utc": format_ns(start),
                    "end_utc": format_ns(end),
                    "covers": "market data, every order/fill/ack clock and the session",
                },
                "session": asdict(session) if session is not None else None,
                "cap_units": cap,
                "cap_lines_in_view": cap_in_view,
                "time_basis": {
                    "clock": "local receipt clock, UTC epoch nanoseconds (int64)",
                    "positions_and_cash": basis_fill,
                    "marks": "book recv_ns: last book received at or before t",
                    "known_inventory": "ack_ns (controller view; not plotted)",
                    "session": "SessionPlan clocks on the local receipt clock",
                },
                "series_definitions": series_records,
                "transformation": {
                    "x_seconds": "(t_ns - origin_ns) / 1e9, display only",
                    "origin_ns": origin,
                    "origin_utc": format_ns(origin),
                    "x_ticks": "round UTC wall-clock times computed with integer nanoseconds",
                    "steps": "post-event: each value holds until the next point",
                    "boundary_points": "flat start at the interval start; final value held to its end",
                    "y_units": {
                        "position": "quantity units (unscaled)",
                        "C, M, drawdown": "settlement currency (unscaled)",
                    },
                },
                "display_reduction": {
                    "method": "extrema_envelope",
                    "applied": any(r["points_plotted"] < r["points_full"] for r in series_records),
                    "buckets": buckets,
                    "bucket_width_ns": max(1, -(-(end + 1 - start) // buckets)),
                    "kept_per_bucket": ["first", "last", "min", "max", "first NaN", "point after the first NaN"],
                    "cap_breach_markers": "drawn from the full records (never reduced)",
                    "metrics_computed_before_reduction": True,
                },
                "terminal_validity": {
                    "valid": bool(st.valid) and not stats["violations"],
                    "reason": st.reason,
                    "violations": stats["violations"],
                    "net_pnl": stats["net_pnl"],
                    "final_position_units": stats["final_position_units"],
                    "rule": (
                        "net settlement currency exists only if every venue ends flat and no venue exceeded the cap"
                    ),
                },
                "reconciliation_record": {
                    "reconciled": bool(stats["reconciled"]) and plotted_ok,
                    "ledger_checks": stats["reconciliation"],
                    "plotted_finals_match_ledger": plotted_ok,
                    "tolerance": {"abs_pnl": metrics.RECONCILE_ABS_TOL, "rel": metrics.RECONCILE_REL_TOL},
                },
                "metrics": stats,
                "annotations": annotations,
                "liquidation": "not shown: M is a zero-fee observed-mid mark, not a liquidation value",
            }
        )
        return _publish(fig, path, fmt, manifest_path, dpi, manifest)


def _account_table(
    stats: dict[str, Any], eps: metrics.Episodes, nv: int, cap: int, open_end: dict[str, int], reason: str | None
) -> list[list[str]]:
    """Rows of the figure's table view (per venue and account), from the summary."""
    trips = [int(np.count_nonzero(eps.closed & (eps.venue == v))) for v in range(nv)]
    wins = [
        int(np.count_nonzero(eps.closed & (eps.venue == v) & (np.nan_to_num(eps.net_pnl) > 0.0))) for v in range(nv)
    ]
    dash = ["-"] * nv
    ratio = stats["fill_ratio"]
    return [
        ["fills", *map(_fmt_int, stats["fills_by_venue"]), _fmt_int(stats["fills"])],
        ["traded (quantity units)", *map(_fmt_int, stats["traded_units_by_venue"]), _fmt_int(stats["traded_units"])],
        ["max |q| (quantity units)", *map(_fmt_int, stats["max_abs_position_units"]), f"cap {cap:,}"],
        ["final q (quantity units)", *map(_fmt_int, stats["final_position_units"]), "NOT FLAT" if open_end else "flat"],
        ["example cash C (settlement currency)", *map(_fmt_pnl, stats["cash_by_venue"]), _fmt_pnl(stats["final_cash"])],
        ["fees (settlement currency)", *(f"{x:.8f}" for x in stats["fees_by_venue"]), f"{stats['fees']:.8f}"],
        [
            "round trips (wins)",
            *(f"{n:,} ({w:,})" for n, w in zip(trips, wins, strict=True)),
            f"{stats['round_trips']:,} ({stats['wins']:,})",
        ],
        [
            "net settlement currency (valid accounts only)",
            *dash,
            _fmt_pnl(stats["net_pnl"]) if stats["net_pnl"] is not None else f"unavailable: {reason}",
        ],
        ["max drawdown of M (settlement currency)", *dash, f"{stats['max_drawdown']:.8f}"],
        [
            "orders (fill ratio)",
            *dash,
            f"{stats['orders']:,} ({ratio:.1%})" if ratio is not None else f"{stats['orders']:,} (n/a)",
        ],
    ]


# ---------------------------------------------------------------------------
# Decision trace
# ---------------------------------------------------------------------------

_REASONS: Final = {REASON_ENTRY: "entry", REASON_EXIT: "exit", REASON_FLATTEN: "flatten"}
_TRACE_TEXT_LIMIT: Final = 12
_CONNECTOR_LIMIT: Final = 400


def _window_mask(clock: NDArray[np.int64], t0: int, t1: int) -> NDArray[np.bool_]:
    return (clock >= t0) & (clock <= t1)


def decision_trace_figure(
    result: ReplayResult,
    prep: PreparedMarket,
    t0_ns: int,
    t1_ns: int,
    out_path: str | os.PathLike[str],
    *,
    title: str | None = None,
    scenario: Scenario | None = None,
    evidence_scope: str = "SYNTHETIC",
    run_id: str | None = None,
    dpi: int = 150,
    overwrite: bool = False,
    max_events: int = 20_000,
) -> FigureArtifact:
    """Every event of ``[t0_ns, t1_ns]`` on one timeline, visible and hidden clocks apart.

    Lanes per venue: observations as *received* (``recv_ns``, visible to the controller) and
    the same books/trades at their *exchange* timestamps (``exch_ns``, hidden). Order lanes:
    submission (visible, the controller's own clock), exchange arrival with its fills (hidden:
    only the simulator knows them then) and the acknowledgement (visible). Each order is linked
    to the book it met, which shows when that book was not yet received at submission. Nothing
    is reduced: a window with more than ``max_events`` events is refused.
    """
    path, fmt, manifest_path = _check_output(out_path, evidence_scope, dpi, overwrite)
    for name, value in (("t0_ns", t0_ns), ("t1_ns", t1_ns), ("max_events", max_events)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer):
            raise InvalidConfigError(f"{name} must be an integer", context={"got": value})
    t0, t1 = int(t0_ns), int(t1_ns)
    if t1 <= t0:
        raise InvalidConfigError("decision-trace window needs t1_ns > t0_ns", context={"t0_ns": t0, "t1_ns": t1})
    if max_events < 1:
        raise InvalidConfigError("max_events must be positive")
    title = _check_run_inputs(
        title if title is not None else f"Decision trace {format_ns(t0)} to {format_ns(t1)}", run_id
    )
    nv = len(prep.data)
    if nv != len(result.settlement.final_position_units):
        raise InvalidConfigError("prepared market does not match the replayed venues")
    if nv >= len(SERIES):
        raise InvalidConfigError("too many venues for one readable figure", context={"venues": nv})
    keys = [spec.key for spec in prep.data.instruments]
    fills, orders = result.fills, result.orders

    books = [
        np.flatnonzero(_window_mask(t.book.recv_ns, t0, t1) | _window_mask(t.book.exch_ns, t0, t1))
        for t in prep.data.venues
    ]
    trades = [
        np.flatnonzero(_window_mask(t.trades.recv_ns, t0, t1) | _window_mask(t.trades.exch_ns, t0, t1))
        for t in prep.data.venues
    ]
    o_sel = np.flatnonzero(
        _window_mask(orders.submit_ns, t0, t1)
        | _window_mask(orders.arrival_ns, t0, t1)
        | _window_mask(orders.ack_ns, t0, t1)
    )
    f_sel = np.flatnonzero(np.isin(fills.order_id, orders.order_id[o_sel]))
    counts = {
        "books": int(sum(b.shape[0] for b in books)),
        "trades": int(sum(t.shape[0] for t in trades)),
        "orders": int(o_sel.shape[0]),
        "fills": int(f_sel.shape[0]),
    }
    n_events = counts["books"] + counts["trades"] + 3 * counts["orders"] + counts["fills"]
    if n_events > max_events:
        raise InvalidConfigError(
            "window holds too many events for a readable trace; narrow it",
            context={"events": n_events, "max_events": max_events},
        )

    order_rows: list[dict[str, Any]] = []
    for k in o_sel.tolist():
        v = int(orders.venue[k])
        book = prep.data.venues[v].book
        submit = int(orders.submit_ns[k])
        row = int(orders.book_row[k])
        mine = f_sel[fills.order_id[f_sel] == orders.order_id[k]]
        tick = float(prep.data.instruments[v].tick_size)
        order_rows.append(
            {
                "order_id": int(orders.order_id[k]),
                "venue": keys[v],
                "side": "BUY" if orders.side[k] > 0 else "SELL",
                "reason": _REASONS.get(int(orders.reason[k]), str(int(orders.reason[k]))),
                "qty_units": int(orders.qty_units[k]),
                "filled_units": int(orders.filled_units[k]),
                "limit_ticks": int(orders.limit_ticks[k]),
                "submit_ns": submit,
                "arrival_ns": int(orders.arrival_ns[k]),
                "ack_ns": int(orders.ack_ns[k]),
                "observed_book_row": int(np.searchsorted(book.recv_ns, submit, side="right")) - 1,
                "book_row": row,
                "book_exch_ns": (int(book.exch_ns[row]) if row >= 0 and int(book.exch_ns[row]) != MISSING_NS else None),
                "book_recv_ns": int(book.recv_ns[row]) if row >= 0 else None,
                "book_hidden_at_submit": bool(row >= 0 and int(book.recv_ns[row]) > submit),
                "fills": [
                    {
                        "px_ticks": int(fills.px_ticks[i]),
                        "price_units": float(fills.px_ticks[i]) * tick,
                        "qty_units": int(fills.qty_units[i]),
                    }
                    for i in mine.tolist()
                ],
            }
        )

    lanes: list[dict[str, Any]] = []
    for spec in prep.data.instruments:
        lanes.append(
            {"lane": f"{spec.venue} received [visible]", "clock": "recv_ns", "visibility": "visible", "venue": spec.key}
        )
        lanes.append(
            {
                "lane": f"{spec.venue} exchange time [hidden]",
                "clock": "exch_ns",
                "visibility": "hidden",
                "venue": spec.key,
            }
        )
    lanes += [
        {"lane": "order submitted [visible]", "clock": "submit_ns", "visibility": "visible", "venue": None},
        {"lane": "exchange arrival + fills [hidden]", "clock": "arrival_ns", "visibility": "hidden", "venue": None},
        {"lane": "fill report acknowledged [visible]", "clock": "ack_ns", "visibility": "visible", "venue": None},
    ]
    n_lanes = len(lanes)
    y_recv = [n_lanes - 1 - 2 * v for v in range(nv)]
    y_exch = [n_lanes - 2 - 2 * v for v in range(nv)]
    y_sub, y_arr, y_ack = 2, 1, 0

    fill_sources = _fill_records(fills)
    run = run_id.strip() if run_id is not None else ("fills-" + sha256_json([r["sha256"] for r in fill_sources])[:16])
    sources = [
        *fill_sources,
        *_order_records(orders),
        {"name": "settlement", "sha256": sha256_json(asdict(result.settlement))},
    ]
    for v, key in enumerate(keys):
        tapes = prep.data.venues[v]
        sources += [
            _array_record(f"window.{key}.book_recv_ns", tapes.book.recv_ns[books[v]]),
            _array_record(f"window.{key}.book_exch_ns", tapes.book.exch_ns[books[v]]),
            _array_record(f"window.{key}.trade_recv_ns", tapes.trades.recv_ns[trades[v]]),
            _array_record(f"window.{key}.trade_exch_ns", tapes.trades.exch_ns[trades[v]]),
        ]

    mpl = _matplotlib()
    figure_mod = importlib.import_module("matplotlib.figure")
    collections = importlib.import_module("matplotlib.collections")
    lines_mod = importlib.import_module("matplotlib.lines")
    patches = importlib.import_module("matplotlib.patches")

    def xs(ns: NDArray[np.int64] | int) -> Any:
        return (np.asarray(ns, np.int64) - t0) / NS_PER_MS

    table_rows = order_rows[:_TRACE_TEXT_LIMIT]
    with _style(mpl):
        trace_h = 1.9 + 0.42 * n_lanes
        table_h = 0.22 * (len(table_rows) + 1) + 0.35 if table_rows else 0.0
        fig = figure_mod.Figure(figsize=(_WIDTH_IN, trace_h + table_h + 0.5), dpi=dpi, layout="constrained")
        if table_rows:
            grid = fig.add_gridspec(2, 1, height_ratios=[trace_h, table_h])
            ax = fig.add_subplot(grid[0])
            ax_tab = fig.add_subplot(grid[1])
        else:
            ax = fig.add_subplot()
            ax_tab = None
        for k, lane in enumerate(lanes):
            if lane["visibility"] == "hidden":
                y = n_lanes - 1 - k
                ax.axhspan(y - 0.42, y + 0.42, color=BAND, lw=0, zorder=0)
        for v in range(nv):
            tapes = prep.data.venues[v]
            color = SERIES[v]
            for idx, recv_all, exch_all, marker, size in (
                (books[v], tapes.book.recv_ns, tapes.book.exch_ns, "o", 22),
                (trades[v], tapes.trades.recv_ns, tapes.trades.exch_ns, "D", 16),
            ):
                if idx.shape[0] == 0:
                    continue
                recv, exch = recv_all[idx], exch_all[idx]
                seen = _window_mask(recv, t0, t1)
                stamped = _window_mask(exch, t0, t1)
                ax.scatter(
                    xs(recv[seen]),
                    np.full(int(seen.sum()), y_recv[v]),
                    marker=marker,
                    s=size,
                    color=color,
                    edgecolors=SURFACE,
                    linewidths=0.6,
                    zorder=4,
                )
                ax.scatter(
                    xs(exch[stamped]),
                    np.full(int(stamped.sum()), y_exch[v]),
                    marker=marker,
                    s=size,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.0,
                    zorder=4,
                )
                both = seen & stamped
                if 0 < int(both.sum()) <= _CONNECTOR_LIMIT:
                    segs = np.stack(
                        [
                            np.column_stack([xs(exch[both]), np.full(int(both.sum()), y_exch[v])]),
                            np.column_stack([xs(recv[both]), np.full(int(both.sum()), y_recv[v])]),
                        ],
                        axis=1,
                    )
                    ax.add_collection(collections.LineCollection(segs, colors=MUTED, linewidths=0.5, zorder=2))

        def link(x0: float, y0: float, x1: float, y1: float) -> None:
            # A patch on the axes is clipped to it, so clocks outside the window stay outside.
            ax.add_patch(
                patches.FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="->",
                    mutation_scale=9.0,
                    color=INK_2,
                    lw=0.8,
                    shrinkA=4.0,
                    shrinkB=4.0,
                    zorder=3,
                )
            )

        x_max = float(xs(t1))
        tags: dict[int, list[str]] = {}
        for o in order_rows:
            v = keys.index(o["venue"])
            color, marker = SERIES[v], ("^" if o["side"] == "BUY" else "v")
            xs_s, xs_a, xs_k = xs(o["submit_ns"]), xs(o["arrival_ns"]), xs(o["ack_ns"])
            ax.scatter([xs_s], [y_sub], marker=marker, s=48, color=color, edgecolors=SURFACE, linewidths=0.6, zorder=5)
            ax.scatter(
                [xs_a], [y_arr], marker=marker, s=48, facecolors="none", edgecolors=color, linewidths=1.1, zorder=5
            )
            ax.scatter([xs_k], [y_ack], marker="s", s=30, color=color, edgecolors=SURFACE, linewidths=0.6, zorder=5)
            if o["fills"]:
                ax.scatter([xs_a], [y_arr + 0.22], marker="|", s=70, color=INK, linewidths=1.2, zorder=5)
            link(xs_s, y_sub, xs_a, y_arr)
            link(xs_a, y_arr, xs_k, y_ack)
            if o["book_row"] >= 0:
                bx_r = xs(o["book_recv_ns"])
                ax.scatter(
                    [bx_r], [y_recv[v]], marker="o", s=120, facecolors="none", edgecolors=INK, linewidths=1.1, zorder=6
                )
                if o["book_exch_ns"] is not None:
                    bx_e = xs(o["book_exch_ns"])
                    ax.scatter(
                        [bx_e],
                        [y_exch[v]],
                        marker="o",
                        s=120,
                        facecolors="none",
                        edgecolors=INK,
                        linewidths=1.1,
                        zorder=6,
                    )
                    ax.plot([bx_e, xs_a], [y_exch[v], y_arr], linestyle=(0, (1.0, 1.5)), color=INK, lw=0.8, zorder=3)
            if t0 <= o["arrival_ns"] <= t1:
                tags.setdefault(o["arrival_ns"], []).append(f"#{o['order_id']}")
        # One tag per arrival instant (orders on several venues can arrive together); the
        # details are in the table view below.
        for arrival_ns, ids in sorted(tags.items()):
            xa = float(xs(arrival_ns))
            right = xa > 0.85 * x_max
            ax.annotate(
                ", ".join(ids),
                xy=(xa, y_arr),
                xytext=(-7 if right else 7, -9),
                textcoords="offset points",
                ha="right" if right else "left",
                va="top",
                fontsize=7.0,
                color=INK,
                zorder=7,
            ).set_in_layout(False)
            submit_ns = next(o["submit_ns"] for o in order_rows if o["arrival_ns"] == arrival_ns)
            if submit_ns >= t0:
                ax.annotate(
                    f"{(arrival_ns - submit_ns) / NS_PER_MS:.3f} ms",
                    xy=((float(xs(submit_ns)) + xa) / 2, (y_sub + y_arr) / 2),
                    xytext=(-4, 0),
                    textcoords="offset points",
                    ha="right",
                    fontsize=6.5,
                    color=INK_2,
                    va="center",
                ).set_in_layout(False)
        ax.set_yticks(range(n_lanes), [lane["lane"] for lane in reversed(lanes)])
        ax.set_ylim(-0.9, n_lanes - 0.4)
        ax.set_xlim(0.0, float(xs(t1)))
        ax.set_xlabel(f"milliseconds after {format_ns(t0)} (UTC)")
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        timing = scenario.timing.name if scenario is not None else "undeclared"
        latency = (
            f" - order latency {scenario.order_latency_ns / NS_PER_MS:.3f} ms - ack "
            f"latency {scenario.ack_ns / NS_PER_MS:.3f} ms"
            if scenario is not None
            else ""
        )
        ax.set_title(
            f"run {run} - scenario {scenario.name if scenario is not None else 'n/a'} - timing {timing}{latency}",
            loc="left",
            fontsize=8.0,
            color=INK_2,
        )
        handles = [
            lines_mod.Line2D([], [], marker="o", linestyle="none", color=SERIES[v], label=key)
            for v, key in enumerate(keys)
        ]
        handles += [
            lines_mod.Line2D(
                [], [], marker="o", linestyle="none", color=INK, label="filled marker: clock visible to the controller"
            ),
            lines_mod.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                mfc="none",
                color=INK,
                label="hollow marker: hidden clock (exchange / simulator only)",
            ),
            patches.Patch(color=BAND, label="hidden-clock lane"),
            lines_mod.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=MUTED,
                label="o book snapshot, D trade print (grey line: feed delay)",
            ),
            lines_mod.Line2D(
                [], [], marker="^", linestyle="none", color=MUTED, label="order submitted (^ buy, v sell)"
            ),
            lines_mod.Line2D(
                [], [], marker="|", linestyle="none", color=INK, markersize=9, label="fill at exchange arrival"
            ),
            lines_mod.Line2D([], [], marker="s", linestyle="none", color=MUTED, label="fill report acknowledged"),
            lines_mod.Line2D(
                [],
                [],
                marker="o",
                linestyle=(0, (1.0, 1.5)),
                mfc="none",
                color=INK,
                markersize=8,
                label="book met by the order",
            ),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, borderaxespad=0.0)
        if ax_tab is not None:
            ax_tab.axis("off")

            def ms(ns: int | None) -> str:
                return "-" if ns is None else f"{(ns - t0) / NS_PER_MS:.3f}"

            cells = []
            for o in table_rows:
                levels = ", ".join(f"{f['qty_units']:,} @ {f['price_units']:,.1f}" for f in o["fills"][:2])
                if len(o["fills"]) > 2:
                    levels += f" (+{len(o['fills']) - 2})"
                venue = prep.data.instruments[keys.index(o["venue"])].venue
                cells.append(
                    [
                        f"#{o['order_id']}",
                        venue,
                        f"{o['side']} ({o['reason']})",
                        f"{o['qty_units']:,} / {o['filled_units']:,}",
                        f"{ms(o['submit_ns'])} / {ms(o['arrival_ns'])} / {ms(o['ack_ns'])}",
                        levels or "no fill",
                        f"{ms(o['book_exch_ns'])} / {ms(o['book_recv_ns'])}",
                        "hidden" if o["book_hidden_at_submit"] else ("seen" if o["book_row"] >= 0 else "-"),
                    ]
                )
            table = ax_tab.table(
                cellText=cells,
                colLabels=[
                    "order",
                    "venue",
                    "side",
                    "qty / filled units",
                    "submit / arrival / ack",
                    "fills (USD @ price)",
                    "book met exch / recv",
                    "at submit",
                ],
                loc="upper left",
                cellLoc="left",
                colLoc="left",
                edges="horizontal",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.0)
            table.auto_set_column_width(list(range(8)))  # sized from the text at draw time
            table.set_in_layout(False)
            for (r, _), cell in table.get_celld().items():
                cell.set_edgecolor(GRID)
                cell.set_linewidth(0.5)
                if r == 0:
                    cell.get_text().set_fontweight("bold")
            extra = len(order_rows) - len(table_rows)
            ax_tab.set_title(
                "orders in the window (times in ms after the window start)"
                + (f"; {extra} more in the manifest" if extra else ""),
                loc="left",
                fontsize=8.0,
                color=INK_2,
            )
        fig.suptitle(
            f"{title}  [{_EVIDENCE_LABEL[evidence_scope]}]",
            x=0.005,
            horizontalalignment="left",
            fontsize=12,
            fontweight="bold",
        )

        st = result.settlement
        manifest = _base_manifest(
            "decision_trace",
            path,
            title=title,
            run_id=run,
            scenario_name=scenario.name if scenario is not None else None,
            scenario=scenario,
            evidence_scope=evidence_scope,
            mpl=mpl,
            sources=sources,
        )
        manifest.update(
            {
                "fills_sha256": sha256_json([r["sha256"] for r in fill_sources]),
                "window": {"t0_ns": t0, "t1_ns": t1, "t0_utc": format_ns(t0), "t1_utc": format_ns(t1)},
                "counts": counts,
                "lanes": lanes,
                "orders": order_rows,
                "time_basis": {
                    "clock": "local receipt clock, UTC epoch nanoseconds (int64)",
                    "visible": "recv_ns of observations, submit_ns of own orders, ack_ns of reports",
                    "hidden": "exch_ns of observations, arrival_ns of orders and their fills",
                },
                "transformation": {"x_ms": "(t_ns - t0_ns) / 1e6, display only", "origin_ns": t0},
                "display_reduction": {
                    "method": "none",
                    "note": "every event in the window is drawn",
                    "max_events": max_events,
                },
                "account_validity": {"valid": bool(st.valid), "reason": st.reason, "net_pnl": st.net_pnl},
            }
        )
        return _publish(fig, path, fmt, manifest_path, dpi, manifest)


__all__ = ["EVIDENCE_SCOPES", "FigureArtifact", "decision_trace_figure", "extrema_envelope", "position_pnl_figure"]
