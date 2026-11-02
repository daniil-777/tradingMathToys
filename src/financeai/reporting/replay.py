"""Offline interactive replay reports; optional Plotly is loaded only when writing HTML."""

from __future__ import annotations

import html
import importlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from decimal import Decimal, localcontext
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

import financeai
from financeai.core.clock import NS_PER_S, format_ns
from financeai.core.errors import InvalidConfigError, InvalidDataError, OptionalDependencyError
from financeai.core.hashing import sha256_file, sha256_json
from financeai.domains.market import metrics
from financeai.domains.market.engine import Orders, PreparedMarket, ReplayResult
from financeai.domains.market.ledger import Fills, Settlement, settle
from financeai.domains.market.scenario import RiskLimits, Scenario
from financeai.domains.market.stream import MarketData
from financeai.reporting.charts import (
    EVIDENCE_SCOPES,
    _array_record,
    _atomic_write,
    _fill_records,
    _instrument_records,
    _interval,
    _order_records,
    extrema_envelope,
)


@dataclass(frozen=True, slots=True)
class ReplayReportConfig:
    """Display settings; limits refuse oversized reports instead of silently removing events."""

    title: str = "Trading replay"
    theme: Literal["light", "dark"] = "dark"
    evidence_scope: str = "SYNTHETIC"
    display_buckets: int = 400
    max_events: int = 20_000
    max_points: int = 200_000
    instruments: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not self.title.strip():
            raise InvalidConfigError("report title must be a non-empty string")
        if self.theme not in ("light", "dark") or self.evidence_scope not in EVIDENCE_SCOPES:
            raise InvalidConfigError("unknown report theme or evidence scope")
        for name, lower, upper in (
            ("display_buckets", 1, 10_000),
            ("max_events", 1, 1_000_000),
            ("max_points", 20, 2_000_000),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise InvalidConfigError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.instruments is not None and (
            not isinstance(self.instruments, tuple)
            or not self.instruments
            or any(not isinstance(key, str) for key in self.instruments)
            or len(set(self.instruments)) != len(self.instruments)
        ):
            raise InvalidConfigError("instruments must be a non-empty tuple of distinct instrument keys")


@dataclass(frozen=True, slots=True)
class ReplayReportArtifact:
    html_path: Path
    manifest_path: Path
    sha256: str
    manifest: dict[str, Any]


def _json_exact(value: Any) -> Any:
    """Preserve integers beyond JavaScript's safe range in downloadable data."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value) if abs(value) > 2**53 - 1 else value
    if isinstance(value, dict):
        return {key: _json_exact(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_exact(item) for item in value]
    return value


def _series(
    times: ArrayLike, values: ArrayLike, start: int, end: int, buckets: int, *, before: float | int = 0
) -> dict[str, Any]:
    t = np.asarray(times, dtype=np.int64)
    y = np.asarray(values)
    if t.shape != y.shape or t.ndim != 1 or np.any(np.isinf(y)):
        raise InvalidDataError("report series must be aligned, finite or NaN, one-dimensional arrays")
    if t.size and (int(t[0]) < start or int(t[-1]) > end or np.any(t[1:] < t[:-1])):
        raise InvalidDataError("report series clocks must be ordered inside the report interval")
    integer = y.dtype.kind in "iu" and isinstance(before, int)
    dtype = np.int64 if integer else np.float64
    y = np.concatenate(
        (np.array([before], dtype=dtype), y.astype(dtype), np.array([y[-1] if y.size else before], dtype=dtype))
    )
    t = np.concatenate((np.array([start], np.int64), t, np.array([end], np.int64)))
    keep = extrema_envelope(t, y, start_ns=start, end_ns=end, n_buckets=buckets)
    exact = [str(int(y[k])) if integer else repr(float(y[k])) if np.isfinite(y[k]) else None for k in keep]
    return {
        "x": [(int(t[k]) - start) / NS_PER_S for k in keep],
        "y": [float(y[k]) if np.isfinite(y[k]) else None for k in keep],
        "ns": [str(int(t[k])) for k in keep],
        "utc": [format_ns(int(t[k]), precision="ns") for k in keep],
        "exact": exact,
        "points_full": len(t),
        "points_displayed": len(keep),
        "unavailable_points": int(np.count_nonzero(~np.isfinite(y))),
    }


def _currency_view(
    result: ReplayResult, prep: PreparedMarket, indices: list[int]
) -> tuple[ReplayResult, PreparedMarket]:
    """Select cash-convention-compatible instruments without converting or summing currencies."""
    mapping = {v: i for i, v in enumerate(indices)}
    tables = []
    for table in (result.fills, result.orders):
        mask = np.isin(table.venue, indices)
        columns = {f.name: np.ascontiguousarray(getattr(table, f.name)[mask]) for f in fields(table)}
        columns["venue"] = np.array([mapping[int(v)] for v in table.venue[mask]], np.int8)
        tables.append(type(table)(**columns))
    st = result.settlement
    values: dict[str, Any] = {
        f.name: tuple(getattr(st, f.name)[v] for v in indices)
        for f in fields(st)
        if f.name not in ("valid", "reason", "currencies")
    }
    values.update(
        valid=st.valid,
        reason=st.reason,
        currencies=tuple(str(prep.data.instruments[v].settlement_currency) for v in indices),
    )
    selected = ReplayResult(cast(Fills, tables[0]), cast(Orders, tables[1]), Settlement(**values), result.decisions)
    return selected, PreparedMarket.build(MarketData(tuple(prep.data.venues[v] for v in indices)))


def _event(
    kind: str,
    side: int,
    qty: int,
    price: float,
    submit: int,
    arrival: int,
    ack: int,
    order_id: int,
    start: int,
    *,
    cash: float | None = None,
    price_ticks: int,
    tick_size: Decimal,
) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = max(28, len(tick_size.as_tuple().digits) + len(str(abs(price_ticks))))
        price_exact = format(Decimal(price_ticks) * tick_size, "f")
    return {
        "kind": kind,
        "side": "BUY" if side > 0 else "SELL",
        "qty_units": str(qty),
        "price": price,
        "price_ticks": str(price_ticks),
        "price_exact": price_exact,
        "submit_ns": str(submit),
        "arrival_ns": str(arrival),
        "ack_ns": str(ack),
        "submit_utc": format_ns(submit, precision="ns"),
        "arrival_utc": format_ns(arrival, precision="ns"),
        "ack_utc": format_ns(ack, precision="ns"),
        "order_latency_ns": str(arrival - submit),
        "ack_latency_ns": str(ack - arrival),
        "order_id": str(order_id),
        "x": (arrival - start) / NS_PER_S,
        "cash": cash,
    }


def build_replay_report_data(
    result: ReplayResult,
    prep: PreparedMarket,
    *,
    scenario: Scenario | None = None,
    limits: RiskLimits | None = None,
    config: ReplayReportConfig | None = None,
) -> dict[str, Any]:
    """Build a strict-JSON-ready report using only NumPy; metrics use complete ledgers.

    Instrument selection filters price/position panels. Currency panels always describe the
    full account in that currency. Epoch nanoseconds and exact quantities are encoded as strings.
    """
    cfg = config or ReplayReportConfig()
    specs = prep.data.instruments
    n = len(specs)
    st = result.settlement
    for f in fields(st):
        if f.name not in ("valid", "reason", "currencies") and len(getattr(st, f.name)) != n:
            raise InvalidDataError("settlement does not match the prepared instrument count")
    if st.currencies and st.currencies != tuple(str(i.settlement_currency) for i in specs):
        raise InvalidDataError("settlement currencies differ from the instrument definitions")
    for table in (result.fills, result.orders):
        if len(table) and (int(table.venue.min()) < 0 or int(table.venue.max()) >= n):
            raise InvalidDataError("report record has an unknown instrument index")
    if not np.all(np.isfinite(result.fills.cash)) or not all(math.isfinite(v) for v in (*st.cash, *st.fees)):
        raise InvalidDataError("report cash and commissions must be finite")
    keys = [i.key for i in specs]
    chosen = cfg.instruments or tuple(keys)
    if set(chosen) - set(keys):
        raise InvalidConfigError("report selects unknown instruments")
    lim = limits or result.limits
    caps = lim.caps_for(specs) if lim else (None,) * n
    if scenario is not None:
        scenario.validate_instruments(specs)
    audited = settle(
        result.fills,
        specs,
        tuple(scenario.fee_for(i) for i in specs) if scenario else (Decimal(0),) * n,
        cap_units=tuple(c if c is not None else 2**62 - 1 for c in caps),
        fees_per_unit=tuple(scenario.fee_per_unit_for(i) for i in specs) if scenario else None,
    )
    for name in ("final_position_units", "max_abs_position_units", "traded_units", "n_fills"):
        if getattr(audited, name) != getattr(st, name):
            raise InvalidDataError("settlement records do not reconcile with fills", context={"field": name})
    for name in ("cash", "fees") if scenario else ("cash",):
        if not all(
            math.isclose(a, b, rel_tol=metrics.RECONCILE_REL_TOL, abs_tol=metrics.RECONCILE_ABS_TOL)
            for a, b in zip(getattr(audited, name), getattr(st, name), strict=True)
        ):
            raise InvalidDataError("settlement cash or fees do not reconcile with fills", context={"field": name})
    account_valid = st.valid and audited.valid
    invalid_reason = audited.reason or st.reason
    start, end = _interval(result, prep, None)
    assets = []
    event_count = 0
    point_count = 0
    for key in chosen:
        v = keys.index(key)
        spec, book = specs[v], prep.data.venues[v].book
        usable = prep.valid[v]
        bid = np.where(usable, prep.bpx[v][:, 0] * spec.tick_float, np.nan)
        ask = np.where(usable, prep.apx[v][:, 0] * spec.tick_float, np.nan)
        t, q = metrics.position_path(result, v)
        kt, kq = metrics.known_position_path(result, v)
        series = {
            "bid": _series(book.recv_ns, bid, start, end, cfg.display_buckets, before=np.nan),
            "ask": _series(book.recv_ns, ask, start, end, cfg.display_buckets, before=np.nan),
            "position": _series(t, q, start, end, cfg.display_buckets),
            "known_position": _series(kt, kq, start, end, cfg.display_buckets),
        }
        fills, orders = result.fills, result.orders
        fill_indices = np.flatnonzero(fills.venue == v)
        unfilled_indices = np.flatnonzero((orders.venue == v) & (orders.filled_units == 0))
        if event_count + len(fill_indices) + len(unfilled_indices) > cfg.max_events:
            raise InvalidConfigError("report exceeds max_events; select fewer instruments or increase max_events")
        events = [
            _event(
                "fill",
                int(fills.side[j]),
                int(fills.qty_units[j]),
                int(fills.px_ticks[j]) * spec.tick_float,
                int(fills.submit_ns[j]),
                int(fills.arrival_ns[j]),
                int(fills.ack_ns[j]),
                int(fills.order_id[j]),
                start,
                cash=float(fills.cash[j]),
                price_ticks=int(fills.px_ticks[j]),
                tick_size=spec.tick_size,
            )
            for j in fill_indices
        ]
        events.extend(
            _event(
                "unfilled",
                int(orders.side[j]),
                int(orders.qty_units[j]),
                int(orders.limit_ticks[j]) * spec.tick_float,
                int(orders.submit_ns[j]),
                int(orders.arrival_ns[j]),
                int(orders.ack_ns[j]),
                int(orders.order_id[j]),
                start,
                price_ticks=int(orders.limit_ticks[j]),
                tick_size=spec.tick_size,
            )
            for j in unfilled_indices
        )
        order_lookup = {int(orders.order_id[j]): j for j in np.flatnonzero(orders.venue == v)}
        for event in events:
            order = order_lookup.get(int(event["order_id"]))
            event["requested_units"] = str(int(orders.qty_units[order])) if order is not None else "not recorded"
            event["order_filled_units"] = str(int(orders.filled_units[order])) if order is not None else "not recorded"
            event["order_status"] = (
                "not recorded"
                if order is None
                else "unfilled"
                if orders.filled_units[order] == 0
                else "partial"
                if orders.filled_units[order] < orders.qty_units[order]
                else "filled"
            )
            if event["kind"] == "unfilled":
                event["qty_units"] = "0"
        events.sort(key=lambda row: (int(row["arrival_ns"]), int(row["order_id"])))
        event_count += len(events)
        point_count += sum(s["points_displayed"] for s in series.values())
        if event_count > cfg.max_events or point_count > cfg.max_points:
            raise InvalidConfigError(
                "report exceeds display limits; select fewer instruments or increase explicit limits"
            )
        assets.append(
            {
                "key": key,
                "instrument": spec.to_mapping(),
                "cap_units": str(caps[v]) if caps[v] is not None else None,
                "series": series,
                "events": events,
                "fills": int(np.count_nonzero(fills.venue == v)),
                "orders": int(np.count_nonzero(orders.venue == v)),
            }
        )
    accounts = []
    for currency in sorted({str(i.settlement_currency) for i in specs}):
        indices = [v for v, spec in enumerate(specs) if spec.settlement_currency == currency]
        selected, market = _currency_view(result, prep, indices)
        mt, marked = metrics.marked_value_path(selected, market)
        ct, cash = metrics.cash_path(selected)
        dd = metrics.drawdown(mt, marked)
        series = {
            "marked": _series(mt, marked, start, end, cfg.display_buckets),
            "cash": _series(ct, cash, start, end, cfg.display_buckets),
            "drawdown": _series(mt, metrics.drawdown_path(marked), start, end, cfg.display_buckets),
        }
        point_count += sum(s["points_displayed"] for s in series.values())
        if point_count > cfg.max_points:
            raise InvalidConfigError("report exceeds max_points; lower display_buckets or increase max_points")
        accounts.append(
            {
                "currency": currency,
                "instruments": [keys[v] for v in indices],
                "series": series,
                "net_pnl": math.fsum(st.cash[v] for v in indices) if account_valid else None,
                "commissions": math.fsum(st.fees[v] for v in indices),
                "max_drawdown": dd.max_pnl,
                "unmarked_points": dd.unmarked_points,
                "fills": sum(st.n_fills[v] for v in indices),
            }
        )
    sources = {
        "fills": _fill_records(result.fills),
        "orders": _order_records(result.orders),
        "instruments": _instrument_records(prep),
        "settlement": asdict(st),
        "risk_limits": lim.to_mapping() if lim else None,
        "scenario": scenario.to_mapping() if scenario else None,
        "mark_inputs": [
            _array_record(f"book.{v}.{name}", getattr(t.book, name))
            for v, t in enumerate(prep.data.venues)
            for name in ("recv_ns", "bid_px", "ask_px", "n_bid", "n_ask", "kind")
        ],
    }
    sources = _json_exact(sources)
    return cast(
        dict[str, Any],
        _json_exact(
            {
                "schema": "financeai.replay_report",
                "schema_version": 1,
                "title": cfg.title,
                "theme": cfg.theme,
                "evidence_scope": cfg.evidence_scope,
                "financeai_version": financeai.__version__,
                "start_ns": str(start),
                "end_ns": str(end),
                "start_utc": format_ns(start, precision="ns"),
                "end_utc": format_ns(end, precision="ns"),
                "account_valid": account_valid,
                "invalid_reason": invalid_reason,
                "assets": assets,
                "accounts": accounts,
                "scenario": scenario.to_mapping() if scenario else None,
                "risk_limits": lim.to_mapping() if lim else None,
                "counts": {
                    "fills": len(result.fills),
                    "orders": len(result.orders),
                    "displayed_events": event_count,
                    "displayed_points": point_count,
                },
                "reduction": {
                    "method": "first/last/min/max and gap boundaries",
                    "buckets": cfg.display_buckets,
                    "statistics": "full records before display reduction",
                },
                "sources": sources,
                "source_sha256": sha256_json(sources),
            }
        ),
    )


def write_replay_report(
    result: ReplayResult,
    prep: PreparedMarket,
    out_path: str | Path,
    *,
    scenario: Scenario | None = None,
    limits: RiskLimits | None = None,
    config: ReplayReportConfig | None = None,
    overwrite: bool = False,
) -> ReplayReportArtifact:
    """Write an offline HTML report and SHA-256 manifest; never imports Plotly at module import."""
    path = Path(out_path)
    manifest_path = path.with_name(path.name + ".manifest.json")
    if path.suffix.lower() != ".html":
        raise InvalidConfigError("interactive report output must have an .html extension")
    if not overwrite and (path.exists() or manifest_path.exists()):
        raise InvalidConfigError("report output already exists; pass overwrite=True to replace it")
    data = build_replay_report_data(result, prep, scenario=scenario, limits=limits, config=config)
    try:
        offline = importlib.import_module("plotly.offline")
        plotly = importlib.import_module("plotly")
    except ImportError as exc:
        raise OptionalDependencyError(
            "interactive reports require the 'interactive' extra", context={"extra": "interactive"}
        ) from exc
    template = files("financeai.reporting").joinpath("_replay_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    replacements = {"TITLE": html.escape(data["title"]), "DATA": payload, "PLOTLY": str(offline.get_plotlyjs())}
    document = re.sub(r"<!--FINANCEAI:(TITLE|DATA|PLOTLY)-->", lambda m: replacements[m[1]], template)

    def save_html(target: Path) -> None:
        target.write_text(document, encoding="utf-8")

    _atomic_write(path, save_html)
    digest = sha256_file(path)
    manifest = {
        "schema": "financeai.replay_report_manifest",
        "schema_version": 1,
        "output": {"file": path.name, "sha256": digest, "bytes": path.stat().st_size},
        "source_sha256": data["source_sha256"],
        "sources": data["sources"],
        "counts": data["counts"],
        "reduction": data["reduction"],
        "encoding": (
            "epoch nanoseconds, exact plotted values and unsafe JavaScript integers are strings; x is relative seconds"
        ),
        "renderer": {
            "financeai": financeai.__version__,
            "plotly": str(plotly.__version__),
            "python_sha256": sha256_file(__file__),
            "template_sha256": sha256_json(template),
            "metrics_sha256": sha256_file(metrics.__file__),
            "clock_sha256": sha256_file(format_ns.__code__.co_filename),
            "chart_helpers_sha256": sha256_file(extrema_envelope.__code__.co_filename),
        },
        "evidence_scope": data["evidence_scope"],
        "account_valid": data["account_valid"],
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"

    def save_manifest(target: Path) -> None:
        target.write_text(text, encoding="utf-8")

    _atomic_write(manifest_path, save_manifest)
    return ReplayReportArtifact(path, manifest_path, digest, manifest)


__all__ = ["ReplayReportArtifact", "ReplayReportConfig", "build_replay_report_data", "write_replay_report"]
