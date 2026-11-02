"""Ledger-derived static/interactive reports; rendering dependencies are loaded lazily."""

from __future__ import annotations

from financeai.reporting.charts import (
    EVIDENCE_SCOPES,
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    FigureArtifact,
    decision_trace_figure,
    extrema_envelope,
    position_pnl_figure,
)
from financeai.reporting.replay import (
    ReplayReportArtifact,
    ReplayReportConfig,
    build_replay_report_data,
    write_replay_report,
)

__all__ = [
    "EVIDENCE_SCOPES",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "FigureArtifact",
    "ReplayReportArtifact",
    "ReplayReportConfig",
    "build_replay_report_data",
    "decision_trace_figure",
    "extrema_envelope",
    "position_pnl_figure",
    "write_replay_report",
]
