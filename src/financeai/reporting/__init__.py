"""Deterministic ledger-derived charts and manifest generation (Matplotlib loaded lazily)."""

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

__all__ = [
    "EVIDENCE_SCOPES",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "FigureArtifact",
    "decision_trace_figure",
    "extrema_envelope",
    "position_pnl_figure",
]
