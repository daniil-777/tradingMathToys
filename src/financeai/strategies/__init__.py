"""Causal trading signals, order-flow imbalance, regime gates, and cost-aware linear forecasts."""

from __future__ import annotations

from financeai.strategies.flow import ofi_per_update, window_sum_per_event
from financeai.strategies.leadlag import (
    build_signal,
    composite_branch,
    delayed,
    depth_weighted_mid,
    event_values,
    ewma_deviation_signal,
    follower_spread_ticks,
    lead_gap_signal,
    observed_rows,
    ofi_signal,
    quote_confirm_signal,
    sweep_signal,
    window_change_bps,
)
from financeai.strategies.linear import (
    DEFAULT_FLOW_SCALE,
    decayed_trade_flow,
    depth_imbalance,
    design_matrix,
    feature_columns,
    follower_labels,
    prediction_signal,
    training_rows,
)
from financeai.strategies.regime import apply_gate, decayed_volume, freshness_mask, normalise

__all__ = [
    "DEFAULT_FLOW_SCALE",
    "apply_gate",
    "build_signal",
    "composite_branch",
    "decayed_trade_flow",
    "decayed_volume",
    "delayed",
    "depth_imbalance",
    "depth_weighted_mid",
    "design_matrix",
    "event_values",
    "ewma_deviation_signal",
    "feature_columns",
    "follower_labels",
    "follower_spread_ticks",
    "freshness_mask",
    "lead_gap_signal",
    "normalise",
    "observed_rows",
    "ofi_per_update",
    "ofi_signal",
    "prediction_signal",
    "quote_confirm_signal",
    "sweep_signal",
    "training_rows",
    "window_change_bps",
    "window_sum_per_event",
]
