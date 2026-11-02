"""Application use cases: strategy configuration, single-policy replay, parallel grid evaluation,
and registered selection rules."""

from __future__ import annotations

from financeai.application.grid import (
    GridExperiment,
    Window,
    attribute_episodes,
    block_windows,
    concentration,
    day_blocks,
    episode_nets,
    evaluate_experiment,
    rescore,
    run_grid,
    with_fees,
)
from financeai.application.selection import Candidate, SelectionRule, select_configs
from financeai.application.strategy import (
    SCHEMA,
    SCHEMA_VERSION,
    StrategyConfig,
    build_strategy_signal,
    load_market,
    model_from_json,
    model_to_json,
    policy_config_id,
    run_strategy,
    summarize,
    write_run_outputs,
)

__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "Candidate",
    "GridExperiment",
    "SelectionRule",
    "StrategyConfig",
    "Window",
    "attribute_episodes",
    "block_windows",
    "build_strategy_signal",
    "concentration",
    "day_blocks",
    "episode_nets",
    "evaluate_experiment",
    "load_market",
    "model_from_json",
    "model_to_json",
    "policy_config_id",
    "rescore",
    "run_grid",
    "run_strategy",
    "select_configs",
    "summarize",
    "with_fees",
    "write_run_outputs",
]
