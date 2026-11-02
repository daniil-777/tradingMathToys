"""The campaign's candidate export must round-trip every policy field into a StrategyConfig whose
identity equals the grid row's config_id (a dropped field would freeze the wrong policy)."""

from __future__ import annotations

import pytest

from financeai.application.grid import GridExperiment
from financeai.application.strategy import StrategyConfig
from financeai.domains.market.engine import ThresholdParams


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"reverse": True},
        {"abs_cap_ns": 15 * 10**9},
        {"early_check_ns": 4 * 10**9, "early_markout_bps": -1.0},
        {"max_clips": 3, "clip_spacing_ns": 2 * 10**9, "refresh_hold": True, "size_units_cap": 2, "depth_ticks": 2},
    ],
)
def test_export_preserves_the_configuration_identity(extra: dict[str, object]) -> None:
    params = {
        "allow_short": True,
        "clip_spacing_ns": 0,
        "cooldown_ns": 0,
        "depth_ticks": -1,
        "direction": [1, 0],
        "enter": 1.0,
        "entry_slip_ticks": 0,
        "exit": -1e9,
        "exit_slip_ticks": 10,
        "max_clips": 1,
        "max_hold_ns": 10 * 10**9,
        "min_hold_ns": 0,
        "refresh_hold": True,
        "size_units_cap": 1,
        "size_units": [1_000_000, 0],
        "stop_bps": 0.0,
        "reverse": False,
        "abs_cap_ns": 0,
        "early_check_ns": 0,
        "early_markout_bps": 0.0,
        **extra,
    }
    signal = {"kind": "sweep", "source": "venue_a/ALPHA-USD", "min_units": 100_000}
    cfg = StrategyConfig.from_threshold_params(
        name="example",
        instruments=("venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"),
        signal=signal,
        params=mod_params(params),
        session={"warmup_s": 0, "entry_stop_s": 3, "flatten_s": 2, "exec_stop_s": 1},
    )
    expected = GridExperiment("T", signal, mod_params(params), 5).config_id
    assert cfg.config_id == expected
    assert GridExperiment("T", cfg.signal, cfg.threshold_params(), 5).config_id == expected


def mod_params(p: dict[str, object]) -> ThresholdParams:
    q = dict(p)
    q["direction"] = tuple(q["direction"])  # type: ignore[arg-type]
    q["size_units"] = tuple(q["size_units"])  # type: ignore[arg-type]
    return ThresholdParams(**q)  # type: ignore[arg-type]
