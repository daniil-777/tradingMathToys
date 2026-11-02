"""Cross-venue signal confirmation, causal entry diagnostics, and exact cash audits.

Instrument keys are supplied by callers. Volume thresholds are expressed in each
instrument's stored quantity units and must be calibrated on training data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from fractions import Fraction
from typing import Any

import numpy as np

from financeai.application.grid import concentration, episode_nets, with_fees
from financeai.application.strategy import StrategyConfig
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.instruments import InstrumentSpec
from financeai.domains.market.ledger import Fills
from financeai.strategies.regime import decayed_volume

NS_PER_MS: int = 1_000_000


@dataclass(frozen=True)
class CrossVenueConfirmationSpec:
    """Parameters for a cross-venue confirmed sweep + lead-gap signal recipe."""

    name: str
    sweep_min_units: float = 1_000.0
    sweep_venue_a_tau_ms: int = 1_000
    sweep_venue_a_min_units: float = 0.0
    sweep_venue_b_tau_ms: int = 1_000
    sweep_venue_b_min_units: float = 2_000.0
    lead_window_ms: int = 100
    lead_depth_units: float = 1_000.0
    lead_gate_ticks: int = 1
    lead_scale: float = 2.5
    lead_venue_a_tau_ms: int = 1_000
    lead_venue_a_min_units: float = 61_000.0
    lead_venue_b_tau_ms: int = 1_000
    lead_venue_b_min_units: float = 1_000.0
    fresh_max_age_ms: int = 500
    exit_slip_ticks: int = 10

    def __post_init__(self) -> None:
        if self.sweep_min_units <= 0:
            raise ValueError("sweep_min_units must be positive")
        if self.sweep_venue_a_min_units < 0 or self.sweep_venue_b_min_units < 0:
            raise ValueError("sweep gate thresholds must be non-negative")
        if self.lead_venue_a_min_units < 0 or self.lead_venue_b_min_units < 0:
            raise ValueError("lead_gap gate thresholds must be non-negative")
        if self.lead_scale <= 0:
            raise ValueError("lead_scale must be positive")
        if self.fresh_max_age_ms <= 0:
            raise ValueError("fresh_max_age_ms must be positive")
        if self.exit_slip_ticks < 0:
            raise ValueError("exit_slip_ticks must be non-negative")


@dataclass(frozen=True)
class RationalInterval:
    """Exact rational settlement value and outward-rounded 18-decimal-place enclosure."""

    exact: Fraction
    lower_str: str
    upper_str: str

    @property
    def width_str(self) -> str:
        lo = Decimal(self.lower_str)
        hi = Decimal(self.upper_str)
        return str(hi - lo)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": str(self.exact.numerator),
            "denominator": str(self.exact.denominator),
            "lower_pnl": self.lower_str,
            "upper_pnl": self.upper_str,
            "rounding_width_pnl": self.width_str,
        }


@dataclass(frozen=True)
class EpisodeCrossVenueDiagnostic:
    """Per-episode net settlement cash paired with entry-time decayed volume on both venues."""

    episode_index: int
    submit_ns: int
    net_pnl: float
    venue_a_decayed_vol_units: float
    venue_b_decayed_vol_units: float


def _gate_leaf(
    base_leaf: dict[str, Any],
    *,
    source: str,
    leader: str,
    venue_a_tau_ms: int,
    venue_a_min_units: float,
    venue_b_tau_ms: int,
    venue_b_min_units: float,
) -> dict[str, Any]:
    """Attach single-venue or dual-venue (``kind='all'``) volume gates using existing grammar."""
    gates: list[dict[str, Any]] = []
    if venue_a_min_units > 0:
        gates.append(
            {
                "venue": source,
                "tau_ms": int(venue_a_tau_ms),
                "min_units": int(venue_a_min_units)
                if float(venue_a_min_units).is_integer()
                else float(venue_a_min_units),
            }
        )
    if venue_b_min_units > 0:
        gates.append(
            {
                "venue": leader,
                "tau_ms": int(venue_b_tau_ms),
                "min_units": int(venue_b_min_units)
                if float(venue_b_min_units).is_integer()
                else float(venue_b_min_units),
            }
        )
    if not gates:
        return dict(base_leaf)
    if len(gates) == 1:
        leaf = dict(base_leaf)
        leaf["gate"] = gates[0]
        return leaf
    # Dual-venue conjunction via existing ``kind: "all"`` composite (sign-agreement + min(|c|))
    components = []
    for g in gates:
        leaf = dict(base_leaf)
        leaf["gate"] = g
        components.append({"scale": 1.0, "spec": leaf})
    return {"kind": "all", "components": components}


def build_cross_venue_signal_spec(
    spec: CrossVenueConfirmationSpec, *, source: str = "venue_a/ALPHA-USD", leader: str = "venue_b/ALPHA-PERP"
) -> dict[str, Any]:
    """Build a JSON-serializable signal recipe compatible with the frozen ``leadlag`` evaluator."""
    fresh = {"max_age_ms": int(spec.fresh_max_age_ms)}
    sw_base: dict[str, Any] = {
        "kind": "sweep",
        "source": source,
        "min_units": int(spec.sweep_min_units)
        if float(spec.sweep_min_units).is_integer()
        else float(spec.sweep_min_units),
        "fresh": fresh,
    }
    lg_base: dict[str, Any] = {
        "kind": "lead_gap",
        "leader": leader,
        "follower": source,
        "window_ms": int(spec.lead_window_ms),
        "depth_units": int(spec.lead_depth_units)
        if float(spec.lead_depth_units).is_integer()
        else float(spec.lead_depth_units),
        "gate_ticks": int(spec.lead_gate_ticks),
        "fresh": fresh,
    }
    comp_sw = _gate_leaf(
        sw_base,
        source=source,
        leader=leader,
        venue_a_tau_ms=spec.sweep_venue_a_tau_ms,
        venue_a_min_units=spec.sweep_venue_a_min_units,
        venue_b_tau_ms=spec.sweep_venue_b_tau_ms,
        venue_b_min_units=spec.sweep_venue_b_min_units,
    )
    comp_lg = _gate_leaf(
        lg_base,
        source=source,
        leader=leader,
        venue_a_tau_ms=spec.lead_venue_a_tau_ms,
        venue_a_min_units=spec.lead_venue_a_min_units,
        venue_b_tau_ms=spec.lead_venue_b_tau_ms,
        venue_b_min_units=spec.lead_venue_b_min_units,
    )
    return {
        "kind": "any",
        "components": [
            {"scale": 1.0, "spec": comp_sw},
            {"scale": float(spec.lead_scale), "spec": comp_lg},
        ],
    }


def build_cross_venue_strategy_config(
    spec: CrossVenueConfirmationSpec,
    base_config: StrategyConfig,
) -> StrategyConfig:
    """Create a canonical ``StrategyConfig`` from a ``CrossVenueConfirmationSpec``."""
    if len(base_config.instruments) != 2:
        raise ValueError("cross-venue confirmation requires exactly two instrument keys")
    sig_spec = build_cross_venue_signal_spec(spec, source=base_config.instruments[0], leader=base_config.instruments[1])
    policy = dict(base_config.policy)
    policy["exit_slip_ticks"] = int(spec.exit_slip_ticks)
    return replace(
        base_config,
        name=spec.name,
        signal=sig_spec,
        policy=policy,
    )


def rational_interval(value: Fraction, places: int = 18) -> RationalInterval:
    """Compute an exact outward-rounded decimal enclosure ``[lower_str, upper_str]`` of ``value``."""
    if places < 1:
        raise ValueError("places must be >= 1")
    scale = 10**places
    num = value.numerator * scale
    den = value.denominator
    q_floor = num // den
    q_ceil = -((-num) // den)
    quant = Decimal(10) ** (-places)
    lo = (Decimal(q_floor) * quant).quantize(quant, rounding=ROUND_FLOOR)
    hi = (Decimal(q_ceil) * quant).quantize(quant, rounding=ROUND_CEILING)
    return RationalInterval(exact=value, lower_str=f"{lo:.{places}f}", upper_str=f"{hi:.{places}f}")


def certify_fill_ledger_interval(
    fills: Fills,
    instruments: Sequence[InstrumentSpec],
    fees: Sequence[Decimal],
    cap_units: int = 1_000_000,
    places: int = 18,
    fees_per_unit: Sequence[Decimal] | None = None,
) -> RationalInterval:
    """Independently audit a fill ledger in exact rational arithmetic and return its settlement-currency enclosure."""
    if cap_units <= 0:
        raise ValueError("cap_units must be positive")
    n_venues = len(instruments)
    fixed = tuple(fees_per_unit) if fees_per_unit is not None else (Decimal(0),) * n_venues
    if len(fees) != n_venues or len(fixed) != n_venues:
        raise ValueError("fees must match the instruments")
    if len({i.settlement_currency for i in instruments}) > 1:
        raise ValueError("a scalar cash certificate requires one settlement currency")
    positions = [0] * n_venues
    ticks = [Fraction(str(inst.tick_size)) for inst in instruments]
    lots = [int(inst.lot_size) for inst in instruments]
    fee_fracs = [Fraction(str(f)) for f in fees]

    total_cash = Fraction(0)
    for venue_raw, side_raw, qty_raw, px_raw in zip(
        fills.venue.tolist(),
        fills.side.tolist(),
        fills.qty_units.tolist(),
        fills.px_ticks.tolist(),
        strict=True,
    ):
        v = int(venue_raw)
        side = int(side_raw)
        qty = int(qty_raw)
        px = int(px_raw)
        if v < 0 or v >= n_venues:
            raise ValueError(f"invalid venue index {v}")
        if side not in (-1, 1):
            raise ValueError(f"invalid side {side}")
        if qty <= 0 or qty % lots[v] != 0:
            raise ValueError(f"fill quantity {qty} not a positive multiple of lot {lots[v]}")
        if px <= 0:
            raise ValueError(f"non-positive fill price ticks {px}")
        positions[v] += side * qty
        if abs(positions[v]) > cap_units:
            raise ValueError(f"position {positions[v]} exceeds cap {cap_units} on venue {v}")
        inst = instruments[v]
        if positions[v] < 0 and not inst.shortable:
            raise ValueError("short inventory is not allowed for this instrument")
        exposure = qty * Fraction(inst.quantity_step) * Fraction(inst.contract_multiplier)
        price = ticks[v] * px
        total_cash += (
            (side - fee_fracs[v]) * exposure / price if inst.inverse else (-side - fee_fracs[v]) * exposure * price
        )
        total_cash -= qty * Fraction(inst.quantity_step) * Fraction(fixed[v])

    if any(p != 0 for p in positions):
        raise ValueError(f"non-flat terminal positions: {positions}")
    return rational_interval(total_cash, places=places)


def diagnose_episode_confirmation(
    prep: PreparedMarket,
    fills: Fills,
    instruments: Sequence[InstrumentSpec],
    fees: Sequence[Decimal],
    *,
    tau_ns: int = 10_000_000_000,
    fees_per_unit: Sequence[Decimal] | None = None,
) -> tuple[list[EpisodeCrossVenueDiagnostic], dict[str, Any]]:
    """Pair each round-trip episode with both venues' decayed volume at episode entry."""
    if len(instruments) != 2 or tuple(instruments) != prep.data.instruments:
        raise ValueError("confirmation diagnostics require the prepared market's two instruments")
    if len({spec.settlement_currency for spec in instruments}) != 1:
        raise ValueError("episode comparisons require a common settlement currency")
    rescored = with_fees(fills, instruments, fees, fees_per_unit)
    eps = episode_nets(rescored, len(instruments))
    vb = decayed_volume(prep, venue=0, tau_ns=tau_ns)
    vd = decayed_volume(prep, venue=1, tau_ns=tau_ns)
    recv_ns = prep.stream.recv_ns

    diagnostics: list[EpisodeCrossVenueDiagnostic] = []
    for idx, (sub_ns, net_pnl) in enumerate(eps):
        ev_idx = int(np.searchsorted(recv_ns, sub_ns, side="right") - 1)
        ev_idx = max(0, min(ev_idx, len(recv_ns) - 1))
        diagnostics.append(
            EpisodeCrossVenueDiagnostic(
                episode_index=idx,
                submit_ns=int(sub_ns),
                net_pnl=float(net_pnl),
                venue_a_decayed_vol_units=float(vb[ev_idx]),
                venue_b_decayed_vol_units=float(vd[ev_idx]),
            )
        )
    conc = concentration([d.net_pnl for d in diagnostics])
    return diagnostics, conc


def example_single_confirmation() -> CrossVenueConfirmationSpec:
    """Illustrative single-source confirmation; thresholds are not calibrated market estimates."""
    return CrossVenueConfirmationSpec(name="single-confirmation")


def example_dual_confirmation() -> CrossVenueConfirmationSpec:
    """Illustrative conjunction requiring activity on both instruments."""
    return CrossVenueConfirmationSpec(name="dual-confirmation", sweep_venue_a_min_units=1500.0)


def example_asymmetric_confirmation() -> CrossVenueConfirmationSpec:
    """Illustrative asymmetric confirmation for the sweep and lead-gap components."""
    return CrossVenueConfirmationSpec(name="asymmetric-confirmation", lead_venue_b_min_units=0.0)


__all__ = [
    "CrossVenueConfirmationSpec",
    "EpisodeCrossVenueDiagnostic",
    "RationalInterval",
    "build_cross_venue_signal_spec",
    "build_cross_venue_strategy_config",
    "certify_fill_ledger_interval",
    "diagnose_episode_confirmation",
    "example_asymmetric_confirmation",
    "example_dual_confirmation",
    "example_single_confirmation",
    "rational_interval",
]
