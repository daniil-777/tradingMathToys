"""Shared input contracts for the compiled and independent reference replays.

Validation belongs at the numerical boundary, so Python, CLI, and service callers
receive the same guarantees. The two engines retain separate execution algorithms.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError
from financeai.domains.market.scenario import MAX_REPLAY_NS, Scenario, SessionPlan


def validate_clock_headroom(
    last_event_ns: int, scenario: Scenario, session: SessionPlan, *, cooldown_ns: int = 0
) -> None:
    """Leave int64 headroom above event, acknowledgement, and session-timer decisions.

    The drain can extend beyond the last market event. A decision can calculate an
    arrival even when it will subsequently refuse that order at the execution cutoff.
    Reserve int64 max for the engine's no-event sentinel as well as avoiding overflow.
    All arithmetic here uses Python ints, before values reach a compiled kernel.
    """
    defaults = (scenario.order_latency_ns, scenario.ack_ns)
    delays = [defaults, *((v.order_latency_ns, v.ack_ns) for v in scenario.latency_by_instrument.values())]
    horizon = max(order + max(ack, scenario.consumption_ttl_ns) for order, ack in delays) + cooldown_ns
    latest = max(last_event_ns, session.end_ns)
    if latest > MAX_REPLAY_NS - horizon:
        raise InvalidDataError(
            "replay clocks leave no int64 room for derived deadlines",
            context={"last_event_ns": str(last_event_ns), "end_ns": str(session.end_ns), "horizon_ns": horizon},
        )


def prepare_signal(values: NDArray[np.float64], events: int, *, name: str) -> NDArray[np.float64]:
    """Require aligned numerical signals; NaN means unavailable, infinity is invalid."""
    try:
        out = np.ascontiguousarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidConfigError(f"{name} must be a numerical array") from exc
    if out.shape != (events,):
        raise InvalidConfigError(
            f"{name} must have one value per stream event", context={"events": events, "got": out.shape}
        )
    if np.any(np.isinf(out)):
        raise InvalidConfigError(f"{name} cannot contain infinity; use NaN for unavailable values")
    return out
