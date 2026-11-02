"""Per-ticket scores computed once, from the prediction actually issued and its released outcome.

A :class:`ScoreRecord` is domain neutral: signed/absolute/squared point error and, for tickets that carried marginal
quantiles, the pinball loss at every issued level. Domains aggregate these records into their own metrics; nothing
here recomputes a forecast or fills a missing outcome with zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from financeai.core.errors import InvalidDataError
from financeai.core.forecast import ForecastTicket, Outcome, decode_ns, encode_ns, require_keys


def pinball_loss(value: float, quantile: float, level: float) -> float:
    """Quantile (pinball) loss ``max(level * d, (level - 1) * d)`` with ``d = value - quantile``."""
    d = value - quantile
    return max(level * d, (level - 1.0) * d)


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """The one score of one ticket (``error = value - mean``)."""

    ticket_id: str
    stream: str
    outcome_version: int
    issued_at_ns: int
    available_at_ns: int
    released_at_ns: int
    value: float
    mean: float
    error: float
    abs_error: float
    squared_error: float
    quantile_levels: tuple[float, ...] = ()
    pinball: tuple[float, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "stream": self.stream,
            "outcome_version": self.outcome_version,
            "issued_at_ns": encode_ns(self.issued_at_ns),
            "available_at_ns": encode_ns(self.available_at_ns),
            "released_at_ns": encode_ns(self.released_at_ns),
            "value": self.value,
            "mean": self.mean,
            "error": self.error,
            "abs_error": self.abs_error,
            "squared_error": self.squared_error,
            "quantile_levels": list(self.quantile_levels),
            "pinball": list(self.pinball),
        }

    @classmethod
    def from_mapping(cls, doc: object) -> ScoreRecord:
        d = require_keys(
            doc,
            (
                "ticket_id",
                "stream",
                "outcome_version",
                "issued_at_ns",
                "available_at_ns",
                "released_at_ns",
                "value",
                "mean",
                "error",
                "abs_error",
                "squared_error",
                "quantile_levels",
                "pinball",
            ),
            where="score",
        )
        if not isinstance(d["quantile_levels"], list) or not isinstance(d["pinball"], list):
            raise InvalidDataError("score quantile levels and pinball losses must be lists")
        reals = [d[k] for k in ("value", "mean", "error", "abs_error", "squared_error")]
        reals += d["quantile_levels"] + d["pinball"]
        if not all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in reals):
            raise InvalidDataError("score values must be finite numbers", context={"ticket_id": d["ticket_id"]})
        version = d["outcome_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise InvalidDataError("outcome_version must be a non-negative integer")
        return cls(
            ticket_id=str(d["ticket_id"]),
            stream=str(d["stream"]),
            outcome_version=version,
            issued_at_ns=decode_ns(d["issued_at_ns"]),
            available_at_ns=decode_ns(d["available_at_ns"]),
            released_at_ns=decode_ns(d["released_at_ns"]),
            value=float(d["value"]),
            mean=float(d["mean"]),
            error=float(d["error"]),
            abs_error=float(d["abs_error"]),
            squared_error=float(d["squared_error"]),
            quantile_levels=tuple(float(v) for v in d["quantile_levels"]),
            pinball=tuple(float(v) for v in d["pinball"]),
        )


def score_ticket(ticket: ForecastTicket, outcome: Outcome, *, released_at_ns: int) -> ScoreRecord:
    """Score the issued ``ticket`` against its ``outcome`` released at ``released_at_ns``."""
    if outcome.ticket_id != ticket.ticket_id:
        raise InvalidDataError(
            "outcome belongs to another ticket",
            context={"ticket_id": ticket.ticket_id, "outcome_ticket_id": outcome.ticket_id},
        )
    if released_at_ns < outcome.available_at_ns:
        raise InvalidDataError(
            "an outcome cannot be scored before it is available",
            context={"ticket_id": ticket.ticket_id, "released_at_ns": released_at_ns},
        )
    error = outcome.value - ticket.mean
    pinball: tuple[float, ...] = ()
    if ticket.quantiles is not None:
        pinball = tuple(
            pinball_loss(outcome.value, q, level)
            for q, level in zip(ticket.quantiles.tolist(), ticket.quantile_levels, strict=True)
        )
    return ScoreRecord(
        ticket_id=ticket.ticket_id,
        stream=ticket.stream,
        outcome_version=outcome.version,
        issued_at_ns=ticket.issued_at_ns,
        available_at_ns=outcome.available_at_ns,
        released_at_ns=released_at_ns,
        value=outcome.value,
        mean=ticket.mean,
        error=error,
        abs_error=abs(error),
        squared_error=error * error,
        quantile_levels=ticket.quantile_levels,
        pinball=pinball,
    )
