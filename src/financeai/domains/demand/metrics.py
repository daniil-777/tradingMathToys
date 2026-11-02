"""Demand forecast metrics over issued-and-scored tickets: MAE, RMSE, bias and pinball loss.

Metrics aggregate the one score of each ticket (see :mod:`financeai.online.scoring`). Unscored tickets are counted,
never zero-filled: with no scored ticket every metric is ``None``. Pinball loss is reported only over tickets that
carried marginal quantiles (quantile-capable adapters); for point-only adapters it is ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from financeai.core.errors import InvalidDataError
from financeai.online.scoring import ScoreRecord


@dataclass(frozen=True, slots=True)
class DemandMetrics:
    n_issued: int
    n_scored: int
    mae: float | None
    rmse: float | None
    bias: float | None
    """Mean of ``value - mean``: positive means demand was under-forecast."""
    pinball: float | None
    """Mean pinball loss over every issued quantile of every scored quantile ticket."""
    n_quantile_scored: int

    @property
    def n_unscored(self) -> int:
        return self.n_issued - self.n_scored

    def to_mapping(self) -> dict[str, Any]:
        return {**asdict(self), "n_unscored": self.n_unscored}


def summarize(scores: Iterable[ScoreRecord], *, n_issued: int) -> DemandMetrics:
    """Aggregate scored tickets (compensated sums); ``n_issued`` counts every issued ticket, scored or not."""
    records = list(scores)
    ids = [r.ticket_id for r in records]
    if len(set(ids)) != len(ids):
        raise InvalidDataError("a ticket appears more than once among the scores")
    if isinstance(n_issued, bool) or not isinstance(n_issued, int) or n_issued < len(records):
        raise InvalidDataError(
            "n_issued must count at least every scored ticket", context={"n_issued": n_issued, "scored": len(records)}
        )
    if not records:
        return DemandMetrics(
            n_issued=n_issued, n_scored=0, mae=None, rmse=None, bias=None, pinball=None, n_quantile_scored=0
        )
    n = len(records)
    losses = [loss for r in records for loss in r.pinball]
    quantile_tickets = sum(1 for r in records if r.pinball)
    return DemandMetrics(
        n_issued=n_issued,
        n_scored=n,
        mae=math.fsum(r.abs_error for r in records) / n,
        rmse=math.sqrt(math.fsum(r.squared_error for r in records) / n),
        bias=math.fsum(r.error for r in records) / n,
        pinball=math.fsum(losses) / len(losses) if losses else None,
        n_quantile_scored=quantile_tickets,
    )
