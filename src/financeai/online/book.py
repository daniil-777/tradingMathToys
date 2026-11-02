"""TicketBook: the single owner of issued tickets, scheduled outcomes and the score-once ledger.

* **Maturity.** An outcome may be scheduled as soon as it is known to the environment, but it stays invisible until
  the book clock reaches its ``available_at_ns``; it can never claim availability earlier than its target's
  availability rule (``feature cutoff + horizon + minimum delay``) or the ticket's issue time.
* **Cohorts.** ``release(t)`` delivers every outcome available at ``t`` in ``(available_at_ns, ticket_id)`` order,
  independent of submission order. Afterwards the clock is ``t``: feedback at or before ``t`` is rejected, so an
  outcome created inside a released cohort cannot train a peer decision of that cohort.
* **Score once.** Each ticket is scored exactly once, when its outcome is released. Re-delivering the identical
  outcome (same ticket and version) is an idempotent no-op; a conflicting payload or a revised version is rejected
  because issued scores are final. Tickets whose outcome never arrives stay unscored, never zero-filled.
* **State.** ``get_state``/``to_json`` produce strict, schema-versioned JSON (clocks as decimal strings). Loading
  replays the recorded releases and re-derives every score, so a tampered or inconsistent checkpoint is rejected.
  Pending outcomes are world truth: a checkpoint holds them, the forecasting side never reads them.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Final

from financeai.core.errors import InvalidDataError
from financeai.core.forecast import (
    INT64_MAX,
    ForecastTicket,
    Outcome,
    decode_ns,
    encode_ns,
    require_keys,
    require_schema,
)
from financeai.core.serialization import canonical_json, loads_strict
from financeai.online.scoring import ScoreRecord, score_ticket

BOOK_SCHEMA: Final = "financeai.online.ticket_book"
BOOK_SCHEMA_VERSION: Final = "1.0"
_STATE_KEYS: Final = ("schema", "schema_version", "clock_ns", "tickets", "outcomes", "scores")


@dataclass(frozen=True, slots=True)
class Released:
    """One matured outcome, the issued ticket it resolves and the ticket's one score."""

    ticket: ForecastTicket
    outcome: Outcome
    score: ScoreRecord


def _clock(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= INT64_MAX:
        raise InvalidDataError("release clock must be a non-negative int64 nanosecond count", context={"got": value})
    return value


class TicketBook:
    """Issued tickets, pending/released outcomes and scores of one owner (not thread-safe by design)."""

    def __init__(self) -> None:
        self._tickets: dict[str, ForecastTicket] = {}
        self._outcomes: dict[str, Outcome] = {}
        self._pending: list[tuple[int, str]] = []
        self._scores: dict[str, ScoreRecord] = {}
        self._clock: int | None = None

    @property
    def clock_ns(self) -> int | None:
        """Last released boundary (``None`` before the first release)."""
        return self._clock

    @property
    def tickets(self) -> tuple[ForecastTicket, ...]:
        """Issued tickets in issue order."""
        return tuple(self._tickets.values())

    @property
    def scores(self) -> tuple[ScoreRecord, ...]:
        """Scores in release order."""
        return tuple(self._scores.values())

    @property
    def n_pending(self) -> int:
        """Number of scheduled outcomes not yet available (their values are not exposed)."""
        return len(self._pending)

    def ticket(self, ticket_id: str) -> ForecastTicket:
        try:
            return self._tickets[ticket_id]
        except KeyError:
            raise InvalidDataError("unknown ticket", context={"ticket_id": ticket_id}) from None

    def score(self, ticket_id: str) -> ScoreRecord | None:
        return self._scores.get(ticket_id)

    def unscored_ids(self) -> tuple[str, ...]:
        """Issued tickets without a released outcome (unresolved: missing, not zero)."""
        return tuple(tid for tid in self._tickets if tid not in self._scores)

    def issue(self, ticket: ForecastTicket) -> bool:
        """Store an issued ticket; ``False`` when the identical ticket was already stored."""
        if not isinstance(ticket, ForecastTicket):
            raise InvalidDataError("only ForecastTicket objects can be issued")
        existing = self._tickets.get(ticket.ticket_id)
        if existing is not None:
            if existing == ticket:
                return False
            raise InvalidDataError(
                "ticket ID reused for a different forecast; a recomputed forecast needs a new ID",
                context={"ticket_id": ticket.ticket_id},
            )
        if self._clock is not None and ticket.issued_at_ns < self._clock:
            raise InvalidDataError(
                "ticket issued before the book clock", context={"ticket_id": ticket.ticket_id, "clock_ns": self._clock}
            )
        self._tickets[ticket.ticket_id] = ticket
        return True

    def submit(self, outcome: Outcome) -> bool:
        """Schedule an outcome; ``False`` for an identical re-delivery (already scheduled or scored)."""
        if not isinstance(outcome, Outcome):
            raise InvalidDataError("only Outcome objects can be submitted")
        ticket = self._tickets.get(outcome.ticket_id)
        if ticket is None:
            raise InvalidDataError("outcome for an unknown ticket", context={"ticket_id": outcome.ticket_id})
        existing = self._outcomes.get(outcome.ticket_id)
        if existing is not None:
            if existing == outcome:
                return False
            if existing.version == outcome.version:
                raise InvalidDataError(
                    "conflicting outcome for the same ticket and version",
                    context={"ticket_id": outcome.ticket_id, "version": outcome.version},
                )
            raise InvalidDataError(
                "ticket already has an outcome of another version; issued scores are final",
                context={"ticket_id": outcome.ticket_id, "versions": (existing.version, outcome.version)},
            )
        if outcome.available_at_ns < ticket.earliest_available_ns:
            raise InvalidDataError(
                "outcome would be available earlier than the target's availability rule allows",
                context={
                    "ticket_id": outcome.ticket_id,
                    "earliest_ns": ticket.earliest_available_ns,
                    "available_at_ns": outcome.available_at_ns,
                },
            )
        if self._clock is not None and outcome.available_at_ns <= self._clock:
            raise InvalidDataError(
                "outcome is at or before the released boundary; schedule feedback before its clock is released",
                context={"ticket_id": outcome.ticket_id, "clock_ns": self._clock},
            )
        self._outcomes[outcome.ticket_id] = outcome
        heapq.heappush(self._pending, (outcome.available_at_ns, outcome.ticket_id))
        return True

    def release(self, now_ns: int) -> tuple[Released, ...]:
        """Advance the clock to ``now_ns`` and score every outcome that became available (deterministic order)."""
        now = _clock(now_ns)
        if self._clock is not None and now < self._clock:
            raise InvalidDataError(
                "release clock must be non-decreasing", context={"clock_ns": self._clock, "now_ns": now}
            )
        self._clock = now
        released: list[Released] = []
        while self._pending and self._pending[0][0] <= now:
            _, ticket_id = heapq.heappop(self._pending)
            ticket, outcome = self._tickets[ticket_id], self._outcomes[ticket_id]
            record = score_ticket(ticket, outcome, released_at_ns=now)
            self._scores[ticket_id] = record
            released.append(Released(ticket, outcome, record))
        return tuple(released)

    def get_state(self) -> dict[str, Any]:
        return {
            "schema": BOOK_SCHEMA,
            "schema_version": BOOK_SCHEMA_VERSION,
            "clock_ns": None if self._clock is None else encode_ns(self._clock),
            "tickets": [t.to_mapping() for t in self._tickets.values()],
            "outcomes": [o.to_mapping() for o in self._outcomes.values()],
            "scores": [s.to_mapping() for s in self._scores.values()],
        }

    def to_json(self) -> str:
        return canonical_json(self.get_state())

    @classmethod
    def from_state(cls, doc: object) -> TicketBook:
        """Rebuild a book by replaying its recorded releases; every score must re-derive exactly."""
        d = require_keys(
            require_schema(doc, schema=BOOK_SCHEMA, version=BOOK_SCHEMA_VERSION, where="ticket book"),
            _STATE_KEYS,
            where="ticket book",
        )
        if not all(isinstance(d[key], list) for key in ("tickets", "outcomes", "scores")):
            raise InvalidDataError("ticket book tickets, outcomes and scores must be lists")
        clock = None if d["clock_ns"] is None else decode_ns(d["clock_ns"])
        book = cls()
        for item in d["tickets"]:
            if not book.issue(ForecastTicket.from_mapping(item)):
                raise InvalidDataError("ticket book state lists a ticket twice")
        for item in d["outcomes"]:
            if not book.submit(Outcome.from_mapping(item)):
                raise InvalidDataError("ticket book state lists an outcome twice")
        recorded = [ScoreRecord.from_mapping(item) for item in d["scores"]]
        if recorded and clock is None:
            raise InvalidDataError("ticket book state has scores but no clock")
        replayed: list[ScoreRecord] = []
        for released_at in dict.fromkeys(r.released_at_ns for r in recorded):
            replayed.extend(r.score for r in book.release(released_at))
        if clock is not None:
            replayed.extend(r.score for r in book.release(clock))
        if replayed != recorded:
            raise InvalidDataError(
                "ticket book state is inconsistent with its recorded scores",
                context={"recorded": len(recorded), "replayed": len(replayed)},
            )
        return book

    @classmethod
    def from_json(cls, text: str | bytes) -> TicketBook:
        return cls.from_state(loads_strict(text))
