"""DelayedFeedbackLearner: the single mutation owner coupling one forecast adapter to its TicketBook.

Per logical time ``t`` (one cohort) ``advance(t)`` releases every outcome available at ``t`` and feeds each one, in
release order, to ``adapter.update``; only adapters declaring ``online_update`` *and* ``delayed_feedback`` are
updated, so no adapter ever sees a label before its availability clock. ``predict(query)`` first advances to the
query's issue time, so prior-ticket feedback is delivered before a forecast is issued, then validates that the
adapter's ticket answers exactly that query before recording it.

``to_json`` checkpoints adapter state, tickets, scheduled/released outcomes and scores atomically in one
strict-JSON document; a run interrupted at a checkpoint and resumed from it equals the uninterrupted run. If an
update raises after the book has released outcomes, the in-memory learner is marked inconsistent and refuses any
further use (including checkpointing): restore the last checkpoint instead, which neither double-applies nor loses
a released outcome. This is a tested local idempotence property, not a distributed exactly-once guarantee.
A restoring owner can reconcile the checkpointed book and update mode with its own records before the adapter state
is applied (``from_state(..., reconcile=...)``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from financeai.core.errors import IncompatibleSchemaError, InvalidDataError, InvariantViolationError
from financeai.core.forecast import (
    Capability,
    ForecastAdapter,
    ForecastQuery,
    ForecastTicket,
    Outcome,
    require_capability,
    require_keys,
    require_schema,
)
from financeai.core.serialization import canonical_json, loads_strict
from financeai.online.book import Released, TicketBook

LEARNER_SCHEMA: Final = "financeai.online.learner"
LEARNER_SCHEMA_VERSION: Final = "1.0"
_STATE_KEYS: Final = (
    "schema",
    "schema_version",
    "model_id",
    "model_version",
    "updating",
    "n_updates",
    "adapter",
    "book",
)


def answer_mismatches(ticket: ForecastTicket, query: ForecastQuery, *, model_id: str) -> tuple[str, ...]:
    """Fields in which ``ticket`` fails to answer ``query`` as issued by ``model_id`` (empty when it answers it)."""
    checks = (
        ("ticket_id", ticket.ticket_id == query.query_id),
        ("stream", ticket.stream == query.stream),
        ("target", ticket.target == query.target),
        ("issued_at_ns", ticket.issued_at_ns == query.issued_at_ns),
        ("feature_cutoff_ns", ticket.feature_cutoff_ns == query.feature_cutoff_ns),
        ("transform_version", ticket.transform_version == query.transform_version),
        ("features", ticket.features.tobytes() == query.features.tobytes()),
        ("quantile_levels", ticket.quantile_levels == query.quantile_levels),
        ("model_id", ticket.model_id == model_id),
    )
    return tuple(name for name, ok in checks if not ok)


def _check_issued(ticket: object, query: ForecastQuery, model_id: str) -> ForecastTicket:
    if not isinstance(ticket, ForecastTicket):
        raise InvariantViolationError("adapter did not return a ForecastTicket", context={"model_id": model_id})
    mismatched = answer_mismatches(ticket, query, model_id=model_id)
    if mismatched:
        raise InvariantViolationError(
            "adapter issued a ticket that does not answer the query",
            context={"query_id": query.query_id, "fields": list(mismatched)},
        )
    return ticket


class DelayedFeedbackLearner:
    """Wrap ``adapter`` so it learns only from matured outcomes of its own issued tickets.

    ``online_update=None`` updates exactly when the adapter declares ``online_update`` (which then also requires
    ``delayed_feedback``); ``True`` demands it and ``False`` scores tickets without ever updating the adapter.
    """

    def __init__(self, adapter: ForecastAdapter, *, online_update: bool | None = None) -> None:
        capabilities = frozenset(adapter.capabilities)
        if online_update is not None and not isinstance(online_update, bool):
            raise InvalidDataError("online_update must be a bool or None")
        updating = Capability.ONLINE_UPDATE in capabilities if online_update is None else online_update
        if updating:
            require_capability(capabilities, Capability.ONLINE_UPDATE, component=adapter.model_id)
            require_capability(capabilities, Capability.DELAYED_FEEDBACK, component=adapter.model_id)
        self._adapter = adapter
        self._book = TicketBook()
        self._updating = updating
        self._n_updates = 0
        self._failure: str | None = None

    @property
    def adapter(self) -> ForecastAdapter:
        return self._adapter

    @property
    def book(self) -> TicketBook:
        return self._book

    @property
    def updating(self) -> bool:
        return self._updating

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def clock_ns(self) -> int | None:
        return self._book.clock_ns

    def _ensure_consistent(self) -> None:
        if self._failure is not None:
            raise InvariantViolationError(
                "learner is inconsistent after a failed update; restore the last checkpoint",
                context={"cause": self._failure},
            )

    def advance(self, now_ns: int) -> tuple[Released, ...]:
        """Release outcomes available at ``now_ns`` and update the adapter with each, in release order."""
        self._ensure_consistent()
        released = self._book.release(now_ns)
        if self._updating:
            for item in released:
                try:
                    self._adapter.update(item.ticket, item.outcome)
                except BaseException as exc:
                    self._failure = f"{type(exc).__name__} while updating with ticket {item.ticket.ticket_id}"
                    raise
                self._n_updates += 1
        return released

    def predict(self, query: ForecastQuery) -> ForecastTicket:
        """Release feedback due at the issue time, then issue and record the adapter's ticket."""
        self._ensure_consistent()
        if not isinstance(query, ForecastQuery):
            raise InvalidDataError("predict needs a ForecastQuery")
        self.advance(query.issued_at_ns)
        ticket = _check_issued(self._adapter.predict(query), query, self._adapter.model_id)
        self._book.issue(ticket)
        return ticket

    def submit(self, outcome: Outcome) -> bool:
        """Schedule an outcome; it reaches the adapter only once released (see :class:`TicketBook`)."""
        self._ensure_consistent()
        return self._book.submit(outcome)

    def get_state(self) -> dict[str, Any]:
        self._ensure_consistent()
        return {
            "schema": LEARNER_SCHEMA,
            "schema_version": LEARNER_SCHEMA_VERSION,
            "model_id": self._adapter.model_id,
            "model_version": self._adapter.model_version,
            "updating": self._updating,
            "n_updates": self._n_updates,
            "adapter": self._adapter.get_state(),
            "book": self._book.get_state(),
        }

    def to_json(self) -> str:
        return canonical_json(self.get_state())

    @classmethod
    def from_state(
        cls, doc: object, adapter: ForecastAdapter, *, reconcile: Callable[[TicketBook, bool], None] | None = None
    ) -> DelayedFeedbackLearner:
        """Restore a checkpoint into a freshly configured ``adapter`` of the same model.

        ``reconcile(book, updating)`` lets the owner compare the restored book and update mode with its own records
        and raise to reject them. It runs before ``adapter`` is touched, so such a rejection leaves it unchanged.
        """
        d = require_keys(
            require_schema(doc, schema=LEARNER_SCHEMA, version=LEARNER_SCHEMA_VERSION, where="learner"),
            _STATE_KEYS,
            where="learner",
        )
        if d["model_id"] != adapter.model_id:
            raise IncompatibleSchemaError(
                "checkpoint belongs to another model",
                context={"checkpoint": d["model_id"], "adapter": adapter.model_id},
            )
        updating, n_updates = d["updating"], d["n_updates"]
        if not isinstance(updating, bool) or isinstance(n_updates, bool) or not isinstance(n_updates, int):
            raise InvalidDataError("learner checkpoint has malformed update fields")
        if not isinstance(d["adapter"], Mapping):
            raise InvalidDataError("learner checkpoint adapter state must be an object")
        book = TicketBook.from_state(d["book"])
        expected = len(book.scores) if updating else 0
        if n_updates != expected:
            raise InvalidDataError(
                "checkpoint update count disagrees with the released scores",
                context={"n_updates": n_updates, "expected": expected},
            )
        if reconcile is not None:
            reconcile(book, updating)
        adapter.set_state(d["adapter"])
        if adapter.model_version != d["model_version"]:
            raise IncompatibleSchemaError(
                "adapter state did not restore the checkpointed model version",
                context={"checkpoint": d["model_version"], "adapter": adapter.model_version},
            )
        learner = cls(adapter, online_update=updating)
        learner._book = book
        learner._n_updates = n_updates
        return learner

    @classmethod
    def from_json(
        cls, text: str | bytes, adapter: ForecastAdapter, *, reconcile: Callable[[TicketBook, bool], None] | None = None
    ) -> DelayedFeedbackLearner:
        return cls.from_state(loads_strict(text), adapter, reconcile=reconcile)
