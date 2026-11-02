"""TicketBook: maturity gating, score-once idempotence, deterministic cohorts and exact strict-JSON state."""

from __future__ import annotations

import json
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.core.errors import IncompatibleSchemaError, InvalidConfigError, InvalidDataError
from financeai.core.forecast import Outcome
from financeai.online import BOOK_SCHEMA, TicketBook, pinball_loss, score_ticket

from .toy import HOUR, T0, ticket


def _book(n: int) -> TicketBook:
    book = TicketBook()
    for i in range(n):
        assert book.issue(ticket(i, mean=float(i)))
    return book


def test_outcomes_stay_invisible_until_their_availability_clock() -> None:
    book = _book(3)
    t = [ticket(i) for i in range(3)]
    # issued at T0 + i h with a 1 h horizon: the earliest legitimate availability is T0 + (i + 1) h
    book.submit(Outcome(t[0].ticket_id, 10.0, T0 + 5 * HOUR))
    book.submit(Outcome(t[1].ticket_id, 11.0, T0 + 2 * HOUR))
    book.submit(Outcome(t[2].ticket_id, 12.0, T0 + 5 * HOUR))
    assert book.release(T0 + 2 * HOUR - 1) == ()
    assert book.score(t[1].ticket_id) is None
    first = book.release(T0 + 2 * HOUR)
    assert [r.ticket.ticket_id for r in first] == [t[1].ticket_id]
    assert first[0].outcome.value == 11.0 and first[0].score.released_at_ns == T0 + 2 * HOUR
    assert book.release(T0 + 4 * HOUR) == ()
    later = book.release(T0 + 9 * HOUR)
    assert [r.ticket.ticket_id for r in later] == [t[0].ticket_id, t[2].ticket_id]  # tie broken by ticket ID
    assert {s.ticket_id for s in book.scores} == {x.ticket_id for x in t}
    assert book.n_pending == 0 and book.unscored_ids() == ()


@settings(max_examples=150, deadline=None)
@given(
    st.lists(st.integers(1, 12), min_size=1, max_size=12),
    st.lists(st.integers(0, 4), min_size=1, max_size=8),
    st.randoms(use_true_random=False),
)
def test_every_outcome_is_released_exactly_once_and_never_early(
    delays: list[int], steps: list[int], rnd: random.Random
) -> None:
    n = len(delays)
    outcomes = [Outcome(ticket(i).ticket_id, float(i), T0 + (i + delays[i]) * HOUR) for i in range(n)]

    def run(order: list[int]) -> list[tuple[int, list[str]]]:
        book = _book(n)
        for i in order:
            book.submit(outcomes[i])
        clock, log = T0 + n * HOUR, []
        for step in [*steps, 100]:
            clock += step * HOUR
            released = book.release(clock)
            assert all(r.outcome.available_at_ns <= clock for r in released)
            log.append((clock, [r.ticket.ticket_id for r in released]))
        return log

    order = list(range(n))
    reference = run(order)
    rnd.shuffle(order)
    assert run(order) == reference  # submission order inside a cohort never changes the release sequence
    released = [tid for _, ids in reference for tid in ids]
    assert sorted(released) == sorted(o.ticket_id for o in outcomes)  # once each, none lost
    by_id = {o.ticket_id: o for o in outcomes}
    previous: int | None = None
    for clock, ids in reference:
        for tid in ids:  # released at the first release clock that reaches its availability
            available = by_id[tid].available_at_ns
            assert available <= clock and (previous is None or available > previous)
        keys = [(by_id[tid].available_at_ns, tid) for tid in ids]
        assert keys == sorted(keys)
        previous = clock


def test_each_ticket_is_scored_exactly_once() -> None:
    book = _book(1)
    outcome = Outcome(ticket(0).ticket_id, 4.0, T0 + 2 * HOUR, version=1)
    assert book.submit(outcome) is True
    assert book.submit(outcome) is False  # duplicate delivery before release: no-op
    assert len(book.release(T0 + 3 * HOUR)) == 1
    assert book.submit(outcome) is False  # duplicate delivery after scoring: no-op
    assert book.release(T0 + 4 * HOUR) == ()
    assert len(book.scores) == 1


@pytest.mark.parametrize(
    "second",
    [
        Outcome("q0000", 5.0, T0 + 2 * HOUR, version=1),  # same version, different value
        Outcome("q0000", 4.0, T0 + 3 * HOUR, version=1),  # same version, different clock
        Outcome("q0000", 4.0, T0 + 2 * HOUR, version=2),  # a revision: issued scores are final
    ],
)
def test_conflicting_or_revised_outcomes_are_rejected(second: Outcome) -> None:
    book = _book(1)
    book.submit(Outcome("q0000", 4.0, T0 + 2 * HOUR, version=1))
    with pytest.raises(InvalidDataError):
        book.submit(second)
    book.release(T0 + 5 * HOUR)
    with pytest.raises(InvalidDataError):
        book.submit(second)
    assert book.scores[0].value == 4.0 and book.scores[0].outcome_version == 1


def test_outcomes_must_reference_a_ticket_and_respect_its_availability_rule() -> None:
    book = _book(1)
    with pytest.raises(InvalidDataError, match="unknown ticket"):
        book.submit(Outcome("nope", 1.0, T0 + HOUR))
    with pytest.raises(InvalidDataError, match="availability rule"):
        book.submit(Outcome("q0000", 1.0, T0 + HOUR - 1))  # before cutoff + horizon
    assert book.submit(Outcome("q0000", 1.0, T0 + HOUR))


def test_feedback_cannot_join_a_cohort_whose_boundary_was_released() -> None:
    book = _book(2)
    book.release(T0 + 3 * HOUR)
    with pytest.raises(InvalidDataError, match="boundary"):
        book.submit(Outcome("q0000", 1.0, T0 + 3 * HOUR))
    with pytest.raises(InvalidDataError, match="boundary"):
        book.submit(Outcome("q0001", 1.0, T0 + 2 * HOUR))
    assert book.submit(Outcome("q0001", 1.0, T0 + 3 * HOUR + 1))


def test_clock_reversal_is_rejected() -> None:
    book = _book(1)
    book.release(T0 + 5 * HOUR)
    with pytest.raises(InvalidDataError):
        book.release(T0 + 4 * HOUR)
    with pytest.raises(InvalidDataError, match="before the book clock"):
        book.issue(ticket(3))  # issued at T0 + 3 h, before the released boundary
    assert book.release(T0 + 5 * HOUR) == ()  # repeating the same clock is allowed


def test_ticket_ids_are_unique() -> None:
    book = _book(1)
    assert book.issue(ticket(0, mean=0.0)) is False  # identical re-issue is idempotent
    with pytest.raises(InvalidDataError):
        book.issue(ticket(0, mean=1.0))  # a recomputed forecast needs a new ticket ID
    assert book.ticket("q0000").mean == 0.0
    with pytest.raises(InvalidDataError):
        book.ticket("missing")


def test_score_records_use_the_issued_forecast() -> None:
    t = ticket(0, mean=5.0, levels=(0.1, 0.5, 0.9), quantiles=[3.0, 5.0, 7.0])
    record = score_ticket(t, Outcome(t.ticket_id, 6.0, T0 + 2 * HOUR, version=2), released_at_ns=T0 + 3 * HOUR)
    assert (record.error, record.abs_error, record.squared_error) == (1.0, 1.0, 1.0)
    assert record.quantile_levels == (0.1, 0.5, 0.9)
    assert record.pinball == pytest.approx((0.3, 0.5, 0.1))
    assert (record.outcome_version, record.released_at_ns, record.mean, record.value) == (2, T0 + 3 * HOUR, 5.0, 6.0)
    point = score_ticket(ticket(1, mean=2.0), Outcome("q0001", -1.0, T0 + 2 * HOUR), released_at_ns=T0 + 2 * HOUR)
    assert (point.error, point.abs_error, point.squared_error, point.pinball) == (-3.0, 3.0, 9.0, ())
    with pytest.raises(InvalidDataError):
        score_ticket(t, Outcome("other", 6.0, T0 + 2 * HOUR), released_at_ns=T0 + 3 * HOUR)
    with pytest.raises(InvalidDataError):
        score_ticket(t, Outcome(t.ticket_id, 6.0, T0 + 2 * HOUR), released_at_ns=T0 + 2 * HOUR - 1)


@pytest.mark.parametrize(
    ("value", "quantile", "level", "expected"),
    [
        (6.0, 3.0, 0.1, 0.3),
        (6.0, 7.0, 0.9, 0.1),
        (5.0, 5.0, 0.5, 0.0),
        (0.0, 2.0, 0.25, 1.5),
        (2.0, 0.0, 0.25, 0.5),
    ],
)
def test_pinball_loss_definition(value: float, quantile: float, level: float, expected: float) -> None:
    assert pinball_loss(value, quantile, level) == pytest.approx(expected)


def test_unscored_tickets_are_reported_never_zero_filled() -> None:
    book = _book(3)
    book.submit(Outcome("q0001", 1.0, T0 + 2 * HOUR))
    book.release(T0 + 10 * HOUR)
    assert book.unscored_ids() == ("q0000", "q0002")
    assert [s.ticket_id for s in book.scores] == ["q0001"]


def _populated() -> TicketBook:
    book = _book(4)
    book.submit(Outcome("q0000", 1.5, T0 + 3 * HOUR))
    book.submit(Outcome("q0001", 2.5, T0 + 3 * HOUR))
    book.submit(Outcome("q0002", 3.5, T0 + 9 * HOUR, version=4))
    book.release(T0 + 4 * HOUR)
    return book


def test_state_round_trips_through_strict_json_and_continues_identically() -> None:
    book = _populated()
    text = book.to_json()
    assert json.loads(text)["schema"] == BOOK_SCHEMA
    restored = TicketBook.from_json(text)
    assert restored.to_json() == text
    assert restored.clock_ns == T0 + 4 * HOUR and restored.n_pending == 1
    for b in (book, restored):
        b.submit(Outcome("q0003", 4.5, T0 + 8 * HOUR))
    assert [r.score for r in book.release(T0 + 10 * HOUR)] == [r.score for r in restored.release(T0 + 10 * HOUR)]
    assert book.to_json() == restored.to_json()
    assert TicketBook.from_json(TicketBook().to_json()).to_json() == TicketBook().to_json()


def test_state_loading_rejects_tampering_and_unknown_schemas() -> None:
    doc = json.loads(_populated().to_json())
    tampered = json.loads(json.dumps(doc))
    tampered["scores"][0]["abs_error"] = 0.0
    with pytest.raises(InvalidDataError):
        TicketBook.from_json(json.dumps(tampered))
    early = json.loads(json.dumps(doc))
    early["clock_ns"] = str(T0 + 2 * HOUR)  # scores released after the recorded clock
    with pytest.raises(InvalidDataError):
        TicketBook.from_json(json.dumps(early))
    for key, value in (("schema", "financeai.other"), ("schema_version", "2.0")):
        with pytest.raises(IncompatibleSchemaError):
            TicketBook.from_json(json.dumps({**doc, key: value}))
    with pytest.raises(InvalidDataError):
        TicketBook.from_json(json.dumps({**doc, "extra": []}))
    text = json.dumps(doc)
    with pytest.raises(InvalidConfigError):
        TicketBook.from_json(text[:-1] + ', "clock_ns": null}')  # duplicate key
    with pytest.raises(InvalidConfigError):
        TicketBook.from_json(text.replace('"value": 1.5', '"value": NaN'))
