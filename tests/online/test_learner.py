"""DelayedFeedbackLearner: updates only from matured outcomes, release-before-predict, exact resume, crash recovery."""

from __future__ import annotations

import json

import numpy as np
import pytest

from financeai.core.errors import (
    IncompatibleSchemaError,
    InvalidDataError,
    InvariantViolationError,
    UnsupportedCapabilityError,
)
from financeai.core.forecast import ForecastAdapter, ForecastTicket, MatureBatch, Outcome
from financeai.online import LEARNER_SCHEMA, DelayedFeedbackLearner, TicketBook, answer_mismatches

from .toy import HOUR, T0, TARGET, Constant, Flaky, ImmediateOnly, Misbehaving, RunningMean, query

N_STEPS = 40
VALUES = np.random.default_rng(7).normal(10.0, 3.0, size=N_STEPS).round(3).tolist()
DELAYS = [1 + (7 * i) % 5 for i in range(N_STEPS)]  # 1..5 h after issue: outcomes mature out of order


def _available(i: int) -> int:
    return T0 + (i + DELAYS[i]) * HOUR


def _step(learner: DelayedFeedbackLearner, i: int) -> None:
    t = learner.predict(query(i))
    learner.submit(Outcome(t.ticket_id, VALUES[i], _available(i)))


def _finish(learner: DelayedFeedbackLearner) -> None:
    learner.advance(T0 + (N_STEPS + 10) * HOUR)


def _uninterrupted() -> DelayedFeedbackLearner:
    learner = DelayedFeedbackLearner(RunningMean())
    for i in range(N_STEPS):
        _step(learner, i)
    _finish(learner)
    return learner


def test_adapter_is_updated_only_with_matured_outcomes() -> None:
    adapter = RunningMean()
    learner = DelayedFeedbackLearner(adapter)
    for i in range(N_STEPS):
        _step(learner, i)
        now = T0 + i * HOUR  # predict(i) advanced the learner to the issue time first
        matured = {f"q{j:04d}" for j in range(i) if _available(j) <= now}
        assert set(adapter.updated) == matured
        assert len(adapter.updated) == len(matured)  # each matured outcome exactly once
    _finish(learner)
    assert sorted(adapter.updated) == [f"q{j:04d}" for j in range(N_STEPS)]
    assert learner.n_updates == N_STEPS and len(learner.book.scores) == N_STEPS


def test_feedback_available_at_the_issue_time_is_released_before_predicting() -> None:
    learner = DelayedFeedbackLearner(RunningMean())
    learner.predict(query(0))
    learner.submit(Outcome("q0000", 6.0, T0 + 2 * HOUR))
    assert learner.predict(query(1)).mean == 0.0  # at T0 + 1 h the label is still in flight
    assert learner.predict(query(2)).mean == 6.0  # at T0 + 2 h it is released before the forecast is issued


def test_an_outcome_created_in_a_cohort_never_trains_a_peer_of_that_cohort() -> None:
    learner = DelayedFeedbackLearner(RunningMean())
    learner.predict(query(0))
    learner.predict(query(1))
    with pytest.raises(InvalidDataError):
        learner.submit(Outcome("q0000", 1.0, T0 + HOUR))  # the T0 + 1 h boundary has already been released


def test_queries_from_the_past_are_rejected() -> None:
    learner = DelayedFeedbackLearner(RunningMean())
    learner.advance(T0 + 5 * HOUR)
    with pytest.raises(InvalidDataError):
        learner.predict(query(4))


def test_frozen_adapters_are_scored_but_never_updated() -> None:
    learner = DelayedFeedbackLearner(Constant())
    assert learner.updating is False
    t = learner.predict(query(0))
    learner.submit(Outcome(t.ticket_id, 5.0, T0 + 2 * HOUR))
    released = learner.advance(T0 + 3 * HOUR)
    assert [r.score.abs_error for r in released] == [3.0]
    assert learner.n_updates == 0
    with pytest.raises(UnsupportedCapabilityError):
        DelayedFeedbackLearner(Constant(), online_update=True)
    with pytest.raises(UnsupportedCapabilityError):
        Constant().update(t, Outcome(t.ticket_id, 5.0, T0 + 2 * HOUR))


def test_online_adapters_must_declare_delayed_feedback() -> None:
    with pytest.raises(UnsupportedCapabilityError) as info:
        DelayedFeedbackLearner(ImmediateOnly())
    assert info.value.context["capability"] == "delayed_feedback"
    assert DelayedFeedbackLearner(ImmediateOnly(), online_update=False).updating is False


def test_unsupported_quantile_requests_raise_capability_errors() -> None:
    learner = DelayedFeedbackLearner(RunningMean())
    with pytest.raises(UnsupportedCapabilityError):
        learner.predict(query(0, levels=(0.5,)))
    assert learner.book.tickets == ()


def test_inconsistent_adapter_tickets_are_invariant_violations() -> None:
    learner = DelayedFeedbackLearner(Misbehaving())
    with pytest.raises(InvariantViolationError) as info:
        learner.predict(query(0))
    assert info.value.context["fields"] == ["ticket_id"]
    assert learner.book.tickets == ()


def test_answer_mismatches_names_every_field_in_which_a_ticket_fails_its_query() -> None:
    q = query(3, x=1.5)
    t = ForecastTicket.from_query(q, model_id="toy", model_version="1", mean=0.0)
    assert answer_mismatches(t, q, model_id="toy") == ()
    assert answer_mismatches(t, query(4, x=2.5), model_id="other") == (
        "ticket_id",
        "issued_at_ns",
        "feature_cutoff_ns",
        "features",
        "model_id",
    )
    assert answer_mismatches(t, query(3, x=1.5, levels=(0.5,)), model_id="toy") == ("quantile_levels",)


def test_fit_and_predict_conform_to_the_adapter_protocol() -> None:
    adapter = RunningMean()
    assert isinstance(adapter, ForecastAdapter)
    batch = MatureBatch(
        target=TARGET,
        feature_names=("i", "x"),
        transform_version="toy@1",
        x=np.zeros((2, 2)),
        y=np.array([2.0, 4.0]),
        feature_cutoff_ns=np.array([T0 - 9 * HOUR, T0 - 8 * HOUR]),
        available_at_ns=np.array([T0 - 8 * HOUR, T0 - 7 * HOUR]),
        as_of_ns=T0,
    )
    adapter.fit(batch)
    ticket = DelayedFeedbackLearner(adapter).predict(query(0))
    assert (ticket.mean, ticket.variance, ticket.model_version) == (3.0, 2.0, "u2")


@pytest.mark.parametrize("cut", [0, 1, 7, 23, N_STEPS - 1, N_STEPS])
def test_interrupted_and_resumed_run_equals_the_uninterrupted_run(cut: int) -> None:
    reference = _uninterrupted().to_json()
    first = DelayedFeedbackLearner(RunningMean())
    for i in range(cut):
        _step(first, i)
    checkpoint = first.to_json()
    assert json.loads(checkpoint)["schema"] == LEARNER_SCHEMA
    resumed = DelayedFeedbackLearner.from_json(checkpoint, RunningMean())
    assert resumed.to_json() == checkpoint
    for i in range(cut, N_STEPS):
        _step(resumed, i)
    _finish(resumed)
    assert resumed.to_json() == reference


@pytest.mark.parametrize("fail_at", [1, 5, 17])
def test_a_crash_between_scoring_and_update_restarts_without_double_or_lost_updates(fail_at: int) -> None:
    reference = _uninterrupted()
    learner = DelayedFeedbackLearner(Flaky(fail_at))
    checkpoint, cursor = learner.to_json(), 0
    with pytest.raises(RuntimeError, match="injected crash"):
        for i in range(N_STEPS):
            if i % 4 == 0:
                checkpoint, cursor = learner.to_json(), i  # an atomic checkpoint of learner + adapter + book
            _step(learner, i)
        _finish(learner)
    for call in (
        lambda: learner.predict(query(N_STEPS + 1)),
        lambda: learner.advance(T0 + 99 * HOUR),
        learner.to_json,
        lambda: learner.submit(Outcome("q0000", 1.0, T0 + 99 * HOUR)),
    ):
        with pytest.raises(InvariantViolationError):
            call()  # the book released outcomes the adapter never consumed: refuse to continue or checkpoint
    resumed = DelayedFeedbackLearner.from_json(checkpoint, RunningMean())
    for i in range(cursor, N_STEPS):
        _step(resumed, i)
        resumed.submit(Outcome(f"q{i:04d}", VALUES[i], _available(i)))  # re-delivered feedback is a no-op
    _finish(resumed)
    assert resumed.to_json() == reference.to_json()
    assert resumed.n_updates == reference.n_updates == N_STEPS
    assert len(resumed.book.scores) == N_STEPS


def test_owners_reconcile_the_restored_book_before_the_adapter_is_touched() -> None:
    state = _uninterrupted().to_json()
    seen: list[tuple[int, int, bool]] = []

    def reject(book: TicketBook, updating: bool) -> None:
        seen.append((len(book.tickets), len(book.scores), updating))
        raise InvalidDataError("not this owner's book")

    adapter = RunningMean()
    with pytest.raises(InvalidDataError, match="not this owner's book"):
        DelayedFeedbackLearner.from_json(state, adapter, reconcile=reject)
    assert seen == [(N_STEPS, N_STEPS, True)]
    assert (adapter.n, adapter.updated) == (0, [])  # a rejected checkpoint never reaches the adapter
    restored = DelayedFeedbackLearner.from_json(state, adapter, reconcile=lambda book, updating: None)
    assert restored.to_json() == state


def test_restoring_into_a_different_adapter_is_rejected() -> None:
    state = _uninterrupted().to_json()
    with pytest.raises(IncompatibleSchemaError):
        DelayedFeedbackLearner.from_json(state, Constant())
    doc = json.loads(state)
    doc["schema_version"] = "2.0"
    with pytest.raises(IncompatibleSchemaError):
        DelayedFeedbackLearner.from_json(json.dumps(doc), RunningMean())
    doc = json.loads(state)
    doc["n_updates"] = 3
    with pytest.raises(InvalidDataError):
        DelayedFeedbackLearner.from_json(json.dumps(doc), RunningMean())
