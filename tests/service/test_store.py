"""SQLite job store: legal transitions, idempotent admission, bounded queue and the publication outbox."""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from financeai.core.errors import IncompatibleSchemaError
from financeai.service.errors import (
    IdempotencyConflictError,
    NotFoundError,
    QueueFullError,
    StateConflictError,
)
from financeai.service.store import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    ArtifactRecord,
    JobStatus,
    JobStore,
    RunRecord,
    check_transition,
    is_artifact_id,
    is_run_id,
    new_id,
)

LEGAL = {
    (JobStatus.QUEUED, JobStatus.RUNNING),
    (JobStatus.QUEUED, JobStatus.CANCELLED),
    (JobStatus.QUEUED, JobStatus.FAILED),
    (JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED),
    (JobStatus.RUNNING, JobStatus.SUCCEEDED),
    (JobStatus.RUNNING, JobStatus.FAILED),
    (JobStatus.RUNNING, JobStatus.INTERRUPTED),
    (JobStatus.CANCEL_REQUESTED, JobStatus.CANCELLED),
    (JobStatus.CANCEL_REQUESTED, JobStatus.SUCCEEDED),
    (JobStatus.CANCEL_REQUESTED, JobStatus.FAILED),
    (JobStatus.CANCEL_REQUESTED, JobStatus.INTERRUPTED),
}


@pytest.fixture
def store(tmp_path: Path) -> Iterator[JobStore]:
    s = JobStore(tmp_path / "jobs.sqlite3")
    yield s
    s.close()


def admit(
    store: JobStore, key: str = "k1", digest: str = "a" * 64, campaign: str = "c", capacity: int = 10
) -> tuple[RunRecord, bool]:
    return store.admit(
        campaign_id=campaign,
        idempotency_key=key,
        request_hash=digest,
        request={"kind": "synthetic_replay", "seed": 1},
        kind="synthetic_replay",
        evidence_scope=("SYNTHETIC",),
        queue_capacity=capacity,
    )


def artifact(run_id: str | None = None, rel: str = "runs/x/summary.json") -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=new_id("art"),
        kind="run_summary",
        media_type="application/json",
        schema="financeai.run_summary/1",
        sha256="b" * 64,
        size_bytes=12,
        rel_path=rel,
        data_role="SYNTHETIC",
        provenance={"producer": "run"},
        created_ns=1,
        run_id=run_id,
    )


@pytest.mark.parametrize(("old", "new"), list(itertools.product(JobStatus, JobStatus)))
def test_transition_table(old: JobStatus, new: JobStatus) -> None:
    if (old, new) in LEGAL:
        check_transition(old, new)
    else:
        with pytest.raises(StateConflictError):
            check_transition(old, new)


def test_status_sets() -> None:
    assert {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED} == TERMINAL_STATUSES
    assert {JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED} == ACTIVE_STATUSES


def test_ids_are_prefixed_hex_and_validated() -> None:
    rid, aid = new_id("run"), new_id("art")
    assert is_run_id(rid) and is_artifact_id(aid) and not is_run_id(aid)
    for bad in ("", "run_", "run_" + "g" * 32, "../" + rid, rid + "/x", rid.upper(), "art_..", None, 5):
        assert not is_run_id(bad) and not is_artifact_id(bad)


def test_admission_is_idempotent_per_campaign_and_key(store: JobStore) -> None:
    first, replayed = admit(store)
    assert not replayed and first.status is JobStatus.QUEUED and is_run_id(first.run_id)
    again, replayed = admit(store)
    assert replayed and again.run_id == first.run_id
    with pytest.raises(IdempotencyConflictError):
        admit(store, digest="c" * 64)
    other, replayed = admit(store, campaign="other")
    assert not replayed and other.run_id != first.run_id
    assert store.get_run(first.run_id).request == {"kind": "synthetic_replay", "seed": 1}
    assert store.get_run(first.run_id).evidence_scope == ("SYNTHETIC",)


def test_concurrent_duplicate_admissions_create_one_run(store: JobStore) -> None:
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: admit(store, key="same"), range(16)))
    assert len({r.run_id for r, _ in results}) == 1
    assert sum(not replayed for _, replayed in results) == 1


def test_queue_capacity_counts_only_queued_runs(store: JobStore) -> None:
    a, _ = admit(store, key="a", capacity=2)
    admit(store, key="b", capacity=2)
    with pytest.raises(QueueFullError):
        admit(store, key="c", capacity=2)
    assert admit(store, key="a", capacity=2)[1]  # a replay is never rejected as full
    store.start_attempt(a.run_id)
    admit(store, key="c", capacity=2)


def test_queued_runs_skip_excluded_campaigns_in_the_query(store: JobStore) -> None:
    """A busy campaign's backlog must not hide other campaigns' runs behind the query window."""
    for i in range(5):
        admit(store, key=f"a{i}", campaign="A", capacity=100)
    b, _ = admit(store, key="b0", campaign="B", capacity=100)
    c, _ = admit(store, key="c0", campaign="C", capacity=100)
    assert [r.campaign_id for r in store.queued_runs(limit=3)] == ["A", "A", "A"]
    assert [r.run_id for r in store.queued_runs(limit=3, exclude_campaigns={"A"})] == [b.run_id, c.run_id]
    assert [r.run_id for r in store.queued_runs(limit=1, exclude_campaigns=["A", "B"])] == [c.run_id]
    assert store.queued_runs(limit=5, exclude_campaigns={"A", "B", "C"}) == []


def test_attempt_lifecycle_and_stale_attempts(store: JobStore) -> None:
    rec, _ = admit(store)
    running = store.start_attempt(rec.run_id)
    assert running.status is JobStatus.RUNNING and running.generation == 1 and running.attempt_id
    assert running.started_ns is not None and running.finished_ns is None
    with pytest.raises(StateConflictError):
        store.start_attempt(rec.run_id)
    assert store.set_progress(
        rec.run_id, running.attempt_id, {"phase": "replay", "completed_steps": 1, "total_steps": 4}
    )
    assert not store.set_progress(rec.run_id, "att_" + "0" * 32, {"phase": "x"})
    assert store.get_run(rec.run_id).progress == {"phase": "replay", "completed_steps": 1, "total_steps": 4}
    with pytest.raises(StateConflictError):
        store.transition(rec.run_id, JobStatus.FAILED, attempt_id="att_" + "0" * 32)
    done = store.transition(
        rec.run_id,
        JobStatus.FAILED,
        attempt_id=running.attempt_id,
        failure={"code": "worker_crashed", "message": "exit 3"},
    )
    assert done.status is JobStatus.FAILED and done.finished_ns is not None
    assert done.failure == {"code": "worker_crashed", "message": "exit 3"}
    with pytest.raises(StateConflictError):
        store.transition(rec.run_id, JobStatus.SUCCEEDED, attempt_id=running.attempt_id)
    assert store.transitions(rec.run_id) == [(None, "QUEUED"), ("QUEUED", "RUNNING"), ("RUNNING", "FAILED")]


def test_cancel_requests(store: JobStore) -> None:
    queued, _ = admit(store, key="q")
    assert store.request_cancel(queued.run_id).status is JobStatus.CANCELLED
    assert store.request_cancel(queued.run_id).status is JobStatus.CANCELLED  # idempotent on terminal
    running, _ = admit(store, key="r")
    store.start_attempt(running.run_id)
    assert store.request_cancel(running.run_id).status is JobStatus.CANCEL_REQUESTED
    assert store.request_cancel(running.run_id).status is JobStatus.CANCEL_REQUESTED
    with pytest.raises(NotFoundError):
        store.request_cancel(new_id("run"))


def test_publication_outbox_commits_exactly_once(store: JobStore) -> None:
    rec, _ = admit(store)
    running = store.start_attempt(rec.run_id)
    assert running.attempt_id is not None
    arts = [artifact(rec.run_id, "runs/a/summary.json"), artifact(rec.run_id, "runs/a/fills.csv")]
    op = store.begin_publication(
        rec.run_id, running.attempt_id, {"artifacts": [a.to_mapping() for a in arts], "result": {"net_pnl": 0.5}}
    )
    assert [o.op_id for o in store.pending_operations()] == [op.op_id]
    with pytest.raises(StateConflictError):  # one pending publication per run
        store.begin_publication(rec.run_id, running.attempt_id, {"artifacts": [], "result": {}})
    store.request_cancel(rec.run_id)  # a cancel request racing a publication that already began
    done = store.complete_publication(op.op_id)
    assert done.status is JobStatus.SUCCEEDED and done.result == {"net_pnl": 0.5}
    again = store.complete_publication(op.op_id)
    assert again.status is JobStatus.SUCCEEDED
    assert sorted(a.artifact_id for a in store.artifacts_for_run(rec.run_id)) == sorted(a.artifact_id for a in arts)
    assert store.get_artifact(arts[0].artifact_id) == arts[0]
    assert store.pending_operations() == []


def test_publication_requires_the_active_attempt(store: JobStore) -> None:
    rec, _ = admit(store)
    running = store.start_attempt(rec.run_id)
    assert running.attempt_id is not None
    with pytest.raises(StateConflictError):
        store.begin_publication(rec.run_id, "att_" + "1" * 32, {"artifacts": [], "result": {}})
    store.transition(rec.run_id, JobStatus.CANCEL_REQUESTED, attempt_id=running.attempt_id)
    store.transition(rec.run_id, JobStatus.CANCELLED, attempt_id=running.attempt_id)
    with pytest.raises(StateConflictError):
        store.begin_publication(rec.run_id, running.attempt_id, {"artifacts": [], "result": {}})


def test_aborted_publication_cannot_complete(store: JobStore) -> None:
    rec, _ = admit(store)
    running = store.start_attempt(rec.run_id)
    assert running.attempt_id is not None
    op = store.begin_publication(rec.run_id, running.attempt_id, {"artifacts": [], "result": {}})
    store.abort_publication(op.op_id)
    assert store.pending_operations() == []
    with pytest.raises(StateConflictError):
        store.complete_publication(op.op_id)
    assert store.get_run(rec.run_id).status is JobStatus.RUNNING


def test_registered_artifacts_and_unknown_ids(store: JobStore) -> None:
    a = artifact(None, "inputs/x/strategy.json")
    store.add_artifacts([a])
    assert store.get_artifact(a.artifact_id) == a
    with pytest.raises(NotFoundError):
        store.get_artifact(new_id("art"))
    with pytest.raises(NotFoundError):
        store.get_run(new_id("run"))


def test_state_survives_reopen_and_rejects_newer_schema(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    s = JobStore(path)
    rec, _ = admit(s)
    s.close()
    s2 = JobStore(path)
    assert s2.get_run(rec.run_id).status is JobStatus.QUEUED
    assert s2.queued_runs(10)[0].run_id == rec.run_id
    s2.close()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(IncompatibleSchemaError):
        JobStore(path)
