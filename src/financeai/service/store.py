"""Durable SQLite job store: runs, registered artifacts, a publication outbox and an audit log.

* ``runs`` carries ``UNIQUE(campaign_id, idempotency_key)`` plus the normalized request hash:
  the same key and request return the same run, the same key with another request is a
  409 conflict. Admission, the bounded-queue check and the insert share one transaction.
* Job states move only along :data:`_TRANSITIONS`; terminal states never change again.
* ``operations`` is the outbox that ties the database to files: a publication is recorded as
  PENDING before the staged files are renamed into the artifact store and only then
  completed (artifacts registered, run SUCCEEDED, operation DONE) in one transaction.
  Recovery completes a matching PENDING operation exactly once or aborts it.
* ``transitions`` is append-only. Clocks are wall-clock int64 epoch nanoseconds.

One connection is shared by the service threads behind a re-entrant lock; SQLite runs in WAL
mode with ``synchronous=FULL``. Other local processes (e.g. a registration command) may use
the same database file concurrently.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from financeai.core.errors import IncompatibleSchemaError
from financeai.core.serialization import canonical_json, loads_strict
from financeai.service.errors import (
    IdempotencyConflictError,
    NotFoundError,
    QueueFullError,
    StateConflictError,
)

SCHEMA_VERSION: Final = 1


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


TERMINAL_STATUSES: Final = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}
)
ACTIVE_STATUSES: Final = frozenset({JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED})
_TRANSITIONS: Final[Mapping[JobStatus, frozenset[JobStatus]]] = {
    # A queued job has no worker yet, so its cancellation is final at once.
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.FAILED}),
    JobStatus.RUNNING: frozenset(
        {JobStatus.CANCEL_REQUESTED, JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.INTERRUPTED}
    ),
    # A request alone never sets CANCELLED on a started job: the worker must stop (or be
    # terminated); a worker that completes first may still be SUCCEEDED.
    JobStatus.CANCEL_REQUESTED: frozenset(
        {JobStatus.CANCELLED, JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.INTERRUPTED}
    ),
}

_ID_RE: Final = {prefix: re.compile(rf"{prefix}_[0-9a-f]{{32}}") for prefix in ("run", "art", "att", "op")}


def new_id(prefix: str) -> str:
    """A fresh opaque ID such as ``run_<32 hex>`` (prefixes: run, art, att, op)."""
    if prefix not in _ID_RE:
        raise ValueError(f"unknown id prefix {prefix!r}")
    return f"{prefix}_{uuid.uuid4().hex}"


def _is_id(prefix: str, value: object) -> bool:
    return isinstance(value, str) and _ID_RE[prefix].fullmatch(value) is not None


def is_run_id(value: object) -> bool:
    return _is_id("run", value)


def is_artifact_id(value: object) -> bool:
    return _is_id("art", value)


def check_transition(old: JobStatus, new: JobStatus) -> None:
    if new not in _TRANSITIONS.get(old, frozenset()):
        raise StateConflictError("illegal job state transition", context={"from": str(old), "to": str(new)})


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    campaign_id: str
    idempotency_key: str
    request_hash: str
    request: Mapping[str, Any]
    """The normalized job command (what the worker executes and the hash covers)."""
    kind: str
    status: JobStatus
    evidence_scope: tuple[str, ...]
    attempt_id: str | None
    generation: int
    progress: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None
    result: Mapping[str, Any] | None
    created_ns: int
    updated_ns: int
    started_ns: int | None
    finished_ns: int | None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    kind: str
    media_type: str
    schema: str | None
    sha256: str
    size_bytes: int
    rel_path: str
    """POSIX path relative to the artifact root (never absolute, never ``..``)."""
    data_role: str
    provenance: Mapping[str, Any]
    created_ns: int
    run_id: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "media_type": self.media_type,
            "schema": self.schema,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rel_path": self.rel_path,
            "data_role": self.data_role,
            "provenance": dict(self.provenance),
            "created_ns": self.created_ns,
            "run_id": self.run_id,
        }

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any]) -> ArtifactRecord:
        return cls(
            **{
                k: doc[k]
                for k in (
                    "artifact_id",
                    "kind",
                    "media_type",
                    "schema",
                    "sha256",
                    "size_bytes",
                    "rel_path",
                    "data_role",
                    "provenance",
                    "created_ns",
                    "run_id",
                )
            }
        )


@dataclass(frozen=True)
class Operation:
    op_id: str
    run_id: str
    attempt_id: str
    state: str
    payload: Mapping[str, Any] = field(repr=False)
    created_ns: int


_DDL: Final = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    request_json TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('QUEUED', 'RUNNING', 'CANCEL_REQUESTED', 'SUCCEEDED', 'FAILED',
                                           'CANCELLED', 'INTERRUPTED')),
    evidence_scope_json TEXT NOT NULL,
    attempt_id TEXT,
    generation INTEGER NOT NULL DEFAULT 0,
    progress_json TEXT,
    failure_json TEXT,
    result_json TEXT,
    created_ns INTEGER NOT NULL,
    updated_ns INTEGER NOT NULL,
    started_ns INTEGER,
    finished_ns INTEGER,
    UNIQUE (campaign_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS runs_by_status ON runs (status, created_ns);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    media_type TEXT NOT NULL,
    schema TEXT,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    rel_path TEXT NOT NULL UNIQUE,
    data_role TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    created_ns INTEGER NOT NULL,
    run_id TEXT REFERENCES runs (run_id)
);
CREATE INDEX IF NOT EXISTS artifacts_by_run ON artifacts (run_id);
CREATE TABLE IF NOT EXISTS operations (
    op_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    attempt_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('PENDING', 'DONE', 'ABORTED')),
    payload_json TEXT NOT NULL,
    created_ns INTEGER NOT NULL,
    updated_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs (run_id),
    from_status TEXT,
    to_status TEXT NOT NULL,
    at_ns INTEGER NOT NULL
);
"""


def _json(value: Any) -> str | None:
    return None if value is None else canonical_json(value)


def _load(text: str | None) -> Any:
    return None if text is None else loads_strict(text)


class JobStore:
    """Thread-safe facade over the service database (see the module docstring)."""

    def __init__(self, path: str | Path, *, clock: Callable[[], int] = time.time_ns) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise IncompatibleSchemaError(
                    "service database uses a newer schema", context={"got": version, "supported": SCHEMA_VERSION}
                )
            with self._tx() as conn:  # (executescript would commit implicitly; run statements one by one)
                for statement in filter(None, (s.strip() for s in _DDL.split(";"))):
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # ------------------------------------------------------------------ runs

    @staticmethod
    def _run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            campaign_id=row["campaign_id"],
            idempotency_key=row["idempotency_key"],
            request_hash=row["request_hash"],
            request=loads_strict(row["request_json"]),
            kind=row["kind"],
            status=JobStatus(row["status"]),
            evidence_scope=tuple(loads_strict(row["evidence_scope_json"])),
            attempt_id=row["attempt_id"],
            generation=int(row["generation"]),
            progress=_load(row["progress_json"]),
            failure=_load(row["failure_json"]),
            result=_load(row["result_json"]),
            created_ns=int(row["created_ns"]),
            updated_ns=int(row["updated_ns"]),
            started_ns=row["started_ns"],
            finished_ns=row["finished_ns"],
        )

    def _get(self, conn: sqlite3.Connection, run_id: str) -> RunRecord:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("run not found", context={"run_id": run_id})
        return self._run(row)

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            return self._get(self._conn, run_id)

    def find_run_by_key(self, campaign_id: str, idempotency_key: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE campaign_id = ? AND idempotency_key = ?", (campaign_id, idempotency_key)
            ).fetchone()
        return None if row is None else self._run(row)

    def runs_with_status(self, *statuses: JobStatus) -> list[RunRecord]:
        marks = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM runs WHERE status IN ({marks}) ORDER BY created_ns, rowid",
                tuple(str(s) for s in statuses),
            ).fetchall()
        return [self._run(r) for r in rows]

    def queued_runs(self, limit: int, *, exclude_campaigns: Iterable[str] = ()) -> list[RunRecord]:
        """The oldest QUEUED runs, leaving out ``exclude_campaigns`` in the query itself.

        The dispatcher excludes the campaigns that already have an active job: filtered in SQL,
        a campaign with a long backlog cannot fill a fixed window and hide the other campaigns'
        runs behind it (head-of-line blocking that left worker slots idle).
        """
        excluded = sorted(set(exclude_campaigns))
        clause = f" AND campaign_id NOT IN ({','.join('?' * len(excluded))})" if excluded else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM runs WHERE status = 'QUEUED'{clause} ORDER BY created_ns, rowid LIMIT ?",
                (*excluded, int(limit)),
            ).fetchall()
        return [self._run(r) for r in rows]

    def admit(
        self,
        *,
        campaign_id: str,
        idempotency_key: str,
        request_hash: str,
        request: Mapping[str, Any],
        kind: str,
        evidence_scope: Sequence[str],
        queue_capacity: int,
    ) -> tuple[RunRecord, bool]:
        """Create a QUEUED run, or return the existing run for the same key and request.

        Returns ``(run, replayed)``. Raises :class:`IdempotencyConflictError` for a reused key
        with another request and :class:`QueueFullError` when the queue is at capacity.
        """
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE campaign_id = ? AND idempotency_key = ?", (campaign_id, idempotency_key)
            ).fetchone()
            if row is not None:
                existing = self._run(row)
                if existing.request_hash != request_hash:
                    raise IdempotencyConflictError(
                        "idempotency key was used for a different request", context={"run_id": existing.run_id}
                    )
                return existing, True
            queued = int(conn.execute("SELECT COUNT(*) FROM runs WHERE status = 'QUEUED'").fetchone()[0])
            if queued >= queue_capacity:
                raise QueueFullError("admission queue is full", context={"capacity": queue_capacity})
            now = self._clock()
            run_id = new_id("run")
            conn.execute(
                "INSERT INTO runs (run_id, campaign_id, idempotency_key, request_hash, request_json, kind, status,"
                " evidence_scope_json, created_ns, updated_ns) VALUES (?, ?, ?, ?, ?, ?, 'QUEUED', ?, ?, ?)",
                (
                    run_id,
                    campaign_id,
                    idempotency_key,
                    request_hash,
                    canonical_json(dict(request)),
                    kind,
                    canonical_json(list(evidence_scope)),
                    now,
                    now,
                ),
            )
            self._log(conn, run_id, None, JobStatus.QUEUED, now)
            return self._get(conn, run_id), False

    def _log(self, conn: sqlite3.Connection, run_id: str, old: JobStatus | None, new: JobStatus, now: int) -> None:
        conn.execute(
            "INSERT INTO transitions (run_id, from_status, to_status, at_ns) VALUES (?, ?, ?, ?)",
            (run_id, None if old is None else str(old), str(new), now),
        )

    def _move(self, conn: sqlite3.Connection, rec: RunRecord, new: JobStatus, **columns: Any) -> RunRecord:
        check_transition(rec.status, new)
        now = self._clock()
        values = dict(columns)
        values["status"] = str(new)
        values["updated_ns"] = now
        if new in TERMINAL_STATUSES:
            values["finished_ns"] = now
        assignments = ", ".join(f"{k} = ?" for k in values)
        conn.execute(f"UPDATE runs SET {assignments} WHERE run_id = ?", (*values.values(), rec.run_id))
        self._log(conn, rec.run_id, rec.status, new, now)
        return self._get(conn, rec.run_id)

    def start_attempt(self, run_id: str) -> RunRecord:
        """QUEUED -> RUNNING with a fresh attempt ID and the next generation number."""
        with self._tx() as conn:
            rec = self._get(conn, run_id)
            return self._move(
                conn,
                rec,
                JobStatus.RUNNING,
                attempt_id=new_id("att"),
                generation=rec.generation + 1,
                started_ns=self._clock(),
                progress_json=_json({"phase": "starting", "completed_steps": 0, "total_steps": 0}),
            )

    def set_progress(self, run_id: str, attempt_id: str, progress: Mapping[str, Any]) -> bool:
        """Record worker progress; ignored (False) unless the attempt is the active one."""
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE runs SET progress_json = ?, updated_ns = ? WHERE run_id = ? AND attempt_id = ?"
                " AND status IN ('RUNNING', 'CANCEL_REQUESTED')",
                (canonical_json(dict(progress)), self._clock(), run_id, attempt_id),
            )
            return cur.rowcount == 1

    def transition(
        self, run_id: str, status: JobStatus, *, attempt_id: str | None = None, failure: Mapping[str, Any] | None = None
    ) -> RunRecord:
        """Move a run to ``status``; with ``attempt_id`` only if that attempt is still the active one."""
        with self._tx() as conn:
            rec = self._get(conn, run_id)
            if attempt_id is not None and rec.attempt_id != attempt_id:
                raise StateConflictError("stale attempt", context={"run_id": run_id})
            columns: dict[str, Any] = {}
            if failure is not None:
                columns["failure_json"] = canonical_json(dict(failure))
            return self._move(conn, rec, status, **columns)

    def request_cancel(self, run_id: str) -> RunRecord:
        """Idempotent: QUEUED -> CANCELLED, RUNNING -> CANCEL_REQUESTED, otherwise unchanged."""
        with self._tx() as conn:
            rec = self._get(conn, run_id)
            if rec.status is JobStatus.QUEUED:
                return self._move(conn, rec, JobStatus.CANCELLED)
            if rec.status is JobStatus.RUNNING:
                return self._move(conn, rec, JobStatus.CANCEL_REQUESTED)
            return rec

    def transitions(self, run_id: str) -> list[tuple[str | None, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT from_status, to_status FROM transitions WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ------------------------------------------------------------- artifacts

    @staticmethod
    def _artifact(row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            kind=row["kind"],
            media_type=row["media_type"],
            schema=row["schema"],
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            rel_path=row["rel_path"],
            data_role=row["data_role"],
            provenance=loads_strict(row["provenance_json"]),
            created_ns=int(row["created_ns"]),
            run_id=row["run_id"],
        )

    @staticmethod
    def _insert_artifacts(conn: sqlite3.Connection, records: Sequence[ArtifactRecord]) -> None:
        conn.executemany(
            "INSERT INTO artifacts (artifact_id, kind, media_type, schema, sha256, size_bytes, rel_path, data_role,"
            " provenance_json, created_ns, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    a.artifact_id,
                    a.kind,
                    a.media_type,
                    a.schema,
                    a.sha256,
                    a.size_bytes,
                    a.rel_path,
                    a.data_role,
                    canonical_json(dict(a.provenance)),
                    a.created_ns,
                    a.run_id,
                )
                for a in records
            ],
        )

    def add_artifacts(self, records: Sequence[ArtifactRecord]) -> None:
        with self._tx() as conn:
            self._insert_artifacts(conn, records)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self._lock:
            row = self._conn.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise NotFoundError("artifact not found", context={"artifact_id": artifact_id})
        return self._artifact(row)

    def artifacts_for_run(self, run_id: str) -> list[ArtifactRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY rel_path", (run_id,)
            ).fetchall()
        return [self._artifact(r) for r in rows]

    # ---------------------------------------------------------------- outbox

    @staticmethod
    def _operation(row: sqlite3.Row) -> Operation:
        return Operation(
            op_id=row["op_id"],
            run_id=row["run_id"],
            attempt_id=row["attempt_id"],
            state=row["state"],
            payload=loads_strict(row["payload_json"]),
            created_ns=int(row["created_ns"]),
        )

    def begin_publication(
        self, run_id: str, attempt_id: str, payload: Mapping[str, Any], *, op_id: str | None = None
    ) -> Operation:
        """Record a PENDING publication for the active attempt of a started, unfinished run."""
        with self._tx() as conn:
            rec = self._get(conn, run_id)
            if rec.status not in ACTIVE_STATUSES or rec.attempt_id != attempt_id:
                raise StateConflictError(
                    "publication rejected: attempt is not active", context={"run_id": run_id, "status": str(rec.status)}
                )
            if conn.execute("SELECT 1 FROM operations WHERE run_id = ? AND state = 'PENDING'", (run_id,)).fetchone():
                raise StateConflictError("a publication is already pending for this run", context={"run_id": run_id})
            now = self._clock()
            op_id = op_id or new_id("op")
            conn.execute(
                "INSERT INTO operations (op_id, run_id, attempt_id, kind, state, payload_json, created_ns,"
                " updated_ns) VALUES (?, ?, ?, 'publish', 'PENDING', ?, ?, ?)",
                (op_id, run_id, attempt_id, canonical_json(dict(payload)), now, now),
            )
            return self._operation(conn.execute("SELECT * FROM operations WHERE op_id = ?", (op_id,)).fetchone())

    def _get_op(self, conn: sqlite3.Connection, op_id: str) -> Operation:
        row = conn.execute("SELECT * FROM operations WHERE op_id = ?", (op_id,)).fetchone()
        if row is None:
            raise NotFoundError("operation not found", context={"op_id": op_id})
        return self._operation(row)

    def complete_publication(self, op_id: str) -> RunRecord:
        """Register the payload's artifacts, set SUCCEEDED and mark the operation DONE (idempotent)."""
        with self._tx() as conn:
            op = self._get_op(conn, op_id)
            if op.state == "DONE":
                return self._get(conn, op.run_id)
            if op.state != "PENDING":
                raise StateConflictError("publication was aborted", context={"op_id": op_id})
            rec = self._get(conn, op.run_id)
            if rec.attempt_id != op.attempt_id:
                raise StateConflictError("publication belongs to a stale attempt", context={"op_id": op_id})
            self._insert_artifacts(conn, [ArtifactRecord.from_mapping(a) for a in op.payload["artifacts"]])
            done = self._move(
                conn,
                rec,
                JobStatus.SUCCEEDED,
                result_json=_json(op.payload["result"]),
                progress_json=_json({"phase": "published", "completed_steps": 1, "total_steps": 1}),
            )
            conn.execute("UPDATE operations SET state = 'DONE', updated_ns = ? WHERE op_id = ?", (self._clock(), op_id))
            return done

    def abort_publication(self, op_id: str) -> None:
        with self._tx() as conn:
            op = self._get_op(conn, op_id)
            if op.state == "PENDING":
                conn.execute(
                    "UPDATE operations SET state = 'ABORTED', updated_ns = ? WHERE op_id = ?", (self._clock(), op_id)
                )

    def pending_operations(self) -> list[Operation]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM operations WHERE state = 'PENDING' ORDER BY created_ns").fetchall()
        return [self._operation(r) for r in rows]
