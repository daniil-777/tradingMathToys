"""Admission and publication coordinator: the single owner of job state and committed artifacts.

* **Admission** validates the job command (same validators as Python/CLI), resolves only
  registered artifacts, hashes the normalized request and inserts it into the bounded queue.
* **Dispatch** starts one ``spawn``-context process per attempt (at most ``max_workers``,
  at most one active job per campaign) with a private pipe and cancel event. Candidates are
  the oldest queued runs of campaigns *without* an active job, selected in SQL, so one
  campaign's backlog never hides another campaign's run while a slot is free.
* **Cancellation** is cooperative first (checkpoints), then bounded: after
  ``cancel_grace_s`` the worker's process group gets SIGTERM, then SIGKILL.
* **Publication** is owned here. Every decision about an attempt goes through
  :meth:`Coordinator.settle_attempt` under one lock, so the commit/cancel order is total:
  whatever the coordinator processes first wins. A completion is accepted only for the active
  attempt of a RUNNING/CANCEL_REQUESTED run whose generation was not invalidated (a forced
  termination invalidates it). Accepted output is re-hashed, checked against the expected
  file set, fsynced (every staged file and directory: the database is ``synchronous=FULL``,
  but a SUCCEEDED commit must never reference output bytes the OS has not written), recorded
  as a PENDING outbox operation, renamed atomically into the artifact store and committed
  (artifacts + SUCCEEDED) in one transaction. Rejected or partial output is moved to
  ``STATE_ROOT/quarantine`` and never registered.
* **Recovery** at start completes or aborts PENDING publications, turns RUNNING and
  CANCEL_REQUESTED runs into INTERRUPTED (they are never resubmitted) and quarantines stray
  staging directories. QUEUED runs stay queued.
* **Ownership**: recovery and dispatch require an exclusive ``flock`` on
  ``STATE_ROOT/service.lock``; a second coordinator on the same root fails with
  :class:`StateConflictError` instead of interrupting the owner's live runs.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import multiprocessing
import os
import pickle
import signal
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Final

from financeai.application.strategy import StrategyConfig
from financeai.core.errors import FinanceAIError, InvalidDataError
from financeai.core.serialization import canonical_json, loads_strict
from financeai.domains.market.cache import read_manifest
from financeai.service.artifacts import (
    ARTIFACT_KINDS,
    FORBIDDEN_ROLES,
    ArtifactStore,
    register_strategy,
    register_tape_cache,
    tree_digest,
)
from financeai.service.errors import (
    ForbiddenDataRoleError,
    IdempotencyConflictError,
    InvalidRequestError,
    StateConflictError,
    WorkerUnavailableError,
)
from financeai.service.jobs import (
    Job,
    StrategyReplayJob,
    SyntheticReplayJob,
    expected_outputs,
    job_from_mapping,
    request_hash,
    result_from_summary,
)
from financeai.service.settings import ServiceSettings
from financeai.service.store import (
    ACTIVE_STATUSES,
    ArtifactRecord,
    JobStatus,
    JobStore,
    Operation,
    RunRecord,
    new_id,
)
from financeai.service.worker import worker_main

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms are untested and unsupported
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger("financeai.service")

JobRunner = Callable[[Mapping[str, Any], Path, Callable[[str, int, int], None]], Mapping[str, Any]]
_KILL_GRACE_S: Final = 2.0
_MESSAGE_LIMIT: Final = 300
DISPATCH_ATTEMPTS_PER_SLOT: Final = 4
"""Launch attempts one :meth:`Coordinator._dispatch` call makes per worker slot before it yields the lock
(a failed launch is a synchronous=FULL write under the lock; an unbounded backlog of runs failing at input
resolution would otherwise stall settlement, cancellation and polling). The call wakes the next cycle."""
MANIFEST_SCHEMA: Final = "financeai.run_manifest"
_INTERRUPTED: Final = {
    "code": "interrupted",
    "message": "the service stopped while the run was active; it was not resubmitted",
}
_SHUTDOWN: Final = {
    "code": "service_shutdown",
    "message": "the service shut down before the run finished; it was not resubmitted",
}


def _failure(code: object, message: object) -> dict[str, str]:
    return {"code": str(code)[:64], "message": str(message)[:_MESSAGE_LIMIT]}


@dataclass
class _Active:
    run_id: str
    campaign_id: str
    attempt_id: str
    process: BaseProcess
    conn: Connection
    cancel_event: Any
    staging: Path
    outcome: tuple[str, dict[str, Any]] | None = None
    conn_open: bool = True
    cancel_deadline: float | None = None
    kill_deadline: float | None = None
    invalidated: bool = False
    """Cancellation won (the worker was terminated): any later completion is rejected."""
    stopping: bool = False


class Coordinator:
    """Owns the job store, the artifact store and the worker processes of one STATE_ROOT."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        job_runner: JobRunner | None = None,
        mp_context: BaseContext | None = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        self.settings = settings
        for directory in (settings.state_root, settings.artifacts_dir, settings.staging_dir, settings.quarantine_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(settings.db_path)
        self.artifacts = ArtifactStore(settings.artifacts_dir)
        # Typed loosely: typeshed declares Process/Pipe/Event on the concrete contexts only.
        self._ctx: Any = mp_context if mp_context is not None else multiprocessing.get_context("spawn")
        self._runner_blob = None if job_runner is None else pickle.dumps(job_runner)
        self._active: dict[str, _Active] = {}
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._poll_s = poll_interval_s
        self._failed: BaseException | None = None
        self._owner_fd: int | None = None

    # ------------------------------------------------------------ lifecycle

    def _acquire_ownership(self) -> None:
        if self._owner_fd is not None or fcntl is None:
            return
        fd = os.open(self.settings.state_root / "service.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise StateConflictError("another coordinator owns this state root") from None
        self._owner_fd = fd

    def _release_ownership(self) -> None:
        if self._owner_fd is not None:
            if fcntl is not None:
                fcntl.flock(self._owner_fd, fcntl.LOCK_UN)
            os.close(self._owner_fd)
            self._owner_fd = None

    def start(self) -> None:
        """Take ownership of STATE_ROOT, recover persisted state, then start the dispatch thread."""
        with self._lock:
            if self._thread is not None:
                return
            self.recover()
            self._thread = threading.Thread(target=self._loop, name="financeai-coordinator", daemon=True)
            self._thread.start()

    @property
    def available(self) -> bool:
        """Workers can be started (the thread runs, no stop requested, capacity configured)."""
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop.is_set() and self.settings.max_workers > 0

    @property
    def active_runs(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._active)

    def stop(self) -> None:
        """Stop dispatching; give active workers the grace period, then terminate them.

        Runs that do not finish (or cancel on request) within the grace period become
        INTERRUPTED; they are never resubmitted.
        """
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=60)
        with self._lock:
            for active in self._active.values():
                active.stopping = True
                active.cancel_event.set()
        deadline = time.monotonic() + self.settings.cancel_grace_s
        while self._active and time.monotonic() < deadline:
            self._poll_once(self._poll_s)
        with self._lock:
            for active in list(self._active.values()):
                self._terminate(active, force=True)
        deadline = time.monotonic() + 10.0
        while self._active and time.monotonic() < deadline:
            self._poll_once(self._poll_s)
        with self._lock:
            for active in list(self._active.values()):  # pragma: no cover - unkillable process
                self._active.pop(active.run_id, None)
                self.settle_attempt(
                    active.run_id, active.attempt_id, ("lost", {}), active.staging, invalidated=True, stopping=True
                )
            self.store.close()
            self._release_ownership()

    # ------------------------------------------------------------ admission

    def submit(self, campaign_id: str, idempotency_key: str, job_doc: Mapping[str, Any]) -> tuple[RunRecord, bool]:
        """Admit a job; returns ``(run, replayed)`` (see :meth:`JobStore.admit`)."""
        job = job_from_mapping(job_doc)
        digest = request_hash(campaign_id, job)
        existing = self.store.find_run_by_key(campaign_id, idempotency_key)
        if existing is not None:
            if existing.request_hash != digest:
                raise IdempotencyConflictError(
                    "idempotency key was used for a different request", context={"run_id": existing.run_id}
                )
            return existing, True
        scope = self._admission_scope(job)
        if not self.available:
            raise WorkerUnavailableError("no worker capacity is available")
        record, replayed = self.store.admit(
            campaign_id=campaign_id,
            idempotency_key=idempotency_key,
            request_hash=digest,
            request=job.to_mapping(),
            kind=job.kind,
            evidence_scope=scope,
            queue_capacity=self.settings.queue_capacity,
        )
        self._wake.set()
        return record, replayed

    def _input_artifact(self, artifact_id: str, kind: str) -> ArtifactRecord:
        record = self.store.get_artifact(artifact_id)
        if record.kind != kind:
            raise InvalidRequestError(
                "artifact has the wrong kind for this job",
                context={"artifact_id": artifact_id, "expected": kind, "got": record.kind},
            )
        if record.data_role in FORBIDDEN_ROLES:
            raise ForbiddenDataRoleError(
                "this service may not open data with this role",
                context={"artifact_id": artifact_id, "data_role": record.data_role},
            )
        return record

    def _admission_scope(self, job: Job) -> tuple[str, ...]:
        if isinstance(job, SyntheticReplayJob):
            return job.evidence_scope
        strategy = self._input_artifact(job.strategy_artifact_id, "strategy_config")
        caches = [self._input_artifact(a, "tape_cache") for a in job.cache_artifact_ids]
        covered = [str(c.provenance.get("instrument")) for c in caches]
        if len(set(covered)) != len(covered):
            raise InvalidRequestError("two caches cover the same instrument", context={"instruments": covered})
        missing = [k for k in strategy.provenance.get("instruments", []) if k not in covered]
        if missing:
            raise InvalidRequestError("caches do not cover the strategy instruments", context={"missing": missing})
        return tuple(sorted({c.data_role for c in caches}))

    def status(self, run_id: str) -> RunRecord:
        return self.store.get_run(run_id)

    def artifact(self, artifact_id: str) -> ArtifactRecord:
        return self.store.get_artifact(artifact_id)

    def run_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        return self.store.artifacts_for_run(run_id)

    def cancel(self, run_id: str) -> RunRecord:
        """Idempotent cancellation request (acknowledgement is not proof the worker stopped)."""
        with self._lock:
            record = self.store.request_cancel(run_id)
            active = self._active.get(run_id)
            if active is not None and record.status is JobStatus.CANCEL_REQUESTED:
                active.cancel_event.set()
                if active.cancel_deadline is None:
                    active.cancel_deadline = time.monotonic() + self.settings.cancel_grace_s
            return record

    def register_strategy(self, source: str | Path) -> ArtifactRecord:
        return register_strategy(self.store, self.artifacts, source)

    def register_tape_cache(self, source_dir: str | Path, *, data_role: str) -> ArtifactRecord:
        return register_tape_cache(self.store, self.artifacts, source_dir, data_role=data_role)

    # ------------------------------------------------------------- dispatch

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._dispatch()
                self._poll_once(self._poll_s)
                self._enforce_deadlines()
        except BaseException as exc:  # the service reports NOT_READY / 503 from here on
            self._failed = exc
            log.exception("coordinator loop stopped")

    def _dispatch(self) -> None:
        attempts = DISPATCH_ATTEMPTS_PER_SLOT * self.settings.max_workers
        with self._lock:
            while not self._stop.is_set():
                free = self.settings.max_workers - len(self._active)
                if free <= 0:
                    return
                busy = {a.campaign_id for a in self._active.values()}
                # Busy campaigns are excluded in the query (not skipped from a fixed window): a campaign
                # with a long backlog would otherwise fill the window and starve every other campaign.
                candidates = self.store.queued_runs(limit=free, exclude_campaigns=busy)
                attempted = False
                for record in candidates:
                    if record.campaign_id in busy:
                        continue  # a second queued run of a campaign launched in this batch
                    attempted = True
                    attempts -= 1
                    if self._launch(record):  # otherwise the run was FAILED: it left the queue either way
                        busy.add(record.campaign_id)
                if not attempted:
                    return  # nothing dispatchable is queued
                if attempts <= 0:
                    self._wake.set()  # the work is bounded per call: the next cycle continues the backlog
                    return

    def _resolve_inputs(self, record: RunRecord) -> dict[str, Any]:
        if record.kind != StrategyReplayJob.kind:
            return {}
        job = StrategyReplayJob.from_mapping(record.request)

        def entry(a: ArtifactRecord) -> dict[str, Any]:
            return {
                "id": a.artifact_id,
                "path": str(self.artifacts.path_of(a.rel_path)),
                "sha256": a.sha256,
                "size_bytes": a.size_bytes,
            }

        return {
            "strategy": entry(self._input_artifact(job.strategy_artifact_id, "strategy_config")),
            "caches": [entry(self._input_artifact(c, "tape_cache")) for c in job.cache_artifact_ids],
        }

    def _launch(self, record: RunRecord) -> bool:
        try:
            inputs = self._resolve_inputs(record)  # roles are re-checked at start, not only at admission
        except FinanceAIError as exc:
            self.store.transition(record.run_id, JobStatus.FAILED, failure=_failure(exc.code, exc.message))
            return False
        running = self.store.start_attempt(record.run_id)
        assert running.attempt_id is not None
        staging = self.settings.staging_dir / running.run_id / running.attempt_id
        staging.mkdir(parents=True)
        payload = {
            "run_id": running.run_id,
            "attempt_id": running.attempt_id,
            "generation": running.generation,
            "job": dict(running.request),
            "inputs": inputs,
        }
        receiver, sender = self._ctx.Pipe(duplex=False)
        cancel_event = self._ctx.Event()
        process = self._ctx.Process(
            target=worker_main,
            name=f"financeai-worker-{running.run_id[4:16]}",
            args=(payload, str(staging), cancel_event, sender, self._runner_blob),
            daemon=True,
        )
        try:
            process.start()
        except Exception as exc:
            receiver.close()
            sender.close()
            log.error("could not start a worker for %s: %s", running.run_id, type(exc).__name__)
            self.store.transition(
                running.run_id,
                JobStatus.FAILED,
                attempt_id=running.attempt_id,
                failure=_failure("worker_unavailable", "could not start a worker process"),
            )
            self._quarantine(staging, running.run_id, running.attempt_id, "spawn-failed")
            return False
        sender.close()  # the child owns the write end; EOF then signals its exit
        self._active[running.run_id] = _Active(
            run_id=running.run_id,
            campaign_id=running.campaign_id,
            attempt_id=running.attempt_id,
            process=process,
            conn=receiver,
            cancel_event=cancel_event,
            staging=staging,
        )
        log.info("run %s attempt %s started (pid %s)", running.run_id, running.attempt_id, process.pid)
        return True

    def _poll_once(self, timeout: float) -> None:
        with self._lock:
            actives = list(self._active.values())
        if not actives:
            self._wake.wait(timeout)
            self._wake.clear()
            return
        waitables: list[Any] = [a.conn for a in actives if a.conn_open]
        waitables += [a.process.sentinel for a in actives]
        ready = set(wait_for(waitables, timeout))
        for active in actives:
            self._drain(active)
            if active.process.sentinel in ready:
                # Clean up the worker's process group *before* reaping it, so the PID (= PGID)
                # cannot have been reused by an unrelated process.
                self._signal_group(active, signal.SIGKILL, fallback=False)
                active.process.join(timeout=5)
                self._drain(active)
                self._finalize(active)

    def _drain(self, active: _Active) -> None:
        while active.conn_open:
            try:
                if not active.conn.poll():
                    return
                message = active.conn.recv()
            except (EOFError, OSError):
                active.conn_open = False
                active.conn.close()
                return
            if not (isinstance(message, tuple) and len(message) == 2 and isinstance(message[1], dict)):
                active.outcome = ("failed", _failure("worker_protocol_error", "malformed worker message"))
                continue
            kind, data = message
            if kind == "progress":
                progress = {
                    "phase": str(data.get("phase", ""))[:64],
                    "completed_steps": int(data.get("completed_steps", 0)),
                    "total_steps": int(data.get("total_steps", 0)),
                }
                with self._lock:
                    self.store.set_progress(active.run_id, active.attempt_id, progress)
            elif kind in ("completed", "failed", "cancelled") and active.outcome is None:
                active.outcome = (kind, data)

    def _finalize(self, active: _Active) -> None:
        with self._lock:
            self._active.pop(active.run_id, None)
            if active.conn_open:
                active.conn_open = False
                active.conn.close()
            outcome = active.outcome or ("crashed", {"exitcode": active.process.exitcode})
            try:
                self.settle_attempt(
                    active.run_id,
                    active.attempt_id,
                    outcome,
                    active.staging,
                    invalidated=active.invalidated,
                    stopping=active.stopping,
                )
            finally:
                self._wake.set()

    def _signal_group(self, active: _Active, sig: int, *, fallback: bool = True) -> None:
        pid = active.process.pid
        if pid is None:
            return
        try:
            os.killpg(pid, sig)  # the worker called setsid(): its PGID is its PID
            return
        except (ProcessLookupError, PermissionError, AttributeError, OSError):
            pass
        if fallback:
            with contextlib.suppress(OSError):
                if sig == signal.SIGKILL:
                    active.process.kill()
                else:
                    active.process.terminate()

    def _terminate(self, active: _Active, *, force: bool = False) -> None:
        active.invalidated = True
        self._signal_group(active, signal.SIGKILL if force else signal.SIGTERM)
        active.kill_deadline = None if force else time.monotonic() + _KILL_GRACE_S

    def _enforce_deadlines(self) -> None:
        now = time.monotonic()
        with self._lock:
            for active in list(self._active.values()):
                if active.cancel_deadline is not None and now >= active.cancel_deadline and not active.invalidated:
                    log.warning("run %s did not stop within the grace period; terminating", active.run_id)
                    self._terminate(active)
                elif active.kill_deadline is not None and now >= active.kill_deadline:
                    self._terminate(active, force=True)

    # ---------------------------------------------------------- publication

    def settle_attempt(
        self,
        run_id: str,
        attempt_id: str,
        outcome: tuple[str, Mapping[str, Any]],
        staging: Path,
        *,
        invalidated: bool = False,
        stopping: bool = False,
    ) -> RunRecord:
        """The single commit/cancel decision point for an attempt's outcome.

        ``outcome`` is ``(kind, data)`` with kind ``completed``, ``failed``, ``cancelled``,
        ``crashed`` or ``lost``. Output of a stale attempt, of a terminal run or of an
        invalidated generation is quarantined and never published.
        """
        kind, data = outcome
        with self._lock:
            record = self.store.get_run(run_id)
            if record.attempt_id != attempt_id or record.status not in ACTIVE_STATUSES:
                log.warning("rejected late %s output of run %s attempt %s", kind, run_id, attempt_id)
                self._quarantine(staging, run_id, attempt_id, "stale")
                return record
            if kind == "completed" and not invalidated:
                try:
                    return self._publish(record, staging, data)
                except FinanceAIError as exc:
                    log.error("publication of run %s rejected: %s", run_id, exc.code)
                    self._quarantine(staging, run_id, attempt_id, "rejected")
                    return self.store.transition(
                        run_id, JobStatus.FAILED, attempt_id=attempt_id, failure=_failure(exc.code, exc.message)
                    )
            self._quarantine(staging, run_id, attempt_id, kind)
            cancelled = record.status is JobStatus.CANCEL_REQUESTED
            if cancelled and (kind == "cancelled" or invalidated):
                failure = (
                    None
                    if kind == "cancelled"
                    else _failure("terminated", "the worker was terminated after the cancellation grace period")
                )
                return self.store.transition(run_id, JobStatus.CANCELLED, attempt_id=attempt_id, failure=failure)
            if stopping:
                return self.store.transition(run_id, JobStatus.INTERRUPTED, attempt_id=attempt_id, failure=_SHUTDOWN)
            if kind == "failed":
                failure = _failure(data.get("code", "worker_error"), data.get("message", ""))
            elif kind == "cancelled":
                failure = _failure("worker_aborted", "the worker stopped without a cancellation request")
            else:
                failure = _failure(
                    "worker_crashed", f"the worker exited unexpectedly (exit code {data.get('exitcode')})"
                )
            return self.store.transition(run_id, JobStatus.FAILED, attempt_id=attempt_id, failure=failure)

    def _check_staging(self, job: Job, staging: Path, descriptor: Mapping[str, Any]) -> list[dict[str, Any]]:
        expected = dict(expected_outputs(job))
        outputs = descriptor.get("outputs")
        if not isinstance(outputs, list) or not all(isinstance(o, dict) for o in outputs):
            raise InvalidDataError("worker descriptor has no output list")
        declared = {o.get("rel_path"): o for o in outputs}
        if set(declared) != set(expected) or len(declared) != len(outputs):
            raise InvalidDataError(
                "worker outputs differ from the expected set",
                context={"expected": sorted(expected), "got": sorted(map(str, declared))},
            )
        top = {p.name for p in staging.iterdir()}
        if top - {"run", "caches"}:
            raise InvalidDataError("unexpected files in the attempt directory", context={"names": sorted(top)})
        checked: list[dict[str, Any]] = []
        for rel_path, kind in expected.items():
            item = declared[rel_path]
            digest, size = tree_digest(staging / rel_path)
            if item.get("kind") != kind or item.get("sha256") != digest or item.get("size_bytes") != size:
                raise InvalidDataError(
                    "staged output does not match the worker descriptor", context={"output": rel_path}
                )
            checked.append({"rel_path": rel_path, "kind": kind, "sha256": digest, "size_bytes": size})
        listed = _regular_files(staging)
        allowed = {r for r in expected if not (staging / r).is_dir()}
        allowed |= {f for f in listed for r in expected if (staging / r).is_dir() and f.startswith(r + "/")}
        if listed - allowed:
            raise InvalidDataError(
                "unexpected files in the attempt directory", context={"names": sorted(listed - allowed)[:10]}
            )
        return checked

    def _publish(self, record: RunRecord, staging: Path, descriptor: Mapping[str, Any]) -> RunRecord:
        assert record.attempt_id is not None
        job = job_from_mapping(record.request)
        outputs = self._check_staging(job, staging, descriptor)
        summary = loads_strict((staging / "run" / "summary.json").read_bytes())
        result = result_from_summary(summary)
        base = f"runs/{record.run_id}/{record.attempt_id}"
        role = "DEVELOPMENT" if "DEVELOPMENT" in record.evidence_scope else "SYNTHETIC"
        now = time.time_ns()
        provenance: dict[str, Any] = {
            "producer": "run",
            "run_id": record.run_id,
            "attempt_id": record.attempt_id,
            "job_kind": record.kind,
            "campaign_id": record.campaign_id,
        }
        if isinstance(job, StrategyReplayJob):
            provenance["input_artifact_ids"] = [job.strategy_artifact_id, *job.cache_artifact_ids]
        else:
            descriptor_doc = job.market.descriptor()
            provenance["generator"] = {**descriptor_doc, "start_ns": str(descriptor_doc["start_ns"])}
        records = [
            ArtifactRecord(
                artifact_id=new_id("art"),
                kind=o["kind"],
                media_type=ARTIFACT_KINDS[o["kind"]][0],
                schema=ARTIFACT_KINDS[o["kind"]][1],
                sha256=o["sha256"],
                size_bytes=o["size_bytes"],
                rel_path=f"{base}/{o['rel_path']}",
                data_role=role,
                provenance={**provenance, **_content_facts(o["kind"], staging / o["rel_path"])},
                created_ns=now,
                run_id=record.run_id,
            )
            for o in outputs
        ]
        op_id = new_id("op")
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": "1.0",
            "op_id": op_id,
            "run_id": record.run_id,
            "attempt_id": record.attempt_id,
            "generation": record.generation,
            "campaign_id": record.campaign_id,
            "job": dict(record.request),
            "result": result,
            "artifacts": [
                {
                    "artifact_id": r.artifact_id,
                    "kind": r.kind,
                    "path": o["rel_path"],
                    "sha256": r.sha256,
                    "size_bytes": r.size_bytes,
                }
                for r, o in zip(records, outputs, strict=True)
            ],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(canonical_json(manifest) + "\n")
        _fsync_tree(staging)  # worker output and the manifest are durable before the commit can reference them
        digest, size = tree_digest(manifest_path)
        media, schema = ARTIFACT_KINDS["run_manifest"]
        records.append(
            ArtifactRecord(
                artifact_id=new_id("art"),
                kind="run_manifest",
                media_type=media,
                schema=schema,
                sha256=digest,
                size_bytes=size,
                rel_path=f"{base}/manifest.json",
                data_role=role,
                provenance=provenance,
                created_ns=now,
                run_id=record.run_id,
            )
        )
        payload = {
            "artifacts": [r.to_mapping() for r in records],
            "result": result,
            "manifest_rel_path": f"{base}/manifest.json",
        }
        operation = self.store.begin_publication(record.run_id, record.attempt_id, payload, op_id=op_id)
        self._move_into_place(operation, staging)
        return self.store.complete_publication(operation.op_id)

    def _move_into_place(self, operation: Operation, staging: Path) -> None:
        final = self.artifacts.path_of(f"runs/{operation.run_id}/{operation.attempt_id}")
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(final)  # atomic on one filesystem; the target must not exist
            _fsync_path(final.parent)
        except OSError as exc:
            self.store.abort_publication(operation.op_id)
            raise InvalidDataError("could not move the attempt into the artifact store") from exc
        with contextlib.suppress(OSError):
            staging.parent.rmdir()  # drop the now-empty staging/<run_id> directory

    # ------------------------------------------------------------- recovery

    def recover(self) -> None:
        """Reconcile the database with the filesystem after a (possibly unclean) stop."""
        with self._lock:
            self._acquire_ownership()
            for operation in self.store.pending_operations():
                self._recover_operation(operation)
            for record in self.store.runs_with_status(JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED):
                self.store.transition(record.run_id, JobStatus.INTERRUPTED, failure=_INTERRUPTED)
                if record.attempt_id is not None:
                    self._quarantine(
                        self.settings.staging_dir / record.run_id / record.attempt_id,
                        record.run_id,
                        record.attempt_id,
                        "interrupted",
                    )
                log.warning("run %s was active at shutdown: INTERRUPTED (not resubmitted)", record.run_id)
            for run_dir in sorted(self.settings.staging_dir.iterdir()):
                for attempt_dir in sorted(run_dir.iterdir()) if run_dir.is_dir() else ():
                    self._quarantine(attempt_dir, run_dir.name, attempt_dir.name, "orphaned")
                with contextlib.suppress(OSError):
                    run_dir.rmdir()

    def _recover_operation(self, operation: Operation) -> None:
        rel = str(operation.payload.get("manifest_rel_path", ""))
        complete = False
        try:
            manifest_path = self.artifacts.path_of(rel)
            if manifest_path.is_file():
                doc = loads_strict(manifest_path.read_bytes())
                if isinstance(doc, dict) and doc.get("op_id") == operation.op_id:
                    for artifact in operation.payload["artifacts"]:
                        self.artifacts.verify(ArtifactRecord.from_mapping(artifact))
                    complete = True
        except FinanceAIError as exc:
            log.error("publication %s cannot be verified: %s", operation.op_id, exc.code)
        if complete:
            self.store.complete_publication(operation.op_id)
            log.info("completed pending publication %s of run %s", operation.op_id, operation.run_id)
            return
        self.store.abort_publication(operation.op_id)
        final = self.artifacts.root / "runs" / operation.run_id / operation.attempt_id
        self._quarantine(final, operation.run_id, operation.attempt_id, "unverified")

    def _quarantine(self, path: Path, run_id: str, attempt_id: str, reason: str) -> None:
        if not path.exists():
            return
        target = self.settings.quarantine_dir / run_id / f"{attempt_id}-{reason}-{uuid.uuid4().hex[:8]}"
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        with contextlib.suppress(OSError):
            path.parent.rmdir()  # drop an empty staging/<run_id> directory


def _content_facts(kind: str, path: Path) -> dict[str, Any]:
    """Per-artifact provenance read from the staged content itself (used by later admissions)."""
    if kind == "tape_cache":
        instrument = read_manifest(path)["instrument"]
        return {"instrument": f"{instrument['venue']}/{instrument['symbol']}"}
    if kind == "strategy_config":
        cfg = StrategyConfig.from_json(path.read_bytes())
        return {"strategy_name": cfg.name, "instruments": list(cfg.instruments)}
    return {}


def _regular_files(root: Path) -> set[str]:
    """POSIX paths (relative to ``root``) of every regular file below it, refusing anything else.

    ``lstat`` based: a symlink, FIFO, socket or device anywhere in the tree raises
    :class:`InvalidDataError` (-> the attempt is quarantined) instead of being overlooked, which
    is what ``rglob`` + ``is_file`` did for a FIFO that :func:`_fsync_tree` then blocked on.
    """
    listed: set[str] = set()
    pending = [root]
    while pending:
        here = pending.pop()
        with os.scandir(here) as entries:
            for entry in entries:
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(mode):
                    listed.add(Path(entry.path).relative_to(root).as_posix())
                else:
                    raise InvalidDataError(
                        "unexpected special file or link in the attempt directory",
                        context={"name": Path(entry.path).relative_to(root).as_posix()},
                    )
    return listed


def _fsync_path(path: Path) -> None:
    """``fsync`` one regular file or directory (a directory sync makes its entries durable).

    Opened without following a link and without blocking, then checked with ``fstat``: a FIFO
    would otherwise block the open (and the coordinator lock) until a writer appears.
    """
    if os.name != "posix":  # pragma: no cover
        return
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError as exc:
        if exc.errno != errno.ELOOP:  # a real filesystem fault keeps its OSError (callers abort on it)
            raise
        raise InvalidDataError("cannot sync a symbolic link", context={"name": path.name}) from exc
    try:
        mode = os.fstat(fd).st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise InvalidDataError("cannot sync a special file", context={"name": path.name})
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    """``fsync`` every file and directory under ``root`` (files before their directories, ``root`` last).

    Workers write their outputs with plain buffered I/O and the coordinator commits SUCCEEDED
    with SQLite ``synchronous=FULL``: without this pass, an OS crash after the commit could
    leave a durable SUCCEEDED run whose published files (or cache columns) are empty or
    truncated, and recovery only checks PENDING operations.
    """
    for current, _dirs, files in os.walk(root, topdown=False):  # bottom-up: subdirectories are synced first
        here = Path(current)
        for name in files:
            _fsync_path(here / name)
        _fsync_path(here)
