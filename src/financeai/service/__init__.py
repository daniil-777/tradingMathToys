"""Local, authenticated replay service over ``financeai.application`` and ``financeai.domains.market``.

The service executes only registered strategy configs against registered DEVELOPMENT or
SYNTHETIC tape caches (or deterministic synthetic markets) in bounded worker subprocesses.
It is not a research notebook, arbitrary-code runner, or route to SEALED_HOLDOUT data.
Importing :mod:`financeai.service` (including when unpickling :func:`financeai.service.worker.worker_main`
in a fresh worker process) never imports FastAPI, Pydantic, or NumPy before native thread limits are set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from financeai.service.errors import (
    AuthenticationError,
    ForbiddenDataRoleError,
    IdempotencyConflictError,
    InvalidHostError,
    InvalidRequestError,
    NotFoundError,
    QueueFullError,
    RequestTooLargeError,
    ServiceError,
    StateConflictError,
    WorkerUnavailableError,
    http_status_for,
)
from financeai.service.settings import ServiceSettings, is_loopback_host, read_token_file
from financeai.service.store import ArtifactRecord, JobStatus, JobStore, Operation, RunRecord

if TYPE_CHECKING:
    from financeai.service.artifacts import (
        ArtifactStore,
        register_strategy,
        register_tape_cache,
        tree_digest,
    )
    from financeai.service.coordinator import Coordinator
    from financeai.service.jobs import (
        StrategyReplayJob,
        SyntheticReplayJob,
        execute_job,
        job_from_mapping,
        request_hash,
    )

_ARTIFACT_EXPORTS = frozenset(
    {
        "ArtifactStore",
        "register_strategy",
        "register_tape_cache",
        "tree_digest",
    }
)
_JOB_EXPORTS = frozenset(
    {
        "StrategyReplayJob",
        "SyntheticReplayJob",
        "execute_job",
        "job_from_mapping",
        "request_hash",
    }
)

__all__ = [
    "ArtifactRecord",
    "ArtifactStore",
    "AuthenticationError",
    "Coordinator",
    "ForbiddenDataRoleError",
    "IdempotencyConflictError",
    "InvalidHostError",
    "InvalidRequestError",
    "JobStatus",
    "JobStore",
    "NotFoundError",
    "Operation",
    "QueueFullError",
    "RequestTooLargeError",
    "RunRecord",
    "ServiceError",
    "ServiceSettings",
    "StateConflictError",
    "StrategyReplayJob",
    "SyntheticReplayJob",
    "WorkerUnavailableError",
    "execute_job",
    "http_status_for",
    "is_loopback_host",
    "job_from_mapping",
    "read_token_file",
    "register_strategy",
    "register_tape_cache",
    "request_hash",
    "tree_digest",
]


def __getattr__(name: str) -> Any:
    if name == "Coordinator":
        from financeai.service.coordinator import Coordinator

        return Coordinator
    if name in _ARTIFACT_EXPORTS:
        from financeai.service import artifacts

        return getattr(artifacts, name)
    if name in _JOB_EXPORTS:
        from financeai.service import jobs

        return getattr(jobs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
