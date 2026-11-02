"""Service error types with stable codes and their HTTP status mapping.

All service failures are :class:`~financeai.core.errors.FinanceAIError` subclasses, so the
HTTP adapter, the CLI and Python callers share one error vocabulary. Core errors keep their
own codes and are mapped here: invalid configuration/data -> 422, incompatible schema -> 409,
missing optional dependency -> 503, invariant violation -> 500.
"""

from __future__ import annotations

from typing import ClassVar, Final

from financeai.core.errors import (
    FinanceAIError,
    IncompatibleSchemaError,
    InvalidConfigError,
    InvalidDataError,
    OptionalDependencyError,
    UnsupportedCapabilityError,
)


class ServiceError(FinanceAIError):
    """Base class of service-layer failures; ``http_status`` is part of the API contract."""

    code: ClassVar[str] = "service_error"
    http_status: ClassVar[int] = 500


class InvalidHostError(ServiceError):
    """The Host header is not in the configured allowlist (DNS-rebinding protection)."""

    code = "invalid_host"
    http_status = 400


class AuthenticationError(ServiceError):
    code = "unauthorized"
    http_status = 401


class ForbiddenDataRoleError(ServiceError):
    """The request would open data whose role this service may not use (e.g. a reserved test)."""

    code = "forbidden_data_role"
    http_status = 403


class NotFoundError(ServiceError):
    """Unknown, malformed or inaccessible run/artifact ID (never distinguishes the three)."""

    code = "not_found"
    http_status = 404


class IdempotencyConflictError(ServiceError):
    """The idempotency key was already used for a different normalized request."""

    code = "idempotency_conflict"
    http_status = 409


class StateConflictError(ServiceError):
    """An illegal job-state transition or a stale generation/attempt."""

    code = "state_conflict"
    http_status = 409


class RequestTooLargeError(ServiceError):
    code = "request_too_large"
    http_status = 413


class InvalidRequestError(ServiceError):
    """The request is well-formed JSON but violates the transport contract."""

    code = "invalid_request"
    http_status = 422


class QueueFullError(ServiceError):
    code = "queue_full"
    http_status = 429


class WorkerUnavailableError(ServiceError):
    code = "worker_unavailable"
    http_status = 503


_CORE_STATUS: Final[tuple[tuple[type[FinanceAIError], int], ...]] = (
    (IncompatibleSchemaError, 409),
    (InvalidConfigError, 422),
    (InvalidDataError, 422),
    (UnsupportedCapabilityError, 422),
    (OptionalDependencyError, 503),
)


def http_status_for(exc: FinanceAIError) -> int:
    """HTTP status of a typed error (unknown/internal failures are 500)."""
    if isinstance(exc, ServiceError):
        return exc.http_status
    for cls, status in _CORE_STATUS:
        if isinstance(exc, cls):
            return status
    return 500
