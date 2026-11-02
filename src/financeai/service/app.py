"""FastAPI adapter of the local research job API v1 (``create_app``).

Deployment defaults: loopback bind, one Uvicorn worker, a local random bearer token, an
explicit Host allowlist (DNS-rebinding protection), no CORS, no docs UI and no live OpenAPI
route (the versioned snapshot lives in ``docs/api/openapi-v1.json``). ``/healthz`` is the
only unauthenticated route and reveals no paths or configuration.

Every error uses ``{"error": {"code", "message", "request_id", "details"}}`` with bounded,
sanitised details: 400 bad Host, 401 token, 403 data role, 404 unknown/malformed ID,
409 idempotency/schema/state conflict, 413 body too large, 422 invalid request/config,
429 queue full, 503 workers unavailable, 500 unexpected (logged with a correlation ID).
Request bodies are read with a byte bound and parsed with ``loads_strict`` (duplicate keys
and non-finite numbers rejected) before Pydantic validation. Handlers only admit, read or
cancel jobs; replays run in worker processes owned by the :class:`Coordinator`.
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import re
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Final

from financeai.core.errors import FinanceAIError, OptionalDependencyError

try:
    import anyio
    from fastapi import Depends, FastAPI, Header, Request, Response
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from fastapi.routing import APIRoute
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
except ImportError as exc:  # pragma: no cover - exampled only without the extra
    raise OptionalDependencyError("the local API needs the 'api' extra: pip install 'financeai[api]'") from exc

from financeai.core.serialization import loads_strict
from financeai.service.coordinator import Coordinator, JobRunner
from financeai.service.errors import (
    AuthenticationError,
    InvalidHostError,
    NotFoundError,
    RequestTooLargeError,
    WorkerUnavailableError,
    http_status_for,
)
from financeai.service.models import (
    API_VERSION,
    IDEMPOTENCY_KEY_PATTERN,
    ArtifactMetadata,
    ErrorEnvelope,
    Failure,
    Health,
    Progress,
    RunAccepted,
    RunRequest,
    RunResult,
    RunStatus,
    job_from_domain,
)
from financeai.service.settings import MIN_TOKEN_LENGTH, ServiceSettings
from financeai.service.store import (
    TERMINAL_STATUSES,
    ArtifactRecord,
    JobStatus,
    RunRecord,
    is_artifact_id,
    is_run_id,
)

log = logging.getLogger("financeai.service")

OPENAPI_VERSION: Final = "1.0.0"
_DETAIL_ITEMS: Final = 16
_DETAIL_TEXT: Final = 200
_DETAIL_DEPTH: Final = 4
_PATH_LIKE: Final = re.compile(r"^(/|~|[A-Za-z]:[\\/])|[\\/](Users|home|private|tmp|var)[\\/]")
_SECRET_KEYS: Final = frozenset({"path", "file", "dir", "directory", "token", "authorization", "secret"})
_ERROR_DESCRIPTIONS: Final = {
    400: "Host header not allowed",
    401: "Missing or invalid bearer token",
    403: "Forbidden data role",
    404: "Unknown, malformed or inaccessible ID",
    409: "Idempotency, schema or state conflict",
    413: "Request body too large",
    422: "Invalid request or configuration",
    429: "Admission queue is full",
    503: "Workers unavailable",
    500: "Unexpected internal error (see correlation_id)",
}


# ---------------------------------------------------------------- errors


def _request_id(scope: Mapping[str, Any]) -> str:
    state = scope.get("state")
    rid = state.get("request_id") if isinstance(state, Mapping) else None
    return rid if isinstance(rid, str) else "unknown"


def _sanitise(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return "<redacted>" if _PATH_LIKE.search(value) else value[:_DETAIL_TEXT]
    if isinstance(value, list | tuple) and depth < _DETAIL_DEPTH:
        return [_sanitise(v, depth + 1) for v in list(value)[:_DETAIL_ITEMS]]
    if isinstance(value, Mapping) and depth < _DETAIL_DEPTH:
        return {
            str(k)[:64]: _sanitise(v, depth + 1)
            for k, v in list(value.items())[:_DETAIL_ITEMS]
            if str(k).lower() not in _SECRET_KEYS
        }
    return str(value)[:_DETAIL_TEXT] if not isinstance(value, bytes) else "<bytes>"


def error_body(code: str, message: str, request_id: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": {"code": code, "message": message, "request_id": request_id, "details": _sanitise(dict(details or {}))}
    }


def _error_response(
    status: int,
    code: str,
    message: str,
    request: Request,
    details: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        error_body(code, message, _request_id(request.scope), details), status_code=status, headers=dict(headers or {})
    )


async def _on_finance_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, FinanceAIError)
    status = http_status_for(exc)
    if status >= 500:
        log.error("request %s failed: %s", _request_id(request.scope), exc.code)
        return _error_response(
            status,
            exc.code,
            "internal error" if status == 500 else exc.message,
            request,
            {"correlation_id": _request_id(request.scope)},
        )
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return _error_response(status, exc.code, exc.message, request, dict(exc.context), headers)


async def _on_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {
            "loc": [str(p) for p in e.get("loc", ())],
            "type": str(e.get("type", "")),
            "msg": str(e.get("msg", ""))[:_DETAIL_TEXT],
        }
        for e in list(exc.errors())[:20]
    ]
    return _error_response(
        422, "invalid_request", "request does not match the API contract", request, {"errors": errors}
    )


async def _on_http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    message = {404: "not found", 405: "method not allowed"}.get(exc.status_code, "request failed")
    return _error_response(exc.status_code, code, message, request, None, exc.headers)


async def _on_unexpected(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request.scope)
    log.error("unhandled error (correlation_id=%s)", rid, exc_info=exc)
    return _error_response(500, "internal_error", "internal error", request, {"correlation_id": rid})


# ------------------------------------------------------------ middleware


def _host_name(raw: str) -> str:
    raw = raw.strip().lower()
    if raw.startswith("["):
        return raw[1 : raw.find("]")] if "]" in raw else raw
    return raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw


class _GuardMiddleware:
    """Request IDs, Host allowlist, bearer token on ``/v1/*`` and a declared-size precheck."""

    def __init__(self, app: ASGIApp, settings: ServiceSettings) -> None:
        self.app = app
        self.settings = settings
        self._token = settings.token.encode()
        self._hosts = {h.strip("[]") for h in settings.allowed_hosts}

    async def _reject(
        self, send: Send, status: int, error: FinanceAIError, rid: str, headers: Mapping[str, str] | None = None
    ) -> None:
        body = json.dumps(error_body(error.code, error.message, rid)).encode()
        raw = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-request-id", rid.encode()),
            (b"cache-control", b"no-store"),
        ]
        raw += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        await send({"type": "http.response.start", "status": status, "headers": raw})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = uuid.uuid4().hex
        scope["state"] = {**dict(scope.get("state") or {}), "request_id": rid}
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if _host_name(headers.get("host", "")) not in self._hosts:
            await self._reject(send, 400, InvalidHostError("host not allowed"), rid)
            return
        if str(scope.get("path", "")).startswith("/v1/"):
            scheme, _, credentials = headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(credentials.strip().encode(), self._token):
                await self._reject(
                    send,
                    401,
                    AuthenticationError("missing or invalid bearer token"),
                    rid,
                    {"WWW-Authenticate": "Bearer"},
                )
                return
        declared = headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.settings.max_body_bytes):
            await self._reject(send, 413, RequestTooLargeError("request body too large"), rid)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                extra = [
                    (b"x-request-id", rid.encode()),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _StrictRequest(Request):
    """A request whose body is size-bounded and whose JSON is parsed by ``loads_strict``."""

    async def body(self) -> bytes:
        if not hasattr(self, "_body"):
            limit = self.scope["app"].state.settings.max_body_bytes
            chunks: list[bytes] = []
            total = 0
            async for chunk in self.stream():
                total += len(chunk)
                if total > limit:
                    raise RequestTooLargeError("request body too large", context={"limit_bytes": limit})
                chunks.append(chunk)
            self._body = b"".join(chunks)
        return self._body

    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            self._json = loads_strict(await self.body())
        return self._json


class _ServiceRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def strict_handler(request: Request) -> Response:
            strict = _StrictRequest(request.scope, request.receive)
            if request.method in {"POST", "PUT", "PATCH"} and await strict.body():
                await strict.json()  # duplicate keys / NaN / malformed JSON -> 422 before validation
            return await handler(strict)

        return strict_handler


# ---------------------------------------------------------------- routes

_bearer = HTTPBearer(auto_error=False, description="Local random bearer token (FINANCEAI_API_TOKEN or token file)")


def _require_token(
    request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
) -> None:
    token = request.app.state.settings.token.encode()
    if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), token):
        raise AuthenticationError("missing or invalid bearer token")


def _responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {code: {"model": ErrorEnvelope, "description": _ERROR_DESCRIPTIONS[code]} for code in codes}


def _coordinator(request: Request) -> Coordinator:
    coordinator = request.app.state.coordinator
    if not isinstance(coordinator, Coordinator):
        raise WorkerUnavailableError("the service is not running")
    return coordinator


def economic_status(record: RunRecord) -> str:
    if record.status is JobStatus.SUCCEEDED and record.result is not None:
        return "VALID_ACCOUNT" if record.result.get("account_valid") else "INVALID_ACCOUNT"
    if record.status in TERMINAL_STATUSES and record.started_ns is not None:
        return "INCOMPLETE"
    return "NOT_EVALUATED"


def _clock(value: int | None) -> str | None:
    return None if value is None else str(value)


def run_status(record: RunRecord, artifacts: list[ArtifactRecord]) -> RunStatus:
    url = f"/v1/runs/{record.run_id}"
    result = None
    if record.result is not None:
        r = record.result
        clocks = {"first_event_ns": str(r["first_event_ns"]), "last_event_ns": str(r["last_event_ns"])}
        result = RunResult(**{**r, **clocks})
    return RunStatus(
        run_id=record.run_id,
        campaign_id=record.campaign_id,
        kind=record.kind,  # type: ignore[arg-type]
        status=record.status.value,
        attempt_id=record.attempt_id,
        generation=record.generation,
        job=job_from_domain(record.request),
        progress=None if record.progress is None else Progress(**record.progress),
        failure=None if record.failure is None else Failure(**record.failure),
        result=result,
        artifact_ids=[a.artifact_id for a in artifacts],
        evidence_scope=list(record.evidence_scope),  # type: ignore[arg-type]
        economic_status=economic_status(record),  # type: ignore[arg-type]
        created_at_ns=str(record.created_ns),
        updated_at_ns=str(record.updated_ns),
        started_at_ns=_clock(record.started_ns),
        finished_at_ns=_clock(record.finished_ns),
        status_url=url,
        cancel_url=f"{url}/cancel",
    )


def artifact_metadata(record: ArtifactRecord) -> ArtifactMetadata:
    return ArtifactMetadata(
        artifact_id=record.artifact_id,
        kind=record.kind,
        media_type=record.media_type,
        schema_type=record.schema or "",
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        data_role=record.data_role,
        run_id=record.run_id,
        created_at_ns=str(record.created_ns),
        provenance=dict(record.provenance),
    )


def create_app(
    settings: ServiceSettings, *, job_runner: JobRunner | None = None, mp_context: Any | None = None
) -> FastAPI:
    """Build the API. The coordinator (database, workers) starts with the ASGI lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        coordinator = await anyio.to_thread.run_sync(
            lambda: Coordinator(settings, job_runner=job_runner, mp_context=mp_context)
        )
        await anyio.to_thread.run_sync(coordinator.start)
        app.state.coordinator = coordinator
        try:
            yield
        finally:
            app.state.coordinator = None
            await anyio.to_thread.run_sync(coordinator.stop)

    app = FastAPI(
        title="FinanceAI local research API",
        version=OPENAPI_VERSION,
        lifespan=lifespan,
        description="Local, single-user research job API (loopback, bearer token). Submits bounded "
        "replay jobs to worker processes and reports their durable state and artifacts.",
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
        separate_input_output_schemas=False,
    )
    app.router.route_class = _ServiceRoute
    app.state.settings = settings
    app.state.coordinator = None
    app.add_middleware(_GuardMiddleware, settings=settings)
    app.add_exception_handler(FinanceAIError, _on_finance_error)
    app.add_exception_handler(RequestValidationError, _on_validation_error)
    app.add_exception_handler(StarletteHTTPException, _on_http_error)
    app.add_exception_handler(Exception, _on_unexpected)
    auth = [Depends(_require_token)]

    @app.get(
        "/healthz",
        response_model=Health,
        tags=["health"],
        responses=_responses(400),
        summary="Process liveness and worker readiness (unauthenticated)",
    )
    def healthz(request: Request) -> Health:
        coordinator = request.app.state.coordinator
        ready = isinstance(coordinator, Coordinator) and coordinator.available
        return Health(status="ok", readiness="READY" if ready else "NOT_READY")

    @app.post(
        "/v1/runs",
        status_code=202,
        response_model=RunAccepted,
        tags=["runs"],
        dependencies=auth,
        responses=_responses(400, 401, 403, 404, 409, 413, 422, 429, 500, 503),
        summary="Submit an allowlisted job (idempotent per campaign and Idempotency-Key)",
    )
    def submit_run(
        payload: RunRequest,
        request: Request,
        response: Response,
        idempotency_key: Annotated[
            str,
            Header(
                alias="Idempotency-Key",
                pattern=IDEMPOTENCY_KEY_PATTERN,
                description="Client key; reuse it only for retries",
            ),
        ],
    ) -> RunAccepted:
        record, replayed = _coordinator(request).submit(payload.campaign_id, idempotency_key, payload.job.to_domain())
        url = f"/v1/runs/{record.run_id}"
        response.headers["Location"] = url
        return RunAccepted(
            run_id=record.run_id,
            status=record.status.value,
            status_url=url,
            cancel_url=f"{url}/cancel",
            replayed=replayed,
        )

    @app.get(
        "/v1/runs/{run_id}",
        response_model=RunStatus,
        tags=["runs"],
        dependencies=auth,
        responses=_responses(400, 401, 404, 422, 500, 503),
        summary="Job state, progress, failure and artifacts",
    )
    def get_run(run_id: str, request: Request) -> RunStatus:
        if not is_run_id(run_id):
            raise NotFoundError("run not found")
        coordinator = _coordinator(request)
        record = coordinator.status(run_id)
        return run_status(record, coordinator.run_artifacts(run_id))

    @app.post(
        "/v1/runs/{run_id}/cancel",
        status_code=202,
        response_model=RunStatus,
        tags=["runs"],
        dependencies=auth,
        summary="Idempotent cancellation request (202 pending, 200 terminal)",
        responses={
            200: {"model": RunStatus, "description": "The run is terminal"},
            **_responses(400, 401, 404, 422, 500, 503),
        },
    )
    def cancel_run(run_id: str, request: Request, response: Response) -> RunStatus:
        if not is_run_id(run_id):
            raise NotFoundError("run not found")
        coordinator = _coordinator(request)
        record = coordinator.cancel(run_id)
        if record.terminal:
            response.status_code = 200
        return run_status(record, coordinator.run_artifacts(run_id))

    @app.get(
        "/v1/artifacts/{artifact_id}",
        response_model=ArtifactMetadata,
        tags=["artifacts"],
        dependencies=auth,
        responses=_responses(400, 401, 404, 422, 500, 503),
        summary="Metadata of a committed artifact (no content)",
    )
    def get_artifact(artifact_id: str, request: Request) -> ArtifactMetadata:
        if not is_artifact_id(artifact_id):
            raise NotFoundError("artifact not found")
        return artifact_metadata(_coordinator(request).artifact(artifact_id))

    return app


def openapi_document() -> dict[str, Any]:
    """The OpenAPI document generated from the transport models and routes (no I/O)."""
    placeholder = ServiceSettings(state_root=Path("/financeai-openapi-placeholder"), token="x" * MIN_TOKEN_LENGTH)
    doc: dict[str, Any] = create_app(placeholder).openapi()
    doc["info"]["x-api-version"] = API_VERSION
    return doc


def render_openapi() -> str:
    """Canonical text of the versioned snapshot ``docs/api/openapi-v1.json``."""
    return json.dumps(openapi_document(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
