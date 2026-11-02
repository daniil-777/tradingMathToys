"""Helpers and fixtures for the local job-service tests (synthetic data only; no network, no market files)."""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

with warnings.catch_warnings():  # starlette warns that its TestClient will move to httpx2
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from financeai.service.app import create_app
from financeai.service.coordinator import Coordinator, JobRunner
from financeai.service.models import EXAMPLE_STRATEGY
from financeai.service.settings import ServiceSettings

TOKEN = "test-token-" + "x" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BASE_URL = "http://127.0.0.1"
TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"})
VENUES = ["venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"]


def synthetic_job(seed: int = 7, **overrides: Any) -> dict[str, Any]:
    return {
        "kind": "synthetic_replay",
        "seed": seed,
        "venues": list(VENUES),
        "scenario": "base",
        "strategy": json.loads(json.dumps(EXAMPLE_STRATEGY)),
        **overrides,
    }


def run_request(job: Mapping[str, Any], campaign: str = "test") -> dict[str, Any]:
    return {"schema_version": "1.0", "campaign_id": campaign, "job": dict(job)}


SettingsFactory = Callable[..., ServiceSettings]


@pytest.fixture
def make_settings(tmp_path: Path) -> SettingsFactory:
    def make(**overrides: Any) -> ServiceSettings:
        options: dict[str, Any] = {"cancel_grace_s": 1.0, **overrides}
        return ServiceSettings(state_root=tmp_path / "state", token=TOKEN, **options)

    return make


@contextmanager
def service(
    settings: ServiceSettings, runner: JobRunner | None = None, *, raise_server_exceptions: bool = True
) -> Iterator[TestClient]:
    app = create_app(settings, job_runner=runner)
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=raise_server_exceptions) as client:
        yield client


def coordinator_of(client: TestClient) -> Coordinator:
    coordinator = client.app.state.coordinator  # type: ignore[attr-defined]
    assert isinstance(coordinator, Coordinator)
    return coordinator


def post_run(
    client: TestClient,
    body: Mapping[str, Any] | str | bytes,
    key: str = "key-1",
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    content = body if isinstance(body, str | bytes) else json.dumps(body)
    response: httpx.Response = client.post(
        "/v1/runs",
        content=content,
        headers={"Content-Type": "application/json", "Idempotency-Key": key, **AUTH, **(headers or {})},
    )
    return response


def get_run(client: TestClient, run_id: str) -> dict[str, Any]:
    response = client.get(f"/v1/runs/{run_id}", headers=AUTH)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def wait_status(
    client: TestClient, run_id: str, statuses: set[str] | frozenset[str], timeout_s: float = 240.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        body = get_run(client, run_id)
        if body["status"] in statuses:
            return body
        if time.monotonic() > deadline:
            raise AssertionError(f"run {run_id} still {body['status']}: {body}")
        time.sleep(0.05)


def wait_progress(client: TestClient, run_id: str, phase: str, timeout_s: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        body = get_run(client, run_id)
        if body["progress"] and body["progress"]["phase"] == phase:
            return body
        if time.monotonic() > deadline or body["status"] in TERMINAL:
            raise AssertionError(f"run {run_id} never reached phase {phase}: {body}")
        time.sleep(0.05)


def assert_error(response: httpx.Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    body: dict[str, Any] = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id", "details"}
    assert body["error"]["code"] == code, body
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    return body
