"""Contract artifacts: the versioned OpenAPI snapshot and its examples, optional-dependency isolation,
the ``python -m financeai.service`` command line and a real Uvicorn subprocess smoke test."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from financeai.domains.market.synthetic import SyntheticMarketSpec
from financeai.service.__main__ import main as service_main
from financeai.service.app import openapi_document, render_openapi
from financeai.service.jobs import job_from_mapping
from financeai.service.models import EXAMPLE_REQUESTS, EXAMPLE_STRATEGY, RunRequest
from financeai.service.settings import STATE_ROOT_ENV, TOKEN_ENV
from financeai.service.store import JobStore

from .support import TOKEN, run_request, synthetic_job
from .test_jobs import write_caches

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "docs" / "api" / "openapi-v1.json"
SRC = REPO / "src"
HTTP_STACK = ("fastapi", "pydantic", "starlette", "uvicorn", "httpx", "anyio")


def test_openapi_snapshot_is_current() -> None:
    assert SNAPSHOT.read_text() == render_openapi(), (
        "contract changed: regenerate with `python -m financeai.service openapi --out docs/api/openapi-v1.json` "
        "and explain the diff"
    )


def test_openapi_documents_the_v1_contract() -> None:
    doc = openapi_document()
    assert doc["info"]["x-api-version"] == "1.0"
    assert set(doc["paths"]) == {
        "/healthz",
        "/v1/runs",
        "/v1/runs/{run_id}",
        "/v1/runs/{run_id}/cancel",
        "/v1/artifacts/{artifact_id}",
    }
    schemas = doc["components"]["schemas"]
    assert "HTTPValidationError" not in schemas and "ValidationError" not in schemas
    assert doc["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    assert "security" not in doc["paths"]["/healthz"]["get"]
    post = doc["paths"]["/v1/runs"]["post"]
    assert {"202", "401", "403", "404", "409", "413", "422", "429", "500", "503"} <= set(post["responses"])
    assert post["security"] and post["parameters"] == [
        {
            "name": "Idempotency-Key",
            "in": "header",
            "required": True,
            "schema": post["parameters"][0]["schema"],
            "description": post["parameters"][0]["description"],
        }
    ]
    for path in doc["paths"].values():
        for operation in path.values():
            for code, response in operation["responses"].items():
                if code.startswith(("4", "5")):
                    assert response["content"]["application/json"]["schema"] == {
                        "$ref": "#/components/schemas/ErrorEnvelope"
                    }
    cancel = doc["paths"]["/v1/runs/{run_id}/cancel"]["post"]["responses"]
    assert {"200", "202"} <= set(cancel)
    for name in ("RunRequest", "SyntheticReplayRequest", "StrategyReplayRequest", "RunStatus", "ArtifactMetadata"):
        assert schemas[name]["additionalProperties"] is False, name
    start = schemas["SyntheticReplayRequest"]["properties"]["start_ns"]
    assert start["type"] == "string" and start["pattern"] == "^(0|[1-9][0-9]{0,18})$"
    assert str(2**62) in start["description"]  # the start cap is part of the documented contract
    assert "JSON integers" in schemas["SyntheticReplayRequest"]["properties"]["strategy"]["description"]
    assert schemas["RunRequest"]["properties"]["job"]["discriminator"]["propertyName"] == "kind"


def test_schema_examples_validate_strictly_round_trip_and_pass_the_domain_validator() -> None:
    examples = openapi_document()["components"]["schemas"]["RunRequest"]["examples"]
    assert examples == json.loads(json.dumps(list(EXAMPLE_REQUESTS)))
    for example in examples:
        model = RunRequest.model_validate(example)
        assert model.model_dump(mode="json") == example
        assert job_from_mapping(model.job.to_domain()).kind == example["job"]["kind"]


def test_core_and_service_internals_import_without_the_http_stack() -> None:
    code = (
        "import json, sys\n"
        "import financeai, financeai.cli, financeai.domains.market, financeai.domains.market.synthetic\n"
        "import financeai.application.strategy, financeai.service, financeai.service.jobs\n"
        "import financeai.service.coordinator, financeai.service.worker, financeai.service.store\n"
        f"print(json.dumps(sorted(m for m in {HTTP_STACK!r} if m in sys.modules)))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600, check=True)
    assert json.loads(out.stdout) == []


def test_service_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "openapi.json"
    assert service_main(["openapi", "--out", str(out)]) == 0
    assert out.read_text() == render_openapi()
    root = tmp_path / "state"
    strategy = tmp_path / "strategy.json"
    strategy.write_text(json.dumps(EXAMPLE_STRATEGY))
    assert service_main(["register-strategy", str(strategy), "--state-root", str(root)]) == 0
    registered = json.loads(capsys.readouterr().out)
    caches = write_caches(SyntheticMarketSpec(seed=1, rows_per_venue=20), tmp_path / "caches")
    assert (
        service_main(["register-cache", str(caches[0]), "--state-root", str(root), "--data-role", "DEVELOPMENT"]) == 0
    )
    cache = json.loads(capsys.readouterr().out)
    assert registered["kind"] == "strategy_config" and cache["data_role"] == "DEVELOPMENT"
    store = JobStore(root / "service.sqlite3")
    assert store.get_artifact(cache["artifact_id"]).sha256 == cache["sha256"]
    store.close()
    assert service_main(["register-strategy", str(tmp_path / "missing.json"), "--state-root", str(root)]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "invalid_config"
    monkeypatch.delenv(STATE_ROOT_ENV, raising=False)
    assert service_main(["register-strategy", str(strategy)]) == 2
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    assert service_main(["--state-root", str(root), "--host", "0.0.0.0"]) == 2  # serve is the default command
    assert "loopback" in capsys.readouterr().err


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def test_real_uvicorn_process_serves_a_job_and_shuts_down_cleanly(tmp_path: Path) -> None:
    port = _free_port()
    log_path = tmp_path / "server.log"
    env = {**os.environ, "PYTHONPATH": str(SRC), TOKEN_ENV: TOKEN}
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "financeai.service",
                "--state-root",
                str(tmp_path / "state"),
                "--port",
                str(port),
                "--log-level",
                "warning",
                "--cancel-grace-s",
                "2",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            deadline = time.monotonic() + 240
            while True:
                assert proc.poll() is None, log_path.read_text()
                try:
                    if client.get("/healthz").json()["readiness"] == "READY":
                        break
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, log_path.read_text()
                time.sleep(0.2)
            assert client.get("/healthz", headers={"Host": "evil.example"}).status_code == 400
            assert client.post("/v1/runs", content="{}", headers={"Idempotency-Key": "x"}).status_code == 401
            accepted = client.post(
                "/v1/runs",
                content=json.dumps(run_request(synthetic_job())),
                headers={**headers, "Idempotency-Key": "smoke"},
            )
            assert accepted.status_code == 202, accepted.text
            run_id = accepted.json()["run_id"]
            while True:
                status = client.get(f"/v1/runs/{run_id}", headers=headers).json()
                if status["status"] in {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}:
                    break
                assert time.monotonic() < deadline + 240, status
                time.sleep(0.2)
            assert status["status"] == "SUCCEEDED", status
            assert len(status["artifact_ids"]) == 7
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover - diagnostic path
            proc.kill()
            raise
    assert code == 0, log_path.read_text()
    assert (tmp_path / "state" / "api-token").exists() is False  # the environment token was used
