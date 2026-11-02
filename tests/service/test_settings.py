"""Service settings, token loading and the typed error -> HTTP status mapping."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from financeai.core.errors import (
    FinanceAIError,
    IncompatibleSchemaError,
    InvalidConfigError,
    InvalidDataError,
    InvariantViolationError,
    OptionalDependencyError,
    UnsupportedCapabilityError,
)
from financeai.service.errors import (
    AuthenticationError,
    ForbiddenDataRoleError,
    IdempotencyConflictError,
    InvalidRequestError,
    NotFoundError,
    QueueFullError,
    RequestTooLargeError,
    StateConflictError,
    WorkerUnavailableError,
    http_status_for,
)
from financeai.service.settings import (
    STATE_ROOT_ENV,
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    ServiceSettings,
    is_loopback_host,
    read_token_file,
)

TOKEN = "t" * 40


def test_settings_defaults_and_layout(tmp_path: Path) -> None:
    s = ServiceSettings(state_root=tmp_path / "state", token=TOKEN)
    assert s.allowed_hosts == ("127.0.0.1", "localhost")
    assert s.max_workers == 1 and s.queue_capacity >= 1 and s.max_body_bytes >= 1024
    assert s.state_root.is_absolute()
    assert s.db_path == s.state_root / "service.sqlite3"
    assert {s.artifacts_dir.name, s.staging_dir.name, s.quarantine_dir.name} == {"artifacts", "staging", "quarantine"}
    assert TOKEN not in repr(s)


@pytest.mark.parametrize(
    "overrides",
    [
        {"token": "short"},
        {"token": "has whitespace " + "x" * 30},
        {"allowed_hosts": ()},
        {"allowed_hosts": ("bad host",)},
        {"queue_capacity": 0},
        {"queue_capacity": True},
        {"max_body_bytes": 10},
        {"max_workers": -1},
        {"max_workers": 1.5},
        {"cancel_grace_s": 0.0},
        {"cancel_grace_s": float("nan")},
    ],
)
def test_invalid_settings_are_rejected(tmp_path: Path, overrides: dict[str, object]) -> None:
    kwargs: dict[str, object] = {"state_root": tmp_path, "token": TOKEN, **overrides}
    with pytest.raises(InvalidConfigError):
        ServiceSettings(**kwargs)  # type: ignore[arg-type]


def test_hosts_are_normalised_to_lowercase(tmp_path: Path) -> None:
    s = ServiceSettings(state_root=tmp_path, token=TOKEN, allowed_hosts=("LocalHost", "127.0.0.1"))
    assert s.allowed_hosts == ("localhost", "127.0.0.1")


def test_token_from_environment_wins(tmp_path: Path) -> None:
    env = {TOKEN_ENV: TOKEN, STATE_ROOT_ENV: str(tmp_path)}
    s = ServiceSettings.from_environment(environ=env)
    assert s.token == TOKEN and s.state_root == tmp_path.resolve()


def test_token_file_is_created_private_and_reused(tmp_path: Path) -> None:
    s1 = ServiceSettings.from_environment(tmp_path, environ={}, create_token=True)
    token_file = tmp_path / "api-token"
    assert token_file.exists() and len(s1.token) >= 32
    if os.name == "posix":
        assert token_file.stat().st_mode & 0o077 == 0
    s2 = ServiceSettings.from_environment(tmp_path, environ={})
    assert s2.token == s1.token


def test_explicit_token_file_and_permission_check(tmp_path: Path) -> None:
    f = tmp_path / "tok"
    f.write_text(TOKEN + "\n")
    os.chmod(f, 0o600)
    s = ServiceSettings.from_environment(tmp_path, environ={TOKEN_FILE_ENV: str(f)})
    assert s.token == TOKEN
    if os.name == "posix":
        os.chmod(f, 0o644)
        with pytest.raises(InvalidConfigError):
            read_token_file(f)


def test_missing_token_or_state_root_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigError):
        ServiceSettings.from_environment(tmp_path, environ={})
    with pytest.raises(InvalidConfigError):
        ServiceSettings.from_environment(environ={TOKEN_ENV: TOKEN})


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("127.0.0.2", True),
        ("0.0.0.0", False),
        ("192.168.1.10", False),
        ("example.com", False),
        ("::", False),
    ],
)
def test_loopback_detection(host: str, loopback: bool) -> None:
    assert is_loopback_host(host) is loopback


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (AuthenticationError("x"), 401),
        (ForbiddenDataRoleError("x"), 403),
        (NotFoundError("x"), 404),
        (IdempotencyConflictError("x"), 409),
        (StateConflictError("x"), 409),
        (IncompatibleSchemaError("x"), 409),
        (RequestTooLargeError("x"), 413),
        (InvalidRequestError("x"), 422),
        (InvalidConfigError("x"), 422),
        (InvalidDataError("x"), 422),
        (UnsupportedCapabilityError("x"), 422),
        (QueueFullError("x"), 429),
        (InvariantViolationError("x"), 500),
        (FinanceAIError("x"), 500),
        (WorkerUnavailableError("x"), 503),
        (OptionalDependencyError("x"), 503),
    ],
)
def test_error_status_mapping(exc: FinanceAIError, status: int) -> None:
    assert http_status_for(exc) == status
    assert exc.code and exc.code.islower()
