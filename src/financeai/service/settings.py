"""Service settings: the STATE_ROOT layout, bearer token, Host allowlist and resource bounds.

``STATE_ROOT`` is a local, single-host, unsynchronised work root holding the SQLite job
database, the immutable artifact store, per-attempt staging and quarantine. Do not place it
on a network or cloud-synchronised folder (SQLite locking and atomic renames are only
tested on a local filesystem). Export committed artifacts elsewhere after publication.
"""

from __future__ import annotations

import ipaddress
import math
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from financeai.core.errors import InvalidConfigError

TOKEN_ENV: Final = "FINANCEAI_API_TOKEN"
TOKEN_FILE_ENV: Final = "FINANCEAI_API_TOKEN_FILE"
STATE_ROOT_ENV: Final = "FINANCEAI_STATE_ROOT"
TOKEN_FILE_NAME: Final = "api-token"
DEFAULT_ALLOWED_HOSTS: Final = ("127.0.0.1", "localhost")
MIN_TOKEN_LENGTH: Final = 24
_TOKEN_RE: Final = re.compile(r"[\x21-\x7e]+")
_HOST_RE: Final = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?|\[?[0-9a-f:.]+\]?")


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals (``127.0.0.0/8``, ``::1``)."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_range(name: str, value: object, lo: int, hi: int) -> None:
    if not _is_int(value) or not lo <= value <= hi:  # type: ignore[operator]
        raise InvalidConfigError(f"{name} must be an integer in [{lo}, {hi}]", context={"got": value})


@dataclass(frozen=True)
class ServiceSettings:
    state_root: Path
    token: str = field(repr=False)
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    queue_capacity: int = 16
    """Maximum number of QUEUED (admitted, not yet started) jobs; more submissions get 429."""
    max_body_bytes: int = 256 * 1024
    max_workers: int = 1
    """Concurrent worker processes; ``0`` disables admission (503 worker_unavailable)."""
    cancel_grace_s: float = 5.0
    """Seconds a cancelled worker may take to stop cooperatively before it is terminated."""

    def __post_init__(self) -> None:
        if not isinstance(self.state_root, str | Path) or not str(self.state_root):
            raise InvalidConfigError("state_root must be a path")
        object.__setattr__(self, "state_root", Path(self.state_root).expanduser().resolve())
        if not isinstance(self.token, str) or len(self.token) < MIN_TOKEN_LENGTH or not _TOKEN_RE.fullmatch(self.token):
            raise InvalidConfigError(f"token must be at least {MIN_TOKEN_LENGTH} printable non-space characters")
        hosts = self.allowed_hosts
        if not isinstance(hosts, tuple | list) or not hosts:
            raise InvalidConfigError("allowed_hosts must be a non-empty list of host names")
        normalised = tuple(h.lower() for h in hosts if isinstance(h, str))
        if len(normalised) != len(hosts) or not all(_HOST_RE.fullmatch(h) for h in normalised):
            raise InvalidConfigError("allowed_hosts contains an invalid host name", context={"hosts": list(hosts)})
        object.__setattr__(self, "allowed_hosts", normalised)
        _check_range("queue_capacity", self.queue_capacity, 1, 100_000)
        _check_range("max_body_bytes", self.max_body_bytes, 1_024, 16 * 1024 * 1024)
        _check_range("max_workers", self.max_workers, 0, 64)
        grace = self.cancel_grace_s
        if (
            isinstance(grace, bool)
            or not isinstance(grace, int | float)
            or not math.isfinite(grace)
            or not 0 < grace <= 600
        ):
            raise InvalidConfigError("cancel_grace_s must be in (0, 600] seconds", context={"got": grace})
        object.__setattr__(self, "cancel_grace_s", float(grace))

    @property
    def db_path(self) -> Path:
        return self.state_root / "service.sqlite3"

    @property
    def artifacts_dir(self) -> Path:
        return self.state_root / "artifacts"

    @property
    def staging_dir(self) -> Path:
        return self.state_root / "staging"

    @property
    def quarantine_dir(self) -> Path:
        return self.state_root / "quarantine"

    @classmethod
    def from_environment(
        cls,
        state_root: str | Path | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        token_file: str | Path | None = None,
        create_token: bool = False,
        **overrides: Any,
    ) -> ServiceSettings:
        """Resolve STATE_ROOT and the token: ``FINANCEAI_API_TOKEN`` first, else a private token file
        (``token_file``, ``FINANCEAI_API_TOKEN_FILE`` or ``STATE_ROOT/api-token``)."""
        env = os.environ if environ is None else environ
        root = state_root if state_root is not None else env.get(STATE_ROOT_ENV)
        if not root:
            raise InvalidConfigError(f"a state root is required (argument or {STATE_ROOT_ENV})")
        root_path = Path(root).expanduser().resolve()
        token = env.get(TOKEN_ENV)
        if not token:
            path = token_file or env.get(TOKEN_FILE_ENV) or root_path / TOKEN_FILE_NAME
            token = read_token_file(path, create=create_token)
        return cls(state_root=root_path, token=token, **overrides)


def read_token_file(path: str | Path, *, create: bool = False) -> str:
    """Read a private token file; with ``create`` a missing file gets a fresh random token (0600)."""
    p = Path(path).expanduser()
    if not p.exists():
        if not create:
            raise InvalidConfigError(
                f"no API token configured: set {TOKEN_ENV} or create a private token file",
                context={"token_file": p.name},
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
    if os.name == "posix" and p.stat().st_mode & 0o077:
        raise InvalidConfigError(
            "token file must not be group/world accessible (chmod 600)", context={"token_file": p.name}
        )
    return p.read_text().strip()
