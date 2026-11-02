"""Immutable artifact store under ``STATE_ROOT/artifacts``: content digests, safe resolution, registration.

An artifact is one file or one directory tree. Its ``sha256`` is the file's SHA-256 or, for
a directory, the SHA-256 of the canonical JSON member list ``{"schema": "financeai.tree/1",
"members": [{"path", "size", "sha256"}, ...]}`` (POSIX paths, sorted); ``size_bytes`` is the
total payload. Symbolic links and special files are rejected everywhere.

Artifacts are addressed only by registered IDs; paths are resolved from the registry and
must stay inside the store. Inputs enter through :func:`register_strategy` and
:func:`register_tape_cache` as private read-only copies (mode :data:`READ_ONLY`, never a hard
link: a shared inode would let an edit of the caller's file change or corrupt the registered
version behind its recorded digest), verified before and after the copy; run outputs are
published by the coordinator. Nothing here serves file contents.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

from financeai.core.errors import InvalidConfigError, InvalidDataError, InvariantViolationError
from financeai.core.hashing import sha256_file, sha256_json
from financeai.core.serialization import loads_strict
from financeai.service.store import ArtifactRecord, JobStore, new_id

ARTIFACT_KINDS: Final[Mapping[str, tuple[str, str]]] = {
    "strategy_config": ("application/json", "financeai.strategy/1"),
    "tape_cache": ("application/vnd.financeai.tape-cache+directory", "financeai.market_tapes/1"),
    "run_summary": ("application/json", "financeai.run_summary/1"),
    "fills_csv": ("text/csv", "financeai.fills/1"),
    "orders_csv": ("text/csv", "financeai.orders/1"),
    "run_manifest": ("application/json", "financeai.run_manifest/1"),
}
DATA_ROLES: Final = ("SYNTHETIC", "DEVELOPMENT", "RESERVED_TEST", "NONE")
CACHE_ROLES: Final = ("SYNTHETIC", "DEVELOPMENT", "RESERVED_TEST")
FORBIDDEN_ROLES: Final = frozenset({"RESERVED_TEST"})
"""Roles this service never opens: a reserved test needs a recorded freeze and opening
authorization, which only the campaign protocol (not an HTTP request) can grant."""
TREE_SCHEMA: Final = "financeai.tree/1"
READ_ONLY: Final = 0o444
"""Mode of every registered input file: immutable for its owner too (directories stay writable so the
store can be moved or removed)."""


def _reject_link(path: Path) -> None:
    if path.is_symlink():
        raise InvalidDataError("artifacts may not contain symbolic links", context={"name": path.name})


def tree_digest(path: str | Path) -> tuple[str, int]:
    """``(sha256, size_bytes)`` of a regular file or a directory tree (see the module docstring)."""
    root = Path(path)
    _reject_link(root)
    if root.is_file():
        return sha256_file(root), root.stat().st_size
    if not root.is_dir():
        raise InvalidDataError("artifact is missing or not a regular file/directory", context={"name": root.name})
    members: list[dict[str, Any]] = []
    total = 0
    for current, dirs, files in os.walk(root):
        here = Path(current)
        for name in dirs:
            _reject_link(here / name)
        dirs.sort()
        for name in sorted(files):
            p = here / name
            _reject_link(p)
            if not p.is_file():
                raise InvalidDataError("artifacts may contain only regular files", context={"name": name})
            size = p.stat().st_size
            members.append({"path": p.relative_to(root).as_posix(), "size": size, "sha256": sha256_file(p)})
            total += size
    members.sort(key=lambda m: str(m["path"]))
    return sha256_json({"schema": TREE_SCHEMA, "members": members}), total


def _safe_parts(rel_path: str) -> tuple[str, ...]:
    p = PurePosixPath(rel_path)
    if "\\" in rel_path or p.is_absolute() or not p.parts or any(part in ("", ".", "..") for part in p.parts):
        raise InvariantViolationError("unsafe artifact path in the registry")
    return p.parts


class ArtifactStore:
    """Path resolution and verification for registered artifacts (no content serving)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_of(self, rel_path: str) -> Path:
        candidate = self.root.joinpath(*_safe_parts(rel_path))
        if not candidate.resolve().is_relative_to(self.root.resolve()):
            raise InvariantViolationError("artifact path escapes the artifact store")
        return candidate

    def verify(self, record: ArtifactRecord) -> Path:
        """Resolve a record and check that its content still matches the registered digest."""
        path = self.path_of(record.rel_path)
        digest, size = tree_digest(path)
        if digest != record.sha256 or size != record.size_bytes:
            raise InvalidDataError("registered artifact content changed", context={"artifact_id": record.artifact_id})
        return path

    def _incoming(self, artifact_id: str) -> Path:
        staging = self.root / ".incoming" / artifact_id
        staging.parent.mkdir(parents=True, exist_ok=True)
        return staging


def _make_writable(root: Path) -> None:
    """Undo :data:`READ_ONLY` below ``root`` so a refused registration can be removed again."""
    for current, _dirs, files in os.walk(root):
        for name in files:
            with contextlib.suppress(OSError):
                os.chmod(Path(current) / name, 0o644)


def _copy_read_only(src: str, dst: str) -> None:
    """A private read-only copy (``copyfile``, so the destination never shares the source's inode)."""
    shutil.copyfile(src, dst)
    os.chmod(dst, READ_ONLY)


def _record(
    artifact_id: str,
    kind: str,
    rel_path: str,
    store: ArtifactStore,
    data_role: str,
    provenance: Mapping[str, Any],
    run_id: str | None = None,
) -> ArtifactRecord:
    digest, size = tree_digest(store.path_of(rel_path))
    media_type, schema = ARTIFACT_KINDS[kind]
    return ArtifactRecord(
        artifact_id=artifact_id,
        kind=kind,
        media_type=media_type,
        schema=schema,
        sha256=digest,
        size_bytes=size,
        rel_path=rel_path,
        data_role=data_role,
        provenance=dict(provenance),
        created_ns=time.time_ns(),
        run_id=run_id,
    )


def register_strategy(jobs: JobStore, store: ArtifactStore, source: str | Path) -> ArtifactRecord:
    """Validate a strategy config file and register an immutable copy of its exact bytes.

    Validation is the strict canonical check of :mod:`financeai.service.canonical`: a file whose
    numbers the engine would truncate or coerce (``direction 0.9``) is refused, not registered.
    """
    from financeai.service.canonical import canonical_strategy

    src = Path(source)
    if not src.is_file() or src.is_symlink():
        raise InvalidConfigError("strategy file not found (regular files only)", context={"name": src.name})
    data = src.read_bytes()
    cfg = canonical_strategy(loads_strict(data))
    artifact_id = new_id("art")
    rel_path = f"inputs/{artifact_id}/strategy.json"
    staging = store._incoming(artifact_id)
    staging.mkdir()
    try:
        (staging / "strategy.json").write_bytes(data)
        os.chmod(staging / "strategy.json", READ_ONLY)
        target = store.path_of(f"inputs/{artifact_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    record = _record(
        artifact_id,
        "strategy_config",
        rel_path,
        store,
        "NONE",
        {
            "producer": "registration",
            "source_name": src.name,
            "strategy_name": cfg.name,
            "instruments": list(cfg.instruments),
        },
    )
    jobs.add_artifacts([record])
    return record


def register_tape_cache(
    jobs: JobStore, store: ArtifactStore, source_dir: str | Path, *, data_role: str
) -> ArtifactRecord:
    """Verify a tape cache (manifest + every column hash) and register an immutable copy.

    Every file is copied into a private read-only file (never hard-linked, however cheap that
    would be on one filesystem: the caller keeps its own inode to edit); every use re-verifies
    the registered digest.
    """
    from financeai.domains.market.cache import load_tapes, read_manifest

    if data_role not in CACHE_ROLES:
        raise InvalidConfigError("unknown data role", context={"got": data_role, "allowed": list(CACHE_ROLES)})
    src = Path(source_dir)
    if not src.is_dir() or src.is_symlink():
        raise InvalidConfigError("tape cache directory not found", context={"name": src.name})
    source_digest, source_size = tree_digest(src)  # rejects links and special files before anything is copied
    manifest = read_manifest(src)
    load_tapes(src, verify=True)
    artifact_id = new_id("art")
    rel_path = f"inputs/{artifact_id}"
    staging = store._incoming(artifact_id)
    try:
        shutil.copytree(src, staging, copy_function=_copy_read_only)
        target = store.path_of(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    inst = manifest["instrument"]
    record = _record(
        artifact_id,
        "tape_cache",
        rel_path,
        store,
        data_role,
        {
            "producer": "registration",
            "source_name": src.name,
            "instrument": f"{inst['venue']}/{inst['symbol']}",
            "source_sha256": manifest["source"]["sha256"],
            "parser": dict(manifest["parser"]),
        },
    )
    if (record.sha256, record.size_bytes) != (source_digest, source_size):
        # The copy is what gets registered, so it is the copy that must be the verified tree: a cache
        # rewritten between the column check and the copy would otherwise register a torn tree whose
        # recorded digest is self-consistent (every later digest check passes, the column hashes do not).
        _make_writable(target)
        shutil.rmtree(target, ignore_errors=True)
        raise InvalidDataError(
            "tape cache changed during registration; register it again once it is stable", context={"name": src.name}
        )
    jobs.add_artifacts([record])
    return record
