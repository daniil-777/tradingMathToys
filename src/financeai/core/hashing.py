"""Content hashes (SHA-256, lowercase hex) for artifacts and configurations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from financeai.core.serialization import canonical_json

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash of the canonical JSON encoding of ``value``."""
    return sha256_bytes(canonical_json(value).encode("utf-8"))
