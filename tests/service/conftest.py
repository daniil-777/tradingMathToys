"""Fixtures of the local job-service tests (see :mod:`tests.service.support`).

Most modules here drive the API through FastAPI's test client (:mod:`tests.service.support`) or
import the Pydantic transport models. Without the ``api`` extra they are reported as skipped
instead of failing to collect, while the plain-Python store, settings and synthetic-market tests
still run. (A ``pytest.importorskip`` in this file would not do: when ``tests/service`` is named on
the command line pytest loads this conftest before collection, where a skip is not handled.)
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path

import pytest

API_EXTRA = ("fastapi", "httpx", "pydantic", "uvicorn")
PLAIN_MODULES = frozenset({"test_settings.py", "test_store.py", "test_synthetic_market.py"})
"""Test modules that need nothing from the ``api`` extra; any other module is assumed to need it."""


def _missing(modules: Iterable[str]) -> list[str]:
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    return missing


MISSING = _missing(API_EXTRA)

if MISSING:

    class _NeedsApiExtra(pytest.Module):
        def collect(self) -> list[pytest.Item]:
            pytest.skip(f"needs the 'api' extra (missing: {', '.join(MISSING)})", allow_module_level=True)

    def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.Module | None:
        return None if module_path.name in PLAIN_MODULES else _NeedsApiExtra.from_parent(parent, path=module_path)
else:
    from .support import make_settings as make_settings
