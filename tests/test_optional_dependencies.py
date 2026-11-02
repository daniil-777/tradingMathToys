"""Dependency hygiene: NumPy is the only core dependency, every other third-party import is declared in
an optional extra of ``pyproject.toml``, and importing :mod:`financeai` loads none of the extras.

The declared names come from ``pyproject.toml`` itself, so a new extra is covered as soon as it is
declared. Import checks run in a fresh interpreter: this test process has long since loaded the
extras through other tests.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
HTTP_LAYER = ("financeai.service.app", "financeai.service.models", "financeai.service.__main__")
"""Modules whose job is the HTTP API: they may import the ``api`` extra at module level."""


def _import_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match is not None, requirement
    return match.group(0).lower().replace("-", "_").replace(".", "_")


def _extra(name: str) -> set[str]:
    return {_import_name(r) for r in PROJECT["optional-dependencies"][name]}


CORE = {_import_name(r) for r in PROJECT["dependencies"]}
RUNTIME_EXTRAS = {name for name in PROJECT["optional-dependencies"] if name != "dev"}
OPTIONAL = set().union(*(_extra(name) for name in RUNTIME_EXTRAS)) - CORE


def _third_party_imports(path: Path) -> set[tuple[str, int]]:
    """Top-level module names imported by ``path`` (statements and literal ``import_module`` calls)."""
    found: set[tuple[str, int]] = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            found |= {(alias.name.split(".")[0], node.lineno) for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add((node.module.split(".")[0], node.lineno))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.add((node.args[0].value.split(".")[0], node.lineno))
    return {
        (name, line)
        for name, line in found
        if name not in sys.stdlib_module_names and name not in ("financeai", "__future__")
    }


def _undeclared(root: Path, allowed: set[str]) -> list[str]:
    local = {path.stem for path in root.glob("*.py")}
    if (root / "__init__.py").is_file():
        local.add(root.name)
    return sorted(
        f"{path.relative_to(REPO)}:{line} imports {name}"
        for path in root.rglob("*.py")
        for name, line in _third_party_imports(path)
        if name not in allowed | local
    )


def test_numpy_is_the_only_core_dependency() -> None:
    assert sorted(CORE) == ["numpy"]


def test_every_third_party_import_of_the_library_is_declared() -> None:
    assert _undeclared(SRC / "financeai", CORE | OPTIONAL) == []
    assert _undeclared(REPO / "examples", CORE | OPTIONAL) == []


def test_every_third_party_import_of_the_tests_is_declared() -> None:
    assert _undeclared(REPO / "tests", CORE | OPTIONAL | _extra("dev")) == []


def _loaded_after(code: str) -> list[str]:
    """Optional-extra modules present in ``sys.modules`` after running ``code`` in a fresh interpreter."""
    probe = (
        f"{code}\nimport json, sys\nprint(json.dumps(sorted(m for m in {sorted(OPTIONAL)!r} if m in sys.modules)))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=600, check=True
    )
    loaded: list[str] = json.loads(out.stdout.strip().splitlines()[-1])
    return loaded


def test_importing_financeai_loads_no_optional_dependency() -> None:
    assert {"numba", "matplotlib", "fastapi", "pydantic", "uvicorn", "httpx"} <= OPTIONAL
    assert _loaded_after("import financeai") == []


def test_only_the_http_layer_loads_the_api_extra_and_nothing_loads_matplotlib() -> None:
    deep = {"financeai.reporting.charts", "financeai.service.coordinator", "financeai.cli"}
    code = (
        "import importlib, pkgutil, sys, financeai\n"
        "for m in pkgutil.walk_packages(financeai.__path__, 'financeai.'):\n"
        f"    if m.name not in {HTTP_LAYER!r}:\n"
        "        importlib.import_module(m.name)\n"
        f"assert set({sorted(deep)!r}) <= set(sys.modules), 'the walk missed a package'\n"
    )
    allowed = _extra("fast")  # the numerical core compiles its kernels with Numba when it is installed
    assert [m for m in _loaded_after(code) if m not in allowed] == []
