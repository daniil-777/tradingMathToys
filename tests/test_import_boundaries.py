"""Architecture boundaries, checked statically (``ast``) on every module under ``src/financeai``.

Layering (a module may import its own layer, the root package and the layers below it)::

    core <- {models, online} <- domains.{market, demand} <- strategies <- application
         <- {reporting, research, service} <- cli

* ``core`` depends on nothing but the standard library and NumPy.
* ``models`` (the ridge adapter) and ``online`` (tickets, maturity-gated feedback) are foundation
  packages over ``core``: ``domains.demand`` is built on them (``docs/architecture.md`` section 8), so they
  sit below the domains rather than beside the entry points, and they import nothing but ``core``.
* The two domains never import each other: ``domains.demand`` is the nonfinancial proof that the
  forecast machinery carries no market knowledge.
* ``strategies`` read prepared markets; they never import ``application`` or ``service``.
* ``application`` composes domains and strategies; it never imports ``reporting``, ``research``,
  ``service`` or ``cli``.
* ``reporting``, ``research`` and ``service`` are peers over ``application``; none imports another.
* ``cli`` is the composition root: it may import every layer (lazily, inside each subcommand) and
  nothing imports it.

The checks see lazy imports too: a lower layer calling a higher one inside a function is still a
dependency. Separately, the optional extras of ``pyproject.toml`` other than ``fast`` (Numba, which
the kernels compile with at import time by design) are never imported at module import time outside
the HTTP adapter of the service; ``tests/test_optional_dependencies.py`` confirms the same in a fresh
interpreter.
"""

from __future__ import annotations

import ast
import importlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "financeai"
PROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]

ROOT = "financeai"
"""The root package: it imports nothing from the library and everything may import it."""
FOUNDATION = frozenset({"core", "models", "online"})
DOMAINS = frozenset({"domains.market", "domains.demand"})
_OVER_APPLICATION = FOUNDATION | DOMAINS | {"strategies", "application"}
ALLOWED: dict[str, frozenset[str]] = {
    ROOT: frozenset(),
    "core": frozenset({"core"}),
    "models": frozenset({"core", "models"}),
    "online": frozenset({"core", "online"}),
    "domains": frozenset(),  # the namespace package itself
    "domains.market": FOUNDATION | {"domains.market"},
    "domains.demand": FOUNDATION | {"domains.demand"},
    "strategies": FOUNDATION | DOMAINS | {"strategies"},
    "application": _OVER_APPLICATION,
    "reporting": _OVER_APPLICATION | {"reporting"},
    "research": _OVER_APPLICATION | {"research"},
    "service": _OVER_APPLICATION | {"service"},
    "cli": _OVER_APPLICATION | {"reporting", "research", "service", "cli"},
}
"""Layers a module of each layer may import from (besides the root package)."""

HTTP_LAYER = frozenset({"financeai.service.app", "financeai.service.models", "financeai.service.__main__"})
"""The HTTP adapter: the only modules allowed to import the ``api`` extra at import time."""


def _import_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match is not None, requirement
    return match.group(0).lower().replace("-", "_").replace(".", "_")


OPTIONAL = frozenset(
    _import_name(r)
    for name, reqs in PROJECT["optional-dependencies"].items()
    if name not in ("dev", "fast")
    for r in reqs
)
"""Import names of the runtime extras that must stay lazy (``api`` and ``report`` today)."""


def layer_of(module: str) -> str:
    """``financeai.domains.market.engine`` -> ``domains.market``; ``financeai.core.jit`` -> ``core``."""
    parts = module.split(".")
    assert parts[0] == ROOT, module
    if len(parts) == 1:
        return ROOT
    if parts[1] == "__main__":
        return "cli"  # python -m financeai is another entry point to the same composition root
    if parts[1] == "domains":
        return "domains" if len(parts) == 2 else f"domains.{parts[2]}"
    return parts[1]


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


@dataclass(frozen=True, slots=True)
class Import:
    module: str
    """The importing module."""
    target: str
    """Absolute dotted name imported (``from a.b import c`` records ``a.b.c``)."""
    line: int
    at_import_time: bool
    """True when the statement runs when the module is imported: not inside a function or lambda and
    not under ``if TYPE_CHECKING:``."""

    def __str__(self) -> str:
        return f"{self.module}:{self.line} imports {self.target}"


def _is_type_checking(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _absolute(module: str, node: ast.ImportFrom, is_package: bool) -> str:
    if node.level == 0:
        assert node.module is not None
        return node.module
    base = module.split(".")
    if not is_package:
        base = base[:-1]
    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
    return ".".join([*base, node.module] if node.module else base)


def _imports_of(path: Path) -> list[Import]:
    module = _module_name(path)
    is_package = path.name == "__init__.py"
    found: list[Import] = []

    def record(target: str, node: ast.AST, deferred: bool) -> None:
        found.append(Import(module, target, node.lineno, not deferred))  # type: ignore[attr-defined]

    def visit(node: ast.AST, deferred: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    record(alias.name, child, deferred)
            elif isinstance(child, ast.ImportFrom):
                base = _absolute(module, child, is_package)
                for alias in child.names:
                    record(f"{base}.{alias.name}" if alias.name != "*" else base, child, deferred)
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "import_module"
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                record(child.args[0].value, child, deferred)
            if _is_type_checking(child):
                assert isinstance(child, ast.If)
                for stmt in child.body:
                    visit(stmt, True)
                for stmt in child.orelse:
                    visit(stmt, deferred)
                continue
            visit(child, deferred or isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda))

    visit(ast.parse(path.read_text(), str(path)), False)
    return found


MODULE_PATHS = sorted(SRC.rglob("*.py"))
MODULES = [_module_name(p) for p in MODULE_PATHS]
IMPORTS = [imp for path in MODULE_PATHS for imp in _imports_of(path)]


def _library_target(target: str) -> bool:
    return target == ROOT or target.startswith(ROOT + ".")


def _top(target: str) -> str:
    return target.split(".")[0].lower().replace("-", "_")


# ---------------------------------------------------------------------------
# The scanner sees what it must (so the checks below are not vacuous)
# ---------------------------------------------------------------------------


def test_the_scanner_finds_every_module_and_classifies_lazy_imports() -> None:
    assert {
        "financeai",
        "financeai.cli",
        "financeai.core.jit",
        "financeai.domains.market.engine",
        "financeai.service.__main__",
    } <= set(MODULES)
    cli_lazy = [i for i in IMPORTS if i.module == "financeai.cli" and i.target.startswith("financeai.application")]
    assert cli_lazy and not any(i.at_import_time for i in cli_lazy), "subcommand imports are lazy"
    pydantic = [i for i in IMPORTS if i.module == "financeai.service.models" and _top(i.target) == "pydantic"]
    assert pydantic and all(i.at_import_time for i in pydantic), "the transport models import pydantic at import time"
    charts = [i for i in IMPORTS if i.module == "financeai.reporting.charts" and _top(i.target) == "matplotlib"]
    assert charts and not any(i.at_import_time for i in charts), "import_module('matplotlib') inside a function"


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


def test_every_module_belongs_to_a_declared_layer() -> None:
    unknown = sorted(m for m in MODULES if layer_of(m) not in ALLOWED)
    assert unknown == [], "place the new package in ALLOWED (and docs/api.md) before adding it"


def test_layers_import_only_what_the_layering_allows() -> None:
    violations = sorted(
        str(i)
        for i in IMPORTS
        if _library_target(i.target)
        and layer_of(i.target) != ROOT
        and layer_of(i.target) not in ALLOWED[layer_of(i.module)]
    )
    assert violations == []


def test_the_root_package_imports_nothing_from_the_library() -> None:
    assert [str(i) for i in IMPORTS if i.module == ROOT and _library_target(i.target)] == []


@pytest.mark.parametrize(
    ("layer", "forbidden"),
    [
        (
            "core",
            {"models", "online", *DOMAINS, "strategies", "application", "reporting", "research", "service", "cli"},
        ),
        ("models", {"online", *DOMAINS, "strategies", "application", "reporting", "research", "service", "cli"}),
        ("online", {"models", *DOMAINS, "strategies", "application", "reporting", "research", "service", "cli"}),
        ("domains.market", {"domains.demand", "strategies", "application", "reporting", "research", "service", "cli"}),
        ("domains.demand", {"domains.market", "strategies", "application", "reporting", "research", "service", "cli"}),
        ("strategies", {"application", "reporting", "research", "service", "cli"}),
        ("application", {"reporting", "research", "service", "cli"}),
        ("reporting", {"research", "service", "cli"}),
        ("research", {"reporting", "service", "cli"}),
        ("service", {"reporting", "research", "cli"}),
    ],
)
def test_the_table_encodes_the_stated_rules(layer: str, forbidden: set[str]) -> None:
    """Guards the table itself: loosening a rule here must be a visible decision."""
    assert not (ALLOWED[layer] & forbidden)
    assert layer in ALLOWED[layer] or layer == ROOT


def test_nothing_imports_the_composition_root() -> None:
    assert not any("cli" in allowed for layer, allowed in ALLOWED.items() if layer != "cli")
    assert [str(i) for i in IMPORTS if i.target.startswith("financeai.cli") and layer_of(i.module) != "cli"] == []


# ---------------------------------------------------------------------------
# Optional extras stay out of the numerical core at import time
# ---------------------------------------------------------------------------


def test_the_optional_extras_named_by_the_contract_are_covered() -> None:
    assert {"fastapi", "matplotlib", "uvicorn", "pydantic", "httpx"} <= OPTIONAL


def test_no_optional_extra_is_imported_at_import_time_outside_the_http_adapter() -> None:
    offending = sorted(
        str(i) for i in IMPORTS if i.at_import_time and i.module not in HTTP_LAYER and _top(i.target) in OPTIONAL
    )
    assert offending == []


# ---------------------------------------------------------------------------
# Public package API surface & PEP 561 marker
# ---------------------------------------------------------------------------


def test_pep561_py_typed_marker_is_present() -> None:
    assert (SRC / "py.typed").is_file()


@pytest.mark.parametrize(
    "pkg_name",
    [
        "financeai",
        "financeai.core",
        "financeai.models",
        "financeai.online",
        "financeai.domains.market",
        "financeai.domains.demand",
        "financeai.strategies",
        "financeai.application",
        "financeai.reporting",
        "financeai.research",
        "financeai.service",
    ],
)
def test_every_layer_package_declares_valid_public_exports(pkg_name: str) -> None:
    pkg = importlib.import_module(pkg_name)
    exported = getattr(pkg, "__all__", None)
    assert isinstance(exported, list) and len(exported) > 0
    for symbol in exported:
        assert hasattr(pkg, symbol), f"{pkg_name} exports missing symbol {symbol!r}"
