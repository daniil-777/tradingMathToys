"""Strict JSON decoding shared by configs, manifests and run records."""

from __future__ import annotations

import pytest

from financeai.core.errors import InvalidConfigError
from financeai.core.serialization import canonical_json, loads_strict


def test_loads_plain_document() -> None:
    assert loads_strict('{"a": 1, "b": [1.5, "x"]}') == {"a": 1, "b": [1.5, "x"]}


def test_duplicate_keys_are_rejected_at_any_depth() -> None:
    with pytest.raises(InvalidConfigError):
        loads_strict('{"a": 1, "a": 2}')
    with pytest.raises(InvalidConfigError):
        loads_strict('{"outer": {"k": 1, "k": 1}}')


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected(token: str) -> None:
    with pytest.raises(InvalidConfigError):
        loads_strict(f'{{"x": {token}}}')


@pytest.mark.parametrize(
    "document",
    [
        "1" * 5000,  # beyond CPython's int-string conversion limit: a plain ValueError, not a JSONDecodeError
        b'{"a": "\xff"}',  # invalid UTF-8: UnicodeDecodeError
        "[" * 100_000 + "]" * 100_000,  # deeper than the interpreter's recursion limit: RecursionError
        "1e400",  # a finite spelling of an infinite number (json.loads returns inf)
        '{"x": [1, -1e999]}',
    ],
    ids=["huge-int", "invalid-utf8", "deep-nesting", "overflowing-float", "nested-overflowing-float"],
)
def test_undecodable_documents_raise_the_typed_error_not_builtin_exceptions(document: str | bytes) -> None:
    """Regression: every decoder failure is the typed error (HTTP 422 at the service), never a 500."""
    with pytest.raises(InvalidConfigError):
        loads_strict(document)


def test_large_finite_numbers_still_decode() -> None:
    assert loads_strict("1" * 4000) == int("1" * 4000)
    assert loads_strict('{"x": 1.5e308}') == {"x": 1.5e308}


def test_canonical_json_is_order_independent_and_compact() -> None:
    assert canonical_json({"b": 1, "a": [2, 1]}) == canonical_json({"a": [2, 1], "b": 1})
    assert canonical_json({"a": 1}) == '{"a":1}'


def test_canonical_json_refuses_non_finite_floats() -> None:
    with pytest.raises(InvalidConfigError):
        canonical_json({"x": float("nan")})


@pytest.mark.parametrize("value", [object(), {"x": {1, 2}}, {"x": b"bytes"}])
def test_canonical_json_reports_unsupported_types_as_config_errors(value: object) -> None:
    with pytest.raises(InvalidConfigError):
        canonical_json(value)
