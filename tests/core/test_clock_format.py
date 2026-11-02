"""Display clocks preserve nanoseconds when explicitly requested."""

from __future__ import annotations

import pytest

from financeai.core.clock import MISSING_NS, format_ns


@pytest.mark.parametrize(
    ("ns", "expected"),
    [
        (0, "1970-01-01T00:00:00.000000000Z"),
        (1, "1970-01-01T00:00:00.000000001Z"),
        (-1, "1969-12-31T23:59:59.999999999Z"),
        (1_735_689_600_123_456_789, "2025-01-01T00:00:00.123456789Z"),
        (MISSING_NS, "<missing>"),
    ],
)
def test_nanosecond_precision(ns: int, expected: str) -> None:
    assert format_ns(ns, precision="ns") == expected


def test_default_keeps_microsecond_compatibility() -> None:
    assert format_ns(1_735_689_600_123_456_789) == "2025-01-01T00:00:00.123456Z"
    assert format_ns(-1) == "1969-12-31T23:59:59.999999Z"
    assert format_ns(MISSING_NS) == "<missing>"


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision"):
        format_ns(0, precision="ms")  # type: ignore[arg-type]
