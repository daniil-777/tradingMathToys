"""Integer clocks: UTC epoch nanoseconds in int64, never floats.

Market timestamps are ``int64`` nanoseconds since 1970-01-01T00:00:00Z. Absent clocks
use the explicit sentinel :data:`MISSING_NS`; they are never fabricated from another
clock.
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from typing import Final, Literal

from financeai.core.errors import InvalidDataError
from financeai.core.jit import njit

NS_PER_US: Final = 1_000
NS_PER_MS: Final = 1_000_000
NS_PER_S: Final = 1_000_000_000
NS_PER_DAY: Final = 86_400 * NS_PER_S
MISSING_NS: Final = -(2**63)
"""Sentinel for an absent timestamp (e.g. an event without an exchange clock)."""


@njit()
def days_from_civil(year: int, month: int, day: int) -> int:
    """Days since 1970-01-01 of a proleptic-Gregorian date (H. Hinnant's algorithm)."""
    y = year - 1 if month <= 2 else year
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = month - 3 if month > 2 else month + 9
    doy = (153 * mp + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146_097 + doe - 719_468


def parse_recorder_datetime_ns(text: bytes) -> int:
    """Parse ``YY-MM-DD HH:MM:SS.ffffff`` (UTC, years 2000-2099) to epoch nanoseconds."""
    try:
        parsed = datetime.strptime(text.decode("ascii"), "%y-%m-%d %H:%M:%S.%f")
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidDataError("malformed recorder timestamp", context={"text": text[:32]}) from exc
    seconds = calendar.timegm(parsed.timetuple())
    return seconds * NS_PER_S + parsed.microsecond * NS_PER_US


def format_ns(ns: int, *, precision: Literal["us", "ns"] = "us") -> str:
    """ISO-8601 UTC rendering; opt into all nine fractional digits with precision='ns'."""
    if precision not in ("us", "ns"):
        raise ValueError("timestamp precision must be 'us' or 'ns'")
    if ns == MISSING_NS:
        return "<missing>"
    seconds, rem = divmod(ns, NS_PER_S)
    stamp = datetime.fromtimestamp(seconds, tz=UTC)
    fraction = f"{rem:09d}" if precision == "ns" else f"{rem // NS_PER_US:06d}"
    return f"{stamp:%Y-%m-%dT%H:%M:%S}.{fraction}Z"
