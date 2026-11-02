"""Explicit price, quantity, and settlement conventions for exchange-traded instruments.

Tapes store integer ticks and integer quantity units. ``quantity_step`` converts one
stored unit to native shares, coins, lots, or contracts. ``contract_multiplier`` converts
native quantity to cash-flow exposure. Currency names never determine the arithmetic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from enum import StrEnum
from typing import Any

from financeai.core.errors import InvalidConfigError

_INT64_MAX = 2**63 - 1
_CURRENCY = re.compile(r"[A-Z][A-Z0-9._-]{0,15}")


class ContractType(StrEnum):
    SPOT = "spot"
    LINEAR = "linear"
    INVERSE = "inverse"


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    venue: str
    symbol: str
    tick_size: Decimal
    lot_size: int = 1
    """Minimum executable lot, measured in integer quantity units."""
    contract_type: ContractType = ContractType.LINEAR
    quantity_step: Decimal = Decimal("1")
    contract_multiplier: Decimal = Decimal("1")
    base_currency: str = "ASSET"
    quote_currency: str = "USD"
    settlement_currency: str | None = None
    shortable: bool | None = None

    def __post_init__(self) -> None:
        for name in ("venue", "symbol"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value or any(c.isspace() for c in value):
                raise InvalidConfigError(f"{name} must be a non-empty identifier without whitespace")
        if "/" in self.venue:
            raise InvalidConfigError("venue cannot contain '/'")
        if not isinstance(self.contract_type, ContractType):
            raise InvalidConfigError("contract_type must be a ContractType")
        for name in ("tick_size", "quantity_step", "contract_multiplier"):
            value = getattr(self, name)
            if (
                not isinstance(value, Decimal)
                or not value.is_finite()
                or value <= 0
                or not math.isfinite(float(value))
                or float(value) == 0
            ):
                raise InvalidConfigError(f"{name} must be a positive, finite Decimal representable as float64")
        factor = float(self.quantity_step) * float(self.contract_multiplier)
        if not math.isfinite(factor) or factor <= 0:
            raise InvalidConfigError("quantity_step * contract_multiplier must fit a positive float64")
        if isinstance(self.lot_size, bool) or not isinstance(self.lot_size, int) or not 0 < self.lot_size <= _INT64_MAX:
            raise InvalidConfigError("lot_size must be a positive int64 integer")
        for name in ("base_currency", "quote_currency"):
            value = getattr(self, name)
            if not isinstance(value, str) or _CURRENCY.fullmatch(value) is None:
                raise InvalidConfigError(f"{name} must be an uppercase currency identifier")
        currency = self.base_currency if self.inverse else self.quote_currency
        if self.settlement_currency is not None and self.settlement_currency != currency:
            raise InvalidConfigError(
                "settlement must be in base currency for inverse contracts, quote currency otherwise"
            )
        object.__setattr__(self, "settlement_currency", currency)
        if self.shortable is None:
            object.__setattr__(self, "shortable", self.contract_type != ContractType.SPOT)
        elif not isinstance(self.shortable, bool):
            raise InvalidConfigError("shortable must be boolean")

    @property
    def key(self) -> str:
        return f"{self.venue}/{self.symbol}"

    @property
    def inverse(self) -> bool:
        return self.contract_type == ContractType.INVERSE

    @property
    def cash_factor(self) -> float:
        return float(self.quantity_step) * float(self.contract_multiplier)

    @property
    def tick_micro(self) -> int:
        """Integer micro-price grid for the optional fast text decoder."""
        value = self.tick_size * 1_000_000
        if value != value.to_integral_value() or not 1 <= value <= _INT64_MAX:
            raise InvalidConfigError("this instrument needs the exact-decimal text decoder")
        return int(value)

    @property
    def fast_text_compatible(self) -> bool:
        if self.quantity_step != 1:
            return False
        try:
            _ = self.tick_micro
        except InvalidConfigError:
            return False
        return True

    @property
    def tick_float(self) -> float:
        return float(self.tick_size)

    def price_of(self, ticks: int) -> float:
        return ticks * self.tick_float

    def quantity_of(self, units: int) -> Decimal:
        return units * self.quantity_step

    @classmethod
    def from_mapping(cls, doc: Mapping[str, Any], *, allow_unknown: bool = False) -> InstrumentSpec:
        """Load explicit JSON metadata without coercing quantities or booleans."""
        allowed = {
            "venue",
            "symbol",
            "tick_size",
            "lot_size",
            "contract_type",
            "quantity_step",
            "contract_multiplier",
            "base_currency",
            "quote_currency",
            "settlement_currency",
            "shortable",
        }
        required = {"venue", "symbol", "tick_size", "contract_type"}
        if not isinstance(doc, Mapping) or required - doc.keys() or (not allow_unknown and doc.keys() - allowed):
            raise InvalidConfigError("instrument metadata has missing or unknown fields")
        values = {key: value for key, value in doc.items() if key in allowed}
        for key in ("tick_size", "quantity_step", "contract_multiplier"):
            if key in values:
                if not isinstance(values[key], str):
                    raise InvalidConfigError(f"instrument {key} must be a decimal string")
                try:
                    values[key] = Decimal(values[key])
                except DecimalException as exc:
                    raise InvalidConfigError(f"invalid instrument {key}") from exc
        try:
            values["contract_type"] = ContractType(values["contract_type"])
        except (ValueError, TypeError) as exc:
            raise InvalidConfigError("unknown instrument contract_type") from exc
        return cls(**values)

    def to_mapping(self) -> dict[str, str | int | bool | None]:
        return {
            "venue": self.venue,
            "symbol": self.symbol,
            "tick_size": str(self.tick_size),
            "lot_size": self.lot_size,
            "contract_type": self.contract_type.value,
            "quantity_step": str(self.quantity_step),
            "contract_multiplier": str(self.contract_multiplier),
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "settlement_currency": self.settlement_currency,
            "shortable": self.shortable,
        }


# Fictitious instruments are conveniences for examples and tests, never an exchange registry.
DEMO_INVERSE_A = InstrumentSpec("venue_a", "ALPHA-USD", Decimal("0.5"), 100, ContractType.INVERSE, base_currency="COIN")
DEMO_INVERSE_B = InstrumentSpec("venue_b", "ALPHA-PERP", Decimal("0.5"), 10, ContractType.INVERSE, base_currency="COIN")
DEMO_SPOT = InstrumentSpec("demo", "SHARE-USD", Decimal("0.01"), contract_type=ContractType.SPOT)
DEMO_LINEAR = InstrumentSpec(
    "demo", "INDEX-FUT", Decimal("0.25"), contract_type=ContractType.LINEAR, contract_multiplier=Decimal("50")
)

INSTRUMENTS: dict[str, InstrumentSpec] = {
    spec.key: spec for spec in (DEMO_INVERSE_A, DEMO_INVERSE_B, DEMO_SPOT, DEMO_LINEAR)
}
