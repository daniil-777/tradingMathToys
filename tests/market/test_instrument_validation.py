"""Invalid contract units must fail before they reach integer replay kernels."""

from decimal import Decimal
from typing import Any

import pytest

from financeai.core.errors import InvalidConfigError
from financeai.domains.market.instruments import InstrumentSpec
from financeai.domains.market.scenario import RiskLimits


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("venue", 5),
        ("symbol", True),
        ("lot_size", 1.5),
        ("lot_size", "100"),
        ("lot_size", True),
        ("lot_size", 0),
        ("shortable", "false"),
        ("shortable", 1),
        ("tick_size", Decimal("NaN")),
        ("tick_size", Decimal("sNaN")),
        ("tick_size", Decimal("Infinity")),
        ("tick_size", Decimal("0")),
        ("lot_size", 2**63),
        ("tick_size", Decimal("1e400")),
        ("tick_size", Decimal("1e-1000000")),
    ],
)
def test_invalid_instrument_fields_raise_config_errors(field: str, value: object) -> None:
    fields: dict[str, Any] = {"venue": "venue_a", "symbol": "ALPHA-USD", "tick_size": Decimal("0.5"), "lot_size": 100}
    fields[field] = value
    with pytest.raises(InvalidConfigError):
        InstrumentSpec(**fields)


@pytest.mark.parametrize("cap", [1.5, "100", True, 0, -1])
def test_risk_cap_requires_a_positive_integer(cap: Any) -> None:
    with pytest.raises(InvalidConfigError):
        RiskLimits(max_abs_position_units=cap)
