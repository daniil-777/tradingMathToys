"""Independent small-state enumeration checks the inventory support proof."""

from itertools import pairwise, product

import numpy as np
import pytest

from financeai.core.errors import InvalidConfigError
from financeai.research.inventory_shadow_bound import certify_inventory_shadow


def enumerate_support(shadow: list[int], resolution: int) -> int:
    """Exhaust every feasible path, with inventory represented in sub-cap units."""
    coefficients = [a - b for a, b in pairwise(shadow)]
    best = 0
    for path in product(range(-resolution, resolution + 1), repeat=len(coefficients)):
        full = (0, *path, 0)
        if any(abs(a - b) > resolution for a, b in pairwise(full)):
            continue
        best = max(best, sum(a * q for a, q in zip(coefficients, path, strict=True)))
    return best


@pytest.mark.parametrize(
    "shadow, expected, old",
    [
        ([], 0, 0),
        ([9], 0, 0),
        ([0, 4], 4, 4),
        ([0, 4, 0], 4, 8),
        ([0, 4, 4, 0], 8, 8),
        ([9, 7, 4, 0], 9, 9),
        ([0, 5, 0, 5, 0], 10, 20),
    ],
)
def test_known_support(shadow: list[int], expected: int, old: int) -> None:
    result = certify_inventory_shadow(np.array(shadow, np.int64), cap_units=100)
    assert result.support_units == 100 * expected
    assert result.independent_cap_units == 100 * old
    assert result.improvement_units == 100 * (old - expected)


def test_exhaustive_fractional_paths() -> None:
    # Every 4-bucket shadow on this alphabet; half-cap inventories include
    # fractional vertices of a finer feasible state discretization.
    for shadow in product((-2, 0, 3), repeat=4):
        result = certify_inventory_shadow(np.array(shadow, np.int64), cap_units=7)
        assert result.support_units * 2 == 7 * enumerate_support(list(shadow), 2)


def test_shift_sign_and_large_python_cap() -> None:
    rates = np.array([1, 7, 3, 4, 0, 5], np.int64)
    cap = 10**30
    reference = certify_inventory_shadow(rates, cap_units=cap)
    assert reference == certify_inventory_shadow(-rates, cap_units=cap)
    assert reference == certify_inventory_shadow(rates + 10**15, cap_units=cap)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_cap(value: int) -> None:
    with pytest.raises(InvalidConfigError):
        certify_inventory_shadow(np.array([1, 2], np.int64), cap_units=value)


@pytest.mark.parametrize(
    "rates",
    [
        np.array([1.0, 2.0]),
        np.array([[1, 2]], np.int64),
        np.array([-(2**63), 2**63 - 1], np.int64),
        np.array([0, 2**62 - 1, 0], np.int64),
    ],
)
def test_invalid_array(rates: np.ndarray) -> None:
    with pytest.raises(InvalidConfigError):
        certify_inventory_shadow(rates, cap_units=1)
