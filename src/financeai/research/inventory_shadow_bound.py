"""Exact support of the capped, rate-limited inventory relaxation.

This replaces the independent ``cap * TV(shadow)`` term of an existing shadow
certificate. Its residual bound must be established separately on the SAME
shadow, buckets, cash convention, and admissible engine paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError
from financeai.core.jit import njit

_SAFE = 2**62 - 1


@dataclass(frozen=True)
class InventoryShadowSupport:
    """Integer support values; multiply by the reciprocal rate unit for settlement currency."""

    support_units: int
    independent_cap_units: int
    improvement_units: int
    cap_units: int
    bucket_count: int


@njit(cache=False)
def _support(shadow: NDArray[np.int64]) -> tuple[int, int]:
    # State is Q/C in {-1, 0, +1}. All three first states are reachable from
    # Q_0=0 in one order, so starting their pre-reward values at zero is exact.
    negative, zero, positive, variation = 0, 0, 0, 0
    for i in range(len(shadow) - 1):
        coefficient = int(shadow[i]) - int(shadow[i + 1])
        next_negative = max(negative, zero) - coefficient
        next_zero = max(negative, zero, positive)
        next_positive = max(zero, positive) + coefficient
        negative, zero, positive = next_negative, next_zero, next_positive
        variation += abs(coefficient)
    # Any last state reaches terminal zero with a single cap-sized order.
    return max(negative, zero, positive), variation


def certify_inventory_shadow(
    shadow_rates: NDArray[np.int64],
    *,
    cap_units: int,
) -> InventoryShadowSupport:
    """Compute the exact support in O(n) time and O(1) auxiliary memory.

    For n shadow buckets, maximize sum_{i=1}^{n-1}(x_i-x_{i+1})Q_i with
    Q_0=Q_n=0, |Q_i|<=C, |Q_i-Q_{i-1}|<=C. Normalize by C. The constraint
    matrix is a node-arc incidence matrix transpose with unit bound rows, hence
    totally unimodular. Integral right-hand sides imply an integral optimal vertex:
    the three states -1,0,+1 suffice even for continuous inventory. This is NOT
    an attainable engine strategy: depth, lots, order times and residual rewards
    remain relaxed. The cap on inventory increments must be justified separately.

    Inputs are exact integer shadow rates. Conservative arithmetic checks reject
    arrays whose int64 dynamic program might overflow. Multiplication by the cap
    occurs afterwards with arbitrary-precision Python integers.
    """
    if not isinstance(shadow_rates, np.ndarray) or shadow_rates.dtype != np.dtype(np.int64):
        raise InvalidConfigError("shadow_rates must be an int64 numpy array")
    if shadow_rates.ndim != 1:
        raise InvalidConfigError("shadow_rates must be one-dimensional")
    if isinstance(cap_units, (bool, np.bool_)) or not isinstance(cap_units, (int, np.integer)):
        raise InvalidConfigError("cap_units must be a positive integer")
    cap = int(cap_units)
    if cap <= 0:
        raise InvalidConfigError("cap_units must be a positive integer")
    n = len(shadow_rates)
    if n > 1:
        spread = int(shadow_rates.max()) - int(shadow_rates.min())
        if spread * (n - 1) > _SAFE:
            raise InvalidConfigError("shadow variation lacks safe int64 headroom")
    support, variation = _support(shadow_rates)
    support_units, independent = int(support) * cap, int(variation) * cap
    return InventoryShadowSupport(
        support_units=support_units,
        independent_cap_units=independent,
        improvement_units=independent - support_units,
        cap_units=cap,
        bucket_count=n,
    )
