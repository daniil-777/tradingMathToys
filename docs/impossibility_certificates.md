# Exact nonattainment certificates

These research utilities distinguish an unattainable numerical upper bound from an unknown optimum. They make no forecast of future returns.

## Rational cash lattice

```python
from fractions import Fraction
from financeai.research.cash_denominator import (
    certify_cash_denominator,
    verify_cash_denominator,
)

# Teaching example: a separate proof has established cash <= 0.33.
generators = [Fraction(1, 4), Fraction(-1, 10)]
certificate = certify_cash_denominator(Fraction(33, 100), generators)
verify_cash_denominator(certificate, generators)
assert certificate.lattice_upper_bound == Fraction(3, 10)
assert certificate.strict_gap == Fraction(3, 100)
```

Supply an already proved upper bound and a complete superset of possible one-lot cash increments. All actual fills must be integer multiples of those generators. If `D` is the common denominator, every cash total is a multiple of `1/D`; the upper bound becomes `floor(U*D)/D`. A zero correction is inconclusive. Even a positive correction need not determine the true optimum. Huge common denominators can produce economically negligible improvements; preserve exact `Fraction` values.

## Coupled inventory support

```python
import numpy as np
from financeai.research.inventory_shadow_bound import certify_inventory_shadow

support = certify_inventory_shadow(np.array([0, 4, 0], np.int64), cap_units=100)
assert support.independent_cap_units == 800
assert support.support_units == 400
```

The utility exactly maximizes the shadow inventory term with flat endpoints, absolute inventory at most `C`, and each bucket's inventory change at most `C`. A three-state dynamic program is exact even for the continuous relaxation. It replaces `C*TV(shadow)` in a separately established shadow certificate; retain residuals for exactly the same shadow and buckets. It does not produce an executable strategy.

The caller must justify at most one order per bucket and the order-face cap. In the audited local engine these follow from one pending order, bucket width no greater than order plus acknowledgement latency, and flattening before reversal. A position cap alone does not imply an order-face cap of the same size.

The direct module APIs accept exact supplied generators and bounds. A certificate is valid only
when the caller establishes coverage of the relevant execution model and inputs.
