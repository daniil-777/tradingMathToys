# Proving that a particular cash value is impossible

`financeai.research.cash_denominator` supplies an exact arithmetic test. It can
exclude a particular rational upper bound without finding the optimal strategy.

```python
from fractions import Fraction
from financeai.research.cash_denominator import (
    certify_cash_denominator,
    verify_cash_denominator,
)

# Assume another argument has proved total cash <= 1/10.
# Every allowed cash increment is an integer multiple of 1/3 or -1/7.
increments = [Fraction(1, 3), Fraction(-1, 7)]
certificate = certify_cash_denominator(Fraction(1, 10), increments)
verify_cash_denominator(certificate, increments)

assert certificate.common_denominator == 21
assert certificate.lattice_upper_bound == Fraction(2, 21)
assert certificate.strict_gap == Fraction(1, 210)
assert certificate.excludes_supplied_bound
```

Every sum of these increments is an integer multiple of `1/21`. The largest such
number below `1/10` is `2/21`, so `1/10` cannot be reached or approached arbitrarily
closely from below. This is a rigorous numerical obstruction, even though it does
not say which of those integer multiples are actually reachable.

## Applying it to historical inverse contracts

For every potentially executable ask price, include the exact credit from buying
one venue lot. For every potentially executable bid price, include the exact
debit from selling one lot. Use the exact fees, tick size, and inverse-contract
formula. A fill of several whole lots is an integer multiple of one generator.

The report caller must establish that its support covers every permitted fill and
that the supplied upper bound is valid. The library checks rational arithmetic,
common-denominator divisibility, the exact lattice floor, and the strict gap.
Cash adjustments such as funding require additional generators if the model
contains them. Binary floating ledger encodings are a different numerical model.

## Scope and interpretation

If `excludes_supplied_bound` is false, this test is inconclusive. It does not prove
attainability. The test also ignores execution order, inventory, and timing
constraints, so its lattice may contain many impossible trading outcomes.

A common denominator can have thousands of digits. The resulting strict gap can
be economically negligible. Preserve the exact fractions and describe that
limitation; a microscopic arithmetic improvement is not evidence of a profitable
strategy or a practical solution to a large optimality gap.
