"""Exact denominator certificates excluding a claimed rational cash value.

If every cash increment is an integer multiple of a supplied rational generator,
all attainable totals have a denominator dividing their common denominator.
This can exclude an exact numerical ceiling without solving the trading problem.
The supplied ceiling must already be a proved upper bound; this module does not
establish market coverage, profitability, attainability, or future performance.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm

from financeai.core.errors import InvalidConfigError


@dataclass(frozen=True)
class CashDenominatorCertificate:
    """A common denominator, exact lattice floor, and nonattainment witness."""

    supplied_upper_bound: Fraction
    common_denominator: int
    lattice_upper_bound: Fraction
    strict_gap: Fraction
    missing_denominator_factor: int
    generators_checked: int

    @property
    def excludes_supplied_bound(self) -> bool:
        return self.strict_gap > 0


def _fraction(value: Fraction, name: str) -> Fraction:
    if not isinstance(value, Fraction):
        raise InvalidConfigError(f"{name} must be an exact Fraction")
    return value


def certify_cash_denominator(
    supplied_upper_bound: Fraction,
    cash_generators: Iterable[Fraction],
) -> CashDenominatorCertificate:
    """Floor an existing upper bound onto an optimistic rational cash lattice.

    Every permitted cash increment must be an integer multiple of a generator.
    For inverse contracts, include each executable ask's whole-lot buy credit and
    each executable bid's whole-lot sell debit, with exact fee and tick size.
    Quantity constraints, order sequencing, and inventory constraints can further
    restrict attainable sums; ignoring them makes this lattice optimistic.

    If D is a common denominator, every attainable total belongs to (1/D) Z.
    Thus an independently established upper bound U can be replaced by floor(U D)/D.
    U itself is excluded exactly when its reduced denominator does not divide D.
    A zero gap means this test is inconclusive, never a proof of attainability.

    Python big integers and Fractions avoid fixed-width overflow. D can be very
    large and the strict gap economically negligible; retain the exact fractions
    and avoid presenting rounded decimals as a materially improved bound.
    """
    upper = _fraction(supplied_upper_bound, "supplied_upper_bound")
    denominator, count = 1, 0
    for value in cash_generators:
        denominator = lcm(denominator, _fraction(value, "cash generator").denominator)
        count += 1
    if count == 0:
        raise InvalidConfigError("at least one cash generator is required")
    product_numerator = upper.numerator * denominator
    quotient, remainder = divmod(product_numerator, upper.denominator)
    return CashDenominatorCertificate(
        supplied_upper_bound=upper,
        common_denominator=denominator,
        lattice_upper_bound=Fraction(quotient, denominator),
        strict_gap=Fraction(remainder, upper.denominator * denominator),
        missing_denominator_factor=upper.denominator // gcd(upper.denominator, denominator),
        generators_checked=count,
    )


def verify_cash_denominator(
    certificate: CashDenominatorCertificate,
    cash_generators: Iterable[Fraction],
) -> None:
    """Independently check coverage, divisibility, lattice floor and strictness.

    The verifier needs every generator again. A source manifest must establish
    their market coverage and the supplied upper bound's validity separately.
    A nonminimal common denominator remains a valid, possibly weaker witness.
    """
    upper = _fraction(certificate.supplied_upper_bound, "supplied_upper_bound")
    denominator = certificate.common_denominator
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise InvalidConfigError("common denominator must be a positive integer")
    count = 0
    for value in cash_generators:
        value = _fraction(value, "cash generator")
        if denominator % value.denominator:
            raise AssertionError("cash generator denominator does not divide certificate")
        count += 1
    if count == 0 or count != certificate.generators_checked:
        raise AssertionError("cash generator coverage/count differs from certificate")
    lower = _fraction(certificate.lattice_upper_bound, "lattice_upper_bound")
    gap = _fraction(certificate.strict_gap, "strict_gap")
    if (lower * denominator).denominator != 1 or not lower <= upper < lower + Fraction(1, denominator):
        raise AssertionError("certificate does not contain the correct lattice floor")
    if gap != upper - lower:
        raise AssertionError("strict gap does not equal supplied bound minus lattice floor")
    missing = upper.denominator // gcd(upper.denominator, denominator)
    if certificate.missing_denominator_factor != missing or (gap > 0) != (missing > 1):
        raise AssertionError("denominator nonattainment witness is inconsistent")
