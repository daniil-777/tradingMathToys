"""Exact additive-lattice tests independent of any trading relaxation."""

from __future__ import annotations

import itertools
from dataclasses import replace
from fractions import Fraction

import pytest

from financeai.core.errors import InvalidConfigError
from financeai.research.cash_denominator import certify_cash_denominator, verify_cash_denominator


def test_denominator_obstruction_proves_strictly_smaller_upper_bound():
    increments = [Fraction(1, 3), Fraction(-1, 7)]
    result = certify_cash_denominator(Fraction(1, 10), increments)
    assert result.common_denominator == 21
    assert result.lattice_upper_bound == Fraction(2, 21)
    assert result.strict_gap == Fraction(1, 210)
    assert result.missing_denominator_factor == 10
    assert result.excludes_supplied_bound
    verify_cash_denominator(result, increments)


def test_on_lattice_is_inconclusive_even_when_bound_is_actually_unattainable():
    result = certify_cash_denominator(Fraction(1), [Fraction(2)])
    assert result.strict_gap == 0
    assert not result.excludes_supplied_bound
    # Odd integers are not sums of2s, but a denominator-only test cannot see that.
    assert result.lattice_upper_bound == 1


@pytest.mark.parametrize("upper", [Fraction(1, 10), Fraction(5, 9), Fraction(-1, 10)])
def test_exhaustive_integer_combinations_below_upper_remain_below_lattice_floor(upper):
    increments = [Fraction(2, 15), Fraction(-3, 14), Fraction(5, 22)]
    result = certify_cash_denominator(upper, increments)
    for amounts in itertools.product(range(-6, 7), repeat=3):
        cash = sum((q * x for q, x in zip(amounts, increments, strict=True)), Fraction(0))
        assert (cash * result.common_denominator).denominator == 1
        if cash <= upper:
            assert cash <= result.lattice_upper_bound
    verify_cash_denominator(result, increments)


def test_prime_exponent_obstruction_survives_arbitrarily_many_cash_terms():
    increments = [Fraction(7, 2**14 * 5**9 * 101), Fraction(-11, 3 * 5**8 * 107)]
    upper = Fraction(11105167525524270, 10**14)
    result = certify_cash_denominator(upper, increments)
    assert result.missing_denominator_factor == 5**4
    assert result.strict_gap > 0
    verify_cash_denominator(result, increments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("common_denominator", 3),
        ("lattice_upper_bound", Fraction(1, 10)),
        ("strict_gap", Fraction(0)),
        ("missing_denominator_factor", 1),
        ("generators_checked", 1),
    ],
)
def test_tampered_certificates_are_rejected(field, value):
    increments = [Fraction(1, 3), Fraction(-1, 7)]
    result = certify_cash_denominator(Fraction(1, 10), increments)
    with pytest.raises((AssertionError, InvalidConfigError)):
        verify_cash_denominator(replace(result, **{field: value}), increments)


def test_nonminimal_common_denominator_can_be_a_valid_weaker_certificate():
    result = certify_cash_denominator(Fraction(1, 10), [Fraction(1, 3)])
    result = replace(result, common_denominator=21, lattice_upper_bound=Fraction(2, 21), strict_gap=Fraction(1, 210))
    verify_cash_denominator(result, [Fraction(1, 3)])


@pytest.mark.parametrize("upper,generators", [(0.1, [Fraction(1, 3)]), (Fraction(1, 10), [0.1]), (Fraction(1), [])])
def test_reject_inexact_or_missing_inputs(upper, generators):
    with pytest.raises(InvalidConfigError):
        certify_cash_denominator(upper, generators)
