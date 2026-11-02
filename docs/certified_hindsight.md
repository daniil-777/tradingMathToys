# A numerical optimality certificate for the cooldown graph

`financeai.research.certified_hindsight.certify_adaptive_depth` supplies a rigorous
integer upper bound for the **declared restricted graph** optimized by
`adaptive_depth_oracle`. Together with a separately replayed schedule and an exact
lower enclosure `L` of its payoff, its result `U` proves

```
L <= optimum of this graph <= U.
```

Thus no graph-admissible schedule improves the witness by more than `U-L` in settlement currency.
This is an epsilon optimality certificate, not a claim that the selected path is
exactly the unique maximizing path. The graph is a subset of the full engine's
actions, so its upper bound is **not** an upper bound for the full-engine optimum.

## 1. Fix the model before interpreting the number

The graph fixes the original tapes, scenario, session, position cap, inventory
step, candidate receipt union, limit-price offset and cooldown. Candidate receipt
cohorts use their last observed row; order arrivals use the original arrival
clock. Every nonzero transition must fill fully, cannot cross inventory zero,
and must respect the entry cutoff. Initial and terminal positions are flat.
Selected orders are separated by at least

```
tau = max(order_latency + acknowledgement_latency, consumption_TTL).
```

The waiting edge needs no cooldown. Earlier consumption expires before the next
selected order's arrival, making inventory a sufficient state in this graph.
See [the graph recurrence](adaptive_hindsight.md).

## 2. Exact upward rounding of every whole lot

Let `ell` be one contract lot in quote-currency face units, `p` the integer price ticks, `h` the exact
decimal tick size, and `f` the exact decimal fee. For order side `s` equal to +1
or -1, one lot's exact cash reward is

```
r(s,p) = (s - f) * ell / (p*h).
```

At positive integer scale `S`, use `r_up = ceil(S*r)`. Numerators and denominators
are constructed with Python `Fraction`, then checked before conversion to int64.
The kernel uses quotient and remainder; it never rounds a floating price or
floating reward. Negative sell rewards are also rounded upward, becoming less
negative. For any order containing `m` lots,

```
exact_order_cash <= sum(r_up)/S < exact_order_cash + m/S.
```

Each displayed depth level is rounded to whole lots using the engine's depth
haircut rule. The available-quantity and limit-price predicates are unchanged;
only rewards become optimistic. The certificate therefore optimizes exactly the
same set of admissible paths. For a non-unit depth haircut, this set follows the
library's specified binary-float haircut and subsequent whole-lot floor, while
the cash-flow calculation remains rational.

## 3. The upper Bellman table

For `j(i)` the last candidate at or before `time[i]-tau`, the integer recurrence is

```
U[i,q] = max(U[i-1,q], max_p(U[j(i),p] + reward_up[i,q-p])).
```

Initially only the flat state has value zero. Other states are unreachable.
Induction gives: every path's rounded reward is no larger than its corresponding
table entry, and every finite table entry is achieved by a graph path. Since
each rounded reward dominates its rational reward, `U[last,flat]/S` bounds the
exact graph optimum from above. No floating optimizer result is needed for this
upper-bound argument. Scores use a ring retaining every needed predecessor;
no parents or schedule are stored by the certificate pass.

If `M` bounds the number of selected orders and `C` is the cap, every order has
at most `C/ell` lots: **direct sign-crossing transitions are excluded**. Therefore

```
0 <= certified_upper - graph_optimum <= M*C/(ell*S).
```

The factor would be `2*C` for an interface permitting a direct `-C` to `+C` order.
That is not this graph. The implementation records the conservative bound
`M = min(number_of_candidates, 1 + floor((last_time-first_time)/tau))`.

## 4. Monge acceleration survives rounding

On each book side, marginal exact rewards are nonincreasing as more lots sweep
worse prices. Ceiling is monotone, so the **rounded per-lot** slopes remain
nonincreasing. Summing equal-size inventory steps preserves concavity on each
side. Arbitrary predecessor scores are column constants and preserve the
anti-Monge inequality; they need not be concave.

Independent rounding can destroy the concavity inequality across zero. The
certificate does not assume that inequality: it separately searches buy and
sell triangles within positive and negative inventory blocks. Flat targets are
scanned directly. Each triangular search has ordered feasible predecessor bounds
and monotone leftmost maximizers. An unreachable midpoint does not narrow its
children's bounds. This handles sparse reachability without claiming an
unconditional `O(K log K)` bound in every unreachable-state pattern.

`accelerated=False` uses the simple quadratic integer transition. Tests compare
both implementations on 1,000 random cases, including tied scores, unreachable
states, truncated capacities and deliberately nonconcave zero kinks. Complete
tiny graph optima are separately enumerated using Python rational arithmetic.

## 5. Integer safety is checked before computation

Let `B` bound the absolute rounded reward of one lot over every active arrival
price. Every path prefix has at most `M` orders and therefore absolute integer
reward at most `M*(C/ell)*B`. The wrapper computes this bound with Python big
integers and rejects it above `2^62-1`. It also checks reduced rational coefficient
numerators and every possible price-denominator product. Thus every subsequent
rate, order reward and Bellman sum fits safely in signed int64; the unreachable
sentinel `-2^63` cannot collide with a reachable score. A runtime check additionally
rejects a finite score outside the declared prefix bound.

Excessive scales or unsupported rational ranges are rejected, not silently
wrapped. The default `S=10^12` is sufficient for a small graph-level enclosure on
these tapes. Report the **actual** `U-L`; the conservative rounding limit is a
separate number. Power-of-ten scales give exact decimal serialization.

## 6. Why this does not close the full-engine gap

A verified four-receipt counterexample has a legal buy at 0 ms and sell at 20 ms,
with 5 ms order and 5 ms acknowledgement latency. The buy consumes ask liquidity;
the sell consumes a separate bid-side pool. It earns positive cash and finishes
flat while the one-second-cooldown graph's exact optimum is zero. Finer lot sizes,
partial fills, acknowledgements and other event types can also introduce actions
outside this graph. No claim of global or future-profit optimality follows.

## Verify

The public entry point is `financeai.research.certified_hindsight.certify_adaptive_depth`.
It uses the same market, scenario, session, and quantity-grid inputs as the adaptive optimizer.
The property tests compare its rounded objective with exhaustive small graphs:

```bash
python -m pytest -q tests/research/test_certified_hindsight.py
```
