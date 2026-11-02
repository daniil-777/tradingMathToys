# Order-aware shadow profit bounds

`optimize_order_aware_shadow` computes a mathematical profit ceiling from
optimistic bucket rates and depths. It returns a complete integer optimality
witness that `verify_order_aware_shadow` can check without rerunning the optimizer.

## Which trading rules support the ceiling?

The inputs must cover every possible execution book, and the trading engine must:

1. keep each venue's inventory in `[-C,C]`, starting and ending flat;
2. submit at most `C` face in one order;
3. execute at most one order in each time bucket.

The generic local `run_reference` enforces these additional order rules for
**every policy**. It redirects an opposite-sign target to zero, then computes
the order quantity. Two same-sign positions in `[-C,C]` differ by at most `C`.
It also blocks a new order while an acknowledgement is pending. With constant
5 ms order and 5 ms acknowledgement latency, consecutive same-venue arrivals
are at least 10 ms apart. Thus half-open 10 ms buckets contain at most one order.

The original example states a position cap and order latency, without requiring
these two extra controller rules. Consequently this ceiling covers the declared
**local engine**, not every strategy imaginable under the broader example.

## The tightening in one example

For a proposed shadow rate `x`, suppose buying could contribute at most 2 units
of extra cash and selling at most 3. A bucket that permits only one order can
contribute at most `max(2,3)=3` extra units. The previous relaxation charged
`2+3=5`. Choosing the maximum removes that unnecessary allowance.

Let `b[i]` be the optimistic buy credit rate and `s[i]` the optimistic sell cost,
both integers in a caller-declared scale. Set

```
u[i] = min(buy_depth[i], C)
v[i] = min(sell_depth[i], C)
f[i](x) = max(0, u[i]*(b[i]-x), v[i]*(x-s[i]))
J(x) = C*sum(abs(x[i]-x[i-1])) + sum(f[i](x[i]))
```

`b[i]==0` means a missing bucket with zero residual. Integration by parts bounds
the inventory-weighted shadow cash by `C*TV(x)`. The single-order residual is
bounded by `f[i](x)`. Therefore every valid shadow produces a valid ceiling.

## Exact integer optimization and its remaining approximation

The implementation minimizes `J` over integer shadow-rate units. A convex,
piecewise-linear interpolation of each `f` between adjacent integers has the
same values on that grid. Its breakpoints are integers and it has an integer
minimizer after adding total variation, so convex Bellman slope clipping finds
the exact grid optimum.

When `b<=s`, the loss already has integer breakpoints `b,s`. When `b>s`, its
continuous breakpoint is

```
r = (u*b + v*s)/(u+v)
```

The interpolant has two knots, `floor(r)` and `floor(r)+1`. This preserves exact
integer arithmetic. If `a = (u*b+v*s) mod (u+v)` and `w=u+v`, its maximum excess
over the original loss is `a*(w-a)/w`. The library sums outward integer bounds
on these excesses and returns `continuous_objective_gap_units = E`. Thus

```
integer optimum - E <= continuous shadow optimum <= integer optimum
```

This interval concerns the shadow relaxation. Its lower endpoint is not a
feasible trading payoff. A potentially much larger gap to executable profit
remains, because the bucket inputs grant optimistic rates and liquidity.

## Independent verification

The result includes `tv_dual`, with zero endpoints and values in `[-C,C]`.
At a changing shadow coordinate it equals the correctly signed `C`; successive
dual differences lie in the interpolated loss's subgradient interval. These
local integer conditions telescope to `J(y)>=J(x)` for every interpolated
shadow sequence. The verifier computes subgradients from direct loss differences
at `x-1,x,x+1`, independently of the optimizer's two-knot construction and tree.

Static product bounds and every potentially accumulated intercept are checked
before accepting a result. A truncated, mistyped, out-of-domain, overflowing or
invalid witness raises an error.

## Example

```python
import numpy as np
from financeai.research.shadow_bound import (
    optimize_order_aware_shadow, verify_order_aware_shadow,
)

buy = np.array([10, 11, 9], dtype=np.int64)
sell = np.array([12, 10, 11], dtype=np.int64)
depth = np.array([3, 2, 3], dtype=np.int64)
result = optimize_order_aware_shadow(buy, sell, depth, depth, cap_units=3)
assert verify_order_aware_shadow(
    result.shadow_rates, result.tv_dual, buy, sell, depth, depth, 3,
) == result.objective_units
```

Work is `O(N log P)` for `N` buckets and `P` distinct interpolation knots;
storage is `O(N+P)`. Optional Numba acceleration uses the library's usual pure
Python fallback. The API performs no data download, trading or model fitting.

Primary mathematical background: [Boyd and Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)
for maxima of affine functions, convex optimality and dual certificates. The
specific engine containment argument and implementation above are derived here.
