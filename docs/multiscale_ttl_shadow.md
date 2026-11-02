# Multi-Scale Sub-TTL Consumption-Memory Shadow Bound (`multiscale_ttl_shadow`)

## 1. Mathematical Foundation

Let $\{0, 1, \dots, N-1\}$ be the shadow-bucket index set of duration $\Delta = \text{bucket\_ns}$ (e.g. $10\text{ ms}$) and let $\tau_{\text{ttl}} = \text{consumption\_ttl\_ns}$ (e.g. $1\text{ s} = 100\Delta$).

By Theorem 1 of `docs/ttl_shadow_bound.md`, for **any** partition $\mathcal{P} = \{B_r\}_{r=0}^{M-1}$ of $\{0, 1, \dots, N-1\}$ into contiguous, non-overlapping half-open bucket blocks $B_r = [\text{first}_r, \text{stop}_r)$ satisfying
$$1 \le \text{stop}_r - \text{first}_r \le \left\lfloor \frac{\tau_{\text{ttl}}}{\Delta} \right\rfloor,$$
any two fills at the same `(side, px_ticks)` inside block $B_r$ occur within strictly less than $\tau_{\text{ttl}}$ of each other. Provided the 32-slot per-side consumption-memory guard holds on $B_r$, the total quantity filled at `(side, px_ticks)` inside $B_r$ cannot exceed $\max_{s \in B_r} \text{visible\_depth}(s, \text{side}, \text{px\_ticks})$.

Therefore, the block residual is bounded above by
$$U(B_r) = \min\!\left(\sum_{b \in B_r} R_b^{\text{depth}},\; R^{\text{ttl}}(B_r)\right),$$
and the global terminal-flat profit is bounded above by
$$\text{Obj}(\mathcal{P}) = C \cdot \text{TV}(u) + \sum_{r=0}^{M-1} U(B_r).$$

## 2. Why shorter blocks can tighten the bound

A long block may expose too many distinct price levels for the consumption-memory guard.
It can also allow the shadow-price range to widen, weakening the residual bound. Shorter
blocks reduce those effects, while longer quiet blocks can cap more repeated consumption.
The partition optimizer chooses among these tradeoffs using certified block costs.

## 3. Exact DAG Shortest-Path Partition Over Certified Sub-TTL Scales

Given a predeclared tuple of sub-TTL block scales $S = (s_1, \dots, s_K)$ with $1 \le s_k \le \lfloor \tau_{\text{ttl}} / \Delta \rfloor$ (default $S = (2, 5, 10, 20, 50, 100)$), `certify_multiscale_ttl_shadow`:

1. Evaluates `certify_ttl_shadow` at each scale $s \in S$ (`phase_buckets = 0`).
2. Solves the exact backward dynamic program from bucket $N$ down to bucket $0$ over the directed acyclic graph (DAG) of all certified blocks $\{[\text{first}_{s,j}, \text{stop}_{s,j})\}_{s \in S}$.
3. Reconstructs the exact minimum-residual contiguous partition $\mathcal{P}^{\star}$ of $\{0, \dots, N-1\}$ and verifies contiguity, sub-TTL length, scale-certificate membership, and exact integer accounting via `verify_multiscale_ttl_partition`.
