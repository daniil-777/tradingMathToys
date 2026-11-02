"""Multi-scale sub-TTL consumption-memory shadow upper bound certificate.

Why uniform 1-second (100-bucket) TTL blocks leave removable slack:
In :func:`financeai.research.ttl_shadow_bound.certify_ttl_shadow`, any disjoint partition of
the bucket index set ``{0, 1, ..., N - 1}`` into contiguous blocks ``B_r = [first_r, stop_r)``
of duration ``(stop_r - first_r) * bucket_ns <= consumption_ttl_ns`` is a valid upper bound on
terminal-flat local-engine profit: within every block ``B_r``, any two fills at the same
``(side, px_ticks)`` occur within ``< consumption_ttl_ns`` of each other and therefore cannot
exceed ``max_{s in B_r} visible_depth(s, side, px_ticks)`` once the 32-slot per-side consumption
memory guard holds.

When uniform 100-bucket (1.0-second) blocks are used:
1. High-activity 1.0-second windows often expose more than 32 distinct ``(side, px_ticks)`` pairs
   across the retained book ladder and must be left unprotected.
2. Over a full 1.0-second window, the block shadow extrema ``[min_{b in B} u_b, max_{b in B} u_b]``
   drift much wider than over 20ms-200ms sub-blocks, weakening the per-level TTL residual cap.

Conversely, during stationary periods where the shadow price is flat and the book exposes <= 32
levels, a longer block (e.g. 20, 50, or 100 buckets) caps repeated consumption of a static quote
across more buckets than short sub-blocks.

This module evaluates :func:`certify_ttl_shadow` across a predeclared tuple of sub-TTL block
scales ``scales_buckets`` (all ``<= consumption_ttl_ns // bucket_ns``) and solves the exact
shortest-path dynamic program on the directed acyclic graph (DAG) of certified blocks from bucket
``0`` to bucket ``N``. Because every path from ``0`` to ``N`` in this DAG is a contiguous,
non-overlapping partition of ``{0, ..., N - 1}`` into certified sub-TTL blocks, the resulting
partition is itself a single valid non-overlapping TTL block certificate whose objective is
provably ``<= min_{s in scales_buckets} objective(s) <= base_depth_objective_units``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from financeai.core.jit import njit
from financeai.domains.market.tape import BookTape
from financeai.research.ttl_shadow_bound import TTLShadowBound, certify_ttl_shadow

DEFAULT_MULTISCALE_BUCKETS: tuple[int, ...] = (2, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class MultiscaleTTLShadowBound:
    """Certified non-overlapping multi-scale sub-TTL shadow upper bound."""

    total_variation_units: int
    base_depth_objective_units: int
    objective_units: int
    rate_scale: int
    consumption_ttl_ns: int
    scales_buckets: tuple[int, ...]
    consumption_slots_per_side: int
    single_scale_objectives: dict[int, int]
    scale_block_counts: dict[int, int]
    strictly_improved_blocks: int
    unprotected_blocks_retained: int
    block_first_bucket: NDArray[np.int64]
    block_stop_bucket: NDArray[np.int64]
    block_scale_buckets: NDArray[np.int64]
    base_block_residual_units: NDArray[np.int64]
    capacity_capped_residual_units: NDArray[np.int64]

    @property
    def best_single_scale_objective_units(self) -> int:
        return min(self.single_scale_objectives.values())

    @property
    def additional_decrease_units(self) -> int:
        return self.base_depth_objective_units - self.objective_units

    @property
    def improvement_over_best_single_scale_units(self) -> int:
        return self.best_single_scale_objective_units - self.objective_units

    @property
    def upper_pnl(self) -> Fraction:
        return Fraction(self.objective_units, self.rate_scale)

    @property
    def base_depth_upper_pnl(self) -> Fraction:
        return Fraction(self.base_depth_objective_units, self.rate_scale)

    @property
    def additional_decrease_pnl(self) -> Fraction:
        return Fraction(self.additional_decrease_units, self.rate_scale)

    @property
    def improvement_over_best_single_scale_pnl(self) -> Fraction:
        return Fraction(self.improvement_over_best_single_scale_units, self.rate_scale)


def _normalize_scales(
    scales_buckets: Sequence[int],
    max_buckets: int,
) -> tuple[int, ...]:
    if not scales_buckets:
        raise ValueError("scales_buckets must be non-empty")
    cleaned = sorted({int(s) for s in scales_buckets})
    if cleaned[0] <= 0:
        raise ValueError("all block scales must be positive integers")
    if cleaned[-1] > max_buckets:
        raise ValueError(f"largest scale {cleaned[-1]} exceeds consumption_ttl_ns // bucket_ns = {max_buckets}")
    return tuple(cleaned)


@njit(cache=False)
def _solve_multiscale_dp(
    n_buckets: int,
    scales_arr: NDArray[np.int64],
    scale_offsets: NDArray[np.int64],
    stops_flat: NDArray[np.int64],
    bases_flat: NDArray[np.int64],
    caps_flat: NDArray[np.int64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    n_scales = len(scales_arr)
    dp_cap = np.full(n_buckets + 1, -1, dtype=np.int64)
    choice_stop = np.zeros(n_buckets, dtype=np.int64)
    choice_scale = np.zeros(n_buckets, dtype=np.int64)
    choice_base = np.zeros(n_buckets, dtype=np.int64)
    choice_cap = np.zeros(n_buckets, dtype=np.int64)
    dp_cap[n_buckets] = 0

    for u in range(n_buckets - 1, -1, -1):
        best_cap = -1
        best_stop = 0
        best_scale = 0
        best_edge_base = 0
        best_edge_cap = 0
        for k in range(n_scales):
            s = int(scales_arr[k])
            if u % s != 0:
                continue
            idx = int(scale_offsets[k]) + (u // s)
            v = int(stops_flat[idx])
            tail_cap = int(dp_cap[v])
            if tail_cap < 0:
                continue
            edge_cap = int(caps_flat[idx])
            cand_cap = edge_cap + tail_cap
            # Tie-break toward larger scale (fewer blocks) when residual is equal.
            if best_cap < 0 or cand_cap < best_cap or (cand_cap == best_cap and s > best_scale):
                best_cap = cand_cap
                best_stop = v
                best_scale = s
                best_edge_base = int(bases_flat[idx])
                best_edge_cap = edge_cap
        if best_cap >= 0:
            dp_cap[u] = best_cap
            choice_stop[u] = best_stop
            choice_scale[u] = best_scale
            choice_base[u] = best_edge_base
            choice_cap[u] = best_edge_cap

    # Count blocks in winning path from 0 to n_buckets
    count = 0
    u = 0
    while u < n_buckets:
        v = int(choice_stop[u])
        if v <= u:
            count = -1
            break
        count += 1
        u = v

    if count <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, empty, empty

    out_first = np.zeros(count, dtype=np.int64)
    out_stop = np.zeros(count, dtype=np.int64)
    out_scale = np.zeros(count, dtype=np.int64)
    out_base = np.zeros(count, dtype=np.int64)
    out_cap = np.zeros(count, dtype=np.int64)

    u = 0
    for i in range(count):
        v = int(choice_stop[u])
        out_first[i] = u
        out_stop[i] = v
        out_scale[i] = choice_scale[u]
        out_base[i] = choice_base[u]
        out_cap[i] = choice_cap[u]
        u = v

    return out_first, out_stop, out_scale, out_base, out_cap


def select_optimal_multiscale_partition(
    scale_bounds: Mapping[int, TTLShadowBound],
) -> MultiscaleTTLShadowBound:
    """Solve the exact DAG shortest-path partition across pre-certified scale bounds."""
    if not scale_bounds:
        raise ValueError("scale_bounds must be non-empty")
    scales = tuple(sorted(int(s) for s in scale_bounds))
    first_cert = scale_bounds[scales[0]]
    n_buckets = int(first_cert.block_stop_bucket[-1])
    tv_units = int(first_cert.total_variation_units)
    base_obj_units = int(first_cert.base_depth_objective_units)
    rate_scale = int(first_cert.rate_scale)
    ttl_ns = int(first_cert.consumption_ttl_ns)
    slots = 32 if first_cert.consumption_slots_per_side is None else int(first_cert.consumption_slots_per_side)

    offsets = [0]
    for s in scales:
        cert = scale_bounds[s]
        if int(cert.phase_buckets) != 0:
            raise ValueError("all scale certificates in multiscale DP must have phase_buckets == 0")
        if int(cert.block_first_bucket[0]) != 0 or int(cert.block_stop_bucket[-1]) != n_buckets:
            raise ValueError("scale certificate bucket domain mismatch")
        if int(cert.total_variation_units) != tv_units:
            raise ValueError("scale certificate total_variation_units mismatch")
        if int(cert.base_depth_objective_units) != base_obj_units:
            raise ValueError("scale certificate base_depth_objective_units mismatch")
        offsets.append(offsets[-1] + len(cert.block_first_bucket))

    scales_arr = np.asarray(scales, dtype=np.int64)
    scale_offsets = np.asarray(offsets[:-1], dtype=np.int64)
    stops_flat = np.concatenate([scale_bounds[s].block_stop_bucket for s in scales])
    bases_flat = np.concatenate([scale_bounds[s].base_block_residual_units for s in scales])
    caps_flat = np.concatenate([scale_bounds[s].capacity_capped_residual_units for s in scales])

    first_arr, stop_arr, scale_arr, base_arr, cap_arr = _solve_multiscale_dp(
        n_buckets,
        scales_arr,
        scale_offsets,
        stops_flat,
        bases_flat,
        caps_flat,
    )
    if len(first_arr) == 0:
        raise RuntimeError("failed to construct a complete partition from bucket 0 to N")

    total_base_res = int(np.sum(base_arr))
    total_cap_res = int(np.sum(cap_arr))
    if tv_units + total_base_res != base_obj_units:
        raise AssertionError("partition base residual sum does not equal parent depth residual")

    scale_counts = {s: int(np.sum(scale_arr == s)) for s in scales}
    improved = int(np.sum(cap_arr < base_arr))
    single_objs = {s: int(scale_bounds[s].objective_units) for s in scales}
    unprotected_upper = min(int(scale_bounds[s].unprotected_blocks_retained) for s in scales)

    result = MultiscaleTTLShadowBound(
        total_variation_units=tv_units,
        base_depth_objective_units=base_obj_units,
        objective_units=tv_units + total_cap_res,
        rate_scale=rate_scale,
        consumption_ttl_ns=ttl_ns,
        scales_buckets=scales,
        consumption_slots_per_side=slots,
        single_scale_objectives=single_objs,
        scale_block_counts=scale_counts,
        strictly_improved_blocks=improved,
        unprotected_blocks_retained=unprotected_upper,
        block_first_bucket=first_arr,
        block_stop_bucket=stop_arr,
        block_scale_buckets=scale_arr,
        base_block_residual_units=base_arr,
        capacity_capped_residual_units=cap_arr,
    )
    verify_multiscale_ttl_partition(result, scale_bounds)
    return result


def verify_multiscale_ttl_partition(
    bound: MultiscaleTTLShadowBound,
    scale_bounds: Mapping[int, TTLShadowBound],
) -> None:
    """Independently verify contiguity, sub-TTL length, block membership, and integer accounting."""
    n_blocks = len(bound.block_first_bucket)
    if n_blocks == 0:
        raise ValueError("partition must contain at least one block")
    if (
        len(bound.block_stop_bucket) != n_blocks
        or len(bound.block_scale_buckets) != n_blocks
        or len(bound.base_block_residual_units) != n_blocks
        or len(bound.capacity_capped_residual_units) != n_blocks
    ):
        raise ValueError("partition array length mismatch")

    first0 = int(bound.block_first_bucket[0])
    if first0 != 0:
        raise ValueError(f"partition must start at bucket 0, got {first0}")
    if not np.array_equal(bound.block_first_bucket[1:], bound.block_stop_bucket[:-1]):
        raise ValueError("partition blocks are not contiguous and non-overlapping")
    lengths = bound.block_stop_bucket - bound.block_first_bucket
    if bool(np.any(lengths <= 0)):
        raise ValueError("all partition blocks must have positive length")
    max_scale = max(bound.scales_buckets)
    if bool(np.any(lengths > max_scale)):
        raise ValueError("partition block length exceeds maximum sub-TTL scale")
    if bool(np.any(bound.capacity_capped_residual_units > bound.base_block_residual_units)):
        raise ValueError("capacity_capped_residual_units exceeds base_block_residual_units")

    # Vectorized per-scale membership verification against each scale certificate.
    matched_total = 0
    for s in bound.scales_buckets:
        if s not in scale_bounds:
            raise ValueError(f"unknown scale {s} in partition")
        mask = bound.block_scale_buckets == s
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        cert = scale_bounds[s]
        first_s = bound.block_first_bucket[mask]
        stop_s = bound.block_stop_bucket[mask]
        base_s = bound.base_block_residual_units[mask]
        cap_s = bound.capacity_capped_residual_units[mask]
        if bool(np.any(first_s % s != 0)):
            raise ValueError(f"block start not aligned to scale {s}")
        idx = first_s // s
        if bool(np.any(idx < 0)) or bool(np.any(idx >= len(cert.block_first_bucket))):
            raise ValueError(f"block index out of range for scale {s}")
        if (
            not np.array_equal(cert.block_first_bucket[idx], first_s)
            or not np.array_equal(cert.block_stop_bucket[idx], stop_s)
            or not np.array_equal(cert.base_block_residual_units[idx], base_s)
            or not np.array_equal(cert.capacity_capped_residual_units[idx], cap_s)
        ):
            raise ValueError(f"partition block values do not match scale {s} certificate")
        matched_total += cnt

    if matched_total != n_blocks:
        raise ValueError("partition contains blocks from uncertified scales")

    base_sum = int(np.sum(bound.base_block_residual_units))
    cap_sum = int(np.sum(bound.capacity_capped_residual_units))
    if bound.total_variation_units + base_sum != bound.base_depth_objective_units:
        raise ValueError("base residual sum mismatch")
    if bound.total_variation_units + cap_sum != bound.objective_units:
        raise ValueError("certified objective sum mismatch")
    if bound.objective_units > bound.best_single_scale_objective_units:
        raise ValueError("multiscale objective exceeds best single-scale objective")


def certify_multiscale_ttl_shadow(
    book: BookTape,
    shadow: NDArray[np.int64],
    *,
    start_ns: int,
    end_ns: int,
    bucket_ns: int = 10_000_000,
    minimum_arrival_gap_ns: int = 10_000_000,
    cap_units: int = 1_000_000,
    lot_units: int = 100,
    tick_size: Decimal | Fraction | str = "0.5",
    taker_fee: Decimal | Fraction | str = "0.00005",
    rate_scale: int = 100_000_000_000_000,
    consumption_ttl_ns: int = 1_000_000_000,
    scales_buckets: Sequence[int] = DEFAULT_MULTISCALE_BUCKETS,
    consumption_slots_per_side: int = 32,
) -> MultiscaleTTLShadowBound:
    """Certify a non-overlapping multi-scale sub-TTL shadow bound across ``scales_buckets``."""
    if int(bucket_ns) <= 0:
        raise ValueError("bucket_ns must be positive")
    max_buckets = int(consumption_ttl_ns) // int(bucket_ns)
    scales = _normalize_scales(scales_buckets, max_buckets)

    scale_bounds: dict[int, TTLShadowBound] = {}
    for s in scales:
        scale_bounds[s] = certify_ttl_shadow(
            book,
            shadow,
            start_ns=start_ns,
            end_ns=end_ns,
            bucket_ns=bucket_ns,
            minimum_arrival_gap_ns=minimum_arrival_gap_ns,
            cap_units=cap_units,
            lot_units=lot_units,
            tick_size=tick_size,
            taker_fee=taker_fee,
            rate_scale=rate_scale,
            consumption_ttl_ns=consumption_ttl_ns,
            block_buckets=s,
            phase_buckets=0,
            consumption_slots_per_side=consumption_slots_per_side,
        )
    return select_optimal_multiscale_partition(scale_bounds)


__all__ = [
    "DEFAULT_MULTISCALE_BUCKETS",
    "MultiscaleTTLShadowBound",
    "certify_multiscale_ttl_shadow",
    "select_optimal_multiscale_partition",
    "verify_multiscale_ttl_partition",
]
