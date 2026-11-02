"""Order-flow imbalance from consecutive book snapshots (Cont, Kukanov & Stoikov; multi-level per
Xu, Gould & Howison), summed over the snapshots received in a trailing window.

Per level ``k`` and consecutive snapshots ``n-1 -> n`` of one venue::

    e_k = 1{Pb_n >= Pb_n-1} qb_n - 1{Pb_n <= Pb_n-1} qb_n-1 - 1{Pa_n <= Pa_n-1} qa_n + 1{Pa_n >= Pa_n-1} qa_n-1

(prices and queues of level ``k`` on each side). The multi-level imbalance is the sum over levels
``1..L``. Snapshot limitations: events between two recorded snapshots are invisible, so a level
whose price moved and came back inside one update contributes nothing; a level missing on either
snapshot (fewer displayed levels, or a cleared book) contributes nothing for that update.

Everything is computed on the venue's own book tape (receipt order) and then read per event of
the merged stream as a trailing-window sum over the updates received in ``(t - w, t]`` - the same
as-of convention as the other signals, so values are causal by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from financeai.core.jit import njit


@njit()
def _ofi_per_update(
    bpx: Any, bsz: Any, apx: Any, asz: Any, nb: Any, na: Any, valid: Any, levels: int, out_e: Any, out_depth: Any
) -> None:
    n = bpx.shape[0]
    for r in range(n):
        e = 0.0
        depth = 0.0
        if r > 0 and valid[r] and valid[r - 1]:
            for k in range(levels):
                if k < nb[r] and k < nb[r - 1]:
                    pb, pb0 = bpx[r, k], bpx[r - 1, k]
                    qb, qb0 = bsz[r, k], bsz[r - 1, k]
                    if pb >= pb0:
                        e += qb
                    if pb <= pb0:
                        e -= qb0
                if k < na[r] and k < na[r - 1]:
                    pa, pa0 = apx[r, k], apx[r - 1, k]
                    qa, qa0 = asz[r, k], asz[r - 1, k]
                    if pa <= pa0:
                        e -= qa
                    if pa >= pa0:
                        e += qa0
        if valid[r]:
            for k in range(levels):
                if k < nb[r]:
                    depth += bsz[r, k]
                if k < na[r]:
                    depth += asz[r, k]
        out_e[r] = e
        out_depth[r] = 0.5 * depth


def ofi_per_update(
    bpx: NDArray[np.int32],
    bsz: NDArray[np.int32],
    apx: NDArray[np.int32],
    asz: NDArray[np.int32],
    nb: NDArray[np.int8],
    na: NDArray[np.int8],
    valid: NDArray[np.bool_],
    levels: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per snapshot: the multi-level OFI of the update that produced it, and the displayed depth
    ``(sum_k qb_k + sum_k qa_k) / 2`` over levels ``1..levels`` at that snapshot."""
    n = bpx.shape[0]
    e = np.empty(n, np.float64)
    depth = np.empty(n, np.float64)
    _ofi_per_update(bpx, bsz, apx, asz, nb, na, valid, levels, e, depth)
    return e, depth


def window_sum_per_event(
    values: NDArray[np.float64],
    book_recv: NDArray[np.int64],
    obs_row: NDArray[np.int64],
    recv: NDArray[np.int64],
    window_ns: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """For each event (observed snapshot ``obs_row[e]`` as of ``recv[e]``): the sum of ``values`` over
    the snapshots received in ``(recv[e] - window_ns, recv[e]]`` and how many there were. NaN where
    no snapshot has been observed yet. Exact for integer-valued inputs below 2**53 (prefix sums)."""
    csum = np.concatenate(([0.0], np.cumsum(values)))
    lo = np.searchsorted(book_recv, recv - window_ns, side="right")  # first snapshot inside the window
    hi = obs_row + 1
    count = np.maximum(hi - lo, 0)
    out = np.where(obs_row >= 0, csum[np.maximum(hi, 0)] - csum[np.minimum(lo, np.maximum(hi, 0))], np.nan)
    return np.ascontiguousarray(out), count
