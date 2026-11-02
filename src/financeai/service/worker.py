"""Worker-process entry point (started by the coordinator with the ``spawn`` context).

This module is deliberately light: unpickling :func:`worker_main` in a fresh interpreter must
not import NumPy/Numba, so that the native thread limits in :data:`THREAD_ENV` are set
before any BLAS/OpenMP/Numba runtime initialises. The worker then becomes the leader of a
new process group (so the coordinator can terminate it *and* its descendants), imports the
job runner, and reports over its private one-way pipe:

``("progress", {...})`` at safe checkpoints, then exactly one of ``("completed", descriptor)``,
``("failed", {"code", "message"})`` or ``("cancelled", {})``.

A worker never publishes: it writes only into its attempt staging directory.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Protocol

from financeai.core.errors import FinanceAIError

THREAD_ENV: Final = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}
_MESSAGE_LIMIT: Final = 300


class JobCancelled(Exception):
    """Raised at a checkpoint once cancellation was requested (or the coordinator vanished)."""


class _Sender(Protocol):
    def send(self, obj: Any) -> None: ...

    def close(self) -> None: ...


class _Flag(Protocol):
    def is_set(self) -> bool: ...


class WorkerContext:
    """Cooperative-cancellation checkpoint and progress reporter handed to the job runner."""

    def __init__(self, cancel: _Flag, conn: _Sender, parent_pid: int) -> None:
        self._cancel = cancel
        self._conn = conn
        self._parent_pid = parent_pid

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set() or os.getppid() != self._parent_pid

    def checkpoint(self, phase: str, completed: int, total: int) -> None:
        if self.cancelled:
            raise JobCancelled(phase)
        self._conn.send(
            ("progress", {"phase": str(phase), "completed_steps": int(completed), "total_steps": int(total)})
        )


Runner = Callable[[Mapping[str, Any], Path, Callable[[str, int, int], None]], Mapping[str, Any]]


def limit_native_threads() -> None:
    """Force single-threaded native pools for this process (call before importing NumPy)."""
    os.environ.update(THREAD_ENV)


def _load_runner(blob: bytes | None) -> Runner:
    if blob is None:
        from financeai.service.jobs import execute_job

        return execute_job
    runner: Runner = pickle.loads(blob)  # produced by the trusted coordinator, never by a request
    return runner


def worker_main(
    payload: Mapping[str, Any], staging: str, cancel: _Flag, conn: _Sender, runner_blob: bytes | None = None
) -> None:
    limit_native_threads()
    if hasattr(os, "setsid"):
        with contextlib.suppress(OSError):  # already a group leader
            os.setsid()
    ctx = WorkerContext(cancel, conn, os.getppid())
    try:
        runner = _load_runner(runner_blob)
        descriptor = runner(payload, Path(staging), ctx.checkpoint)
        conn.send(("completed", dict(descriptor)))
    except JobCancelled:
        conn.send(("cancelled", {}))
    except FinanceAIError as exc:
        conn.send(("failed", {"code": exc.code, "message": exc.message[:_MESSAGE_LIMIT]}))
    except Exception as exc:
        # Report the type only: messages may carry paths or data; the traceback goes to stderr.
        logging.getLogger("financeai.service.worker").exception("worker failed")
        conn.send(("failed", {"code": "worker_error", "message": type(exc).__name__}))
    finally:
        conn.close()
