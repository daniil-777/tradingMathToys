"""Test-only job runners executed inside spawned worker processes.

They are injected through ``create_app(..., job_runner=...)`` (trusted Python, never HTTP)
and pickled by reference, so this module must stay importable without NumPy: the worker
sets its native-thread limits before the runner is loaded. The state root is derived from
the attempt directory ``STATE_ROOT/staging/<run_id>/<attempt_id>``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

Checkpoint = Callable[[str, int, int], None]
_WAIT_S = 120.0


def state_root_of(staging: Path) -> Path:
    return staging.parents[2]


def release_file(state_root: Path, run_id: str) -> Path:
    return state_root / f"release-{run_id}"


def _real(payload: Mapping[str, Any], staging: Path, checkpoint: Checkpoint) -> Mapping[str, Any]:
    from financeai.service.jobs import execute_job

    return execute_job(payload, staging, checkpoint)


def blocking_runner(payload: Mapping[str, Any], staging: Path, checkpoint: Checkpoint) -> Mapping[str, Any]:
    """Cooperative: leaves partial output, checkpoints until cancelled or released, then runs the job."""
    partial = staging / "partial.txt"
    partial.write_text("incomplete output")
    release = release_file(state_root_of(staging), payload["run_id"])
    deadline = time.monotonic() + _WAIT_S
    while not release.exists():
        checkpoint("blocked", 0, 1)
        if time.monotonic() > deadline:
            raise RuntimeError("blocking runner was never released")
        time.sleep(0.02)
    partial.unlink()
    return _real(payload, staging, checkpoint)


def stubborn_runner(payload: Mapping[str, Any], staging: Path, checkpoint: Checkpoint) -> Mapping[str, Any]:
    """Never reaches a checkpoint (like a long native fit): only termination stops it."""
    (staging / "partial.txt").write_text("incomplete output")
    time.sleep(_WAIT_S)
    return {"outputs": []}


def child_runner(payload: Mapping[str, Any], staging: Path, checkpoint: Checkpoint) -> Mapping[str, Any]:
    """Starts a long-lived child process, records its PID, then ignores cancellation."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    pid_file = state_root_of(staging) / f"child-{payload['run_id']}.pid"
    pid_file.write_text(str(child.pid))
    time.sleep(_WAIT_S)
    return {"outputs": []}


def crash_runner(payload: Mapping[str, Any], staging: Path, checkpoint: Checkpoint) -> Mapping[str, Any]:
    """Dies abruptly (no message) after writing partial output."""
    (staging / "partial.txt").write_text("incomplete output")
    os._exit(3)


def probe_runner(payload: Mapping[str, Any], staging: Path, checkpoint: Checkpoint) -> Mapping[str, Any]:
    """Records the worker's environment before any heavy import, then runs the real job."""
    probe = {
        "thread_env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
        "numpy_loaded_before_job": "numpy" in sys.modules,
        "fastapi_loaded": "fastapi" in sys.modules,
        "process_group_leader": os.getpgid(0) == os.getpid(),
        "pid": os.getpid(),
    }
    (state_root_of(staging) / f"probe-{payload['run_id']}.json").write_text(json.dumps(probe))
    return _real(payload, staging, checkpoint)
