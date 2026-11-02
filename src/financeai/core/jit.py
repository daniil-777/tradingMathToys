"""Optional Numba acceleration with an exact pure-Python fallback.

Kernels are written in the Numba-compatible subset of Python. With the ``fast`` extra
installed they compile to machine code; without it (or with
``FINANCEAI_DISABLE_JIT=1``) the very same functions run as plain Python, which keeps
the core importable everywhere and gives tests a debuggable path.

A *missing* Numba is a supported configuration. A Numba that is installed but fails to
import is a broken environment and is allowed to raise.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _numba_enabled() -> bool:
    if os.environ.get("FINANCEAI_DISABLE_JIT", "") == "1":
        return False
    return importlib.util.find_spec("numba") is not None


HAVE_NUMBA: bool = _numba_enabled()


def njit(*, cache: bool = True, **options: Any) -> Callable[[F], F]:
    """Decorate a kernel; compiles with ``numba.njit`` when available."""

    def decorate(fn: F) -> F:
        if not HAVE_NUMBA:
            return fn
        import numba

        compiled: F = numba.njit(cache=cache, **options)(fn)
        return compiled

    return decorate
