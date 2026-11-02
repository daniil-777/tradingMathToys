"""Figure tests render off-screen (Agg) and never touch a GUI backend."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
