"""Runtime lossless-acceleration kill switch (`INVAR_ACCEL_CACHE=0`, BUG-040)."""
from __future__ import annotations

import os


def accel_enabled() -> bool:
    """Return True unless `INVAR_ACCEL_CACHE` is explicitly set to `'0'` (or false/off)."""
    raw = os.environ.get("INVAR_ACCEL_CACHE", "").strip().lower()
    return raw not in {"0", "false", "off", "no"}
