"""Low-level scratch lifecycle shared by research and builder execution."""
from __future__ import annotations

import gc
import shutil
from pathlib import Path


def rollback_overlay(scratch: Path) -> None:
    shutil.rmtree(scratch, ignore_errors=True)


def collect_transients(scratch: Path) -> None:
    rollback_overlay(scratch)
    gc.collect()
