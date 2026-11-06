"""Dynamic C-ABI bridge: compile C++ kernels and execute them via ctypes.

Uses Clang++ ``-O3 -std=c++20 -fPIC -shared`` when available, falling back to
G++. The zero-copy :class:`SimulationBuffer` ctypes struct mirrors the C
``SimulationBuffer`` layout exactly.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional


class SimulationBuffer(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("dt", ctypes.c_double),
        ("steps", ctypes.c_uint32),
        ("data", ctypes.POINTER(ctypes.c_double)),
    ]


class DynamicCABIBridge:
    """Compiles C++ source to a shared library and drives ``execute_step``."""

    def __init__(self, workspace: Path, compiler: Optional[str] = None):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.compiler = compiler or self._detect_compiler()
        self._cache: Dict[str, Path] = {}

    @staticmethod
    def _detect_compiler() -> str:
        for cand in ("clang++", "g++", "c++"):
            if shutil.which(cand):
                return cand
        raise RuntimeError("no C++ compiler found (clang++/g++/c++)")

    @staticmethod
    def cache_key(cpp_code: str) -> str:
        return hashlib.blake2b(cpp_code.encode(), digest_size=16).hexdigest()

    def compile(self, cpp_code: str) -> Path:
        key = self.cache_key(cpp_code)
        cached = self._cache.get(key)
        if cached and cached.exists():
            return cached
        cpp_path = self.workspace / f"kernel_{key}.cpp"
        so_path = self.workspace / f"kernel_{key}.so"
        cpp_path.write_text(cpp_code)
        cmd = [self.compiler, "-O3", "-std=c++20", "-fPIC", "-shared",
               str(cpp_path), "-o", str(so_path)]
        if "clang" in self.compiler:
            cmd[1:1] = ["-march=native"]
        cmd.insert(-2, "-ffast-math")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(
                f"C-ABI compilation failed ({self.compiler}):\n{proc.stderr.strip()[:2000]}"
            )
        self._cache[key] = so_path
        return so_path

    def load(self, so_path: Path):
        lib = ctypes.CDLL(str(so_path))
        lib.execute_step.argtypes = [ctypes.POINTER(SimulationBuffer)]
        lib.execute_step.restype = None
        return lib

    def run(self, cpp_code: str, initial_state: List[float], dt: float,
            steps: int) -> Dict[str, object]:
        """Compile (cached), load, and run ``steps`` iterations."""
        start = time.perf_counter()
        so_path = self.compile(cpp_code)
        lib = self.load(so_path)
        n = len(initial_state)
        data = (ctypes.c_double * n)(*initial_state)
        buf = SimulationBuffer(size=n, dt=dt, steps=0, data=data)
        for _ in range(steps):
            lib.execute_step(ctypes.byref(buf))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "state": list(data),
            "steps": int(buf.steps),
            "execution_time_ms": elapsed_ms,
            "bundle_hash": self.cache_key(cpp_code),
        }
