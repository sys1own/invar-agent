"""UPG-017 (BUG-031, 032, 033, 034): Dataset schema inference, fresh-unzip pytest proof, and numeric honesty."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from invar_agent.builder.ingestion.schemas import DatasetColumn, DatasetSchema
from invar_agent.ir.units import UnitError, parse_unit

_PAREN_UNIT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*\(([^)]+)\)$")
_SUFFIX_UNIT_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_([A-Za-z0-9^/*-]+)$")
_TIME_NAMES = frozenset({"t", "time", "tau"})


class InvarDomainError(ValueError):
    """UPG-017 / BUG-033: Explicit domain error replacing silent 0.0 in Builder KERNEL_RUNTIME."""


def infer_dataset_schema(
    file_name: str,
    headers: Sequence[str],
    *,
    known_symbols: Optional[set[str]] = None,
) -> Tuple[DatasetSchema, List[str]]:
    """UPG-017 step 1: Infer `DatasetSchema` from headers (`name`, `name (unit)`, `name_unit`).

    Returns `(schema, flags)` where `flags` contains `'DATA_SCHEMA_REQUIRED'` if any
    column header is ambiguous or unparseable.
    """
    cols: List[DatasetColumn] = []
    flags: List[str] = []
    time_col: Optional[str] = None
    time_unit: Optional[str] = None
    known = known_symbols or set()

    for raw_h in headers:
        h = (raw_h or "").strip()
        if not h:
            flags.append("DATA_SCHEMA_REQUIRED: empty header")
            continue
        sym: str
        unit: str = "1"
        m_par = _PAREN_UNIT_RE.match(h)
        if m_par:
            sym = m_par.group(1)
            unit_cand = m_par.group(2).strip()
            try:
                parse_unit(unit_cand)
                unit = unit_cand
            except UnitError:
                flags.append(f"DATA_SCHEMA_REQUIRED: unknown unit {unit_cand!r} in header {h!r}")
                unit = unit_cand
        else:
            m_suf = _SUFFIX_UNIT_RE.match(h)
            if m_suf and h not in known:
                base_cand, unit_cand = m_suf.group(1), m_suf.group(2)
                try:
                    parse_unit(unit_cand)
                    sym, unit = base_cand, unit_cand
                except UnitError:
                    sym, unit = h, "1"
            elif re.match(r"^[A-Za-z][A-Za-z0-9_]*$", h):
                sym, unit = h, "1"
            else:
                flags.append(f"DATA_SCHEMA_REQUIRED: ambiguous header {h!r}")
                sym, unit = h, "1"

        if sym.lower() in _TIME_NAMES and time_col is None:
            time_col = h
            time_unit = unit
        cols.append(DatasetColumn(name=h, symbol=sym, unit=unit))

    return (
        DatasetSchema(file=file_name, time_column=time_col, time_unit=time_unit, columns=cols),
        flags,
    )


def verify_fresh_unzip_pytest(
    zip_path: Path,
    *,
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """UPG-017 step 3: Extract zip into a fresh temp directory outside the workspace and
    run `python -m pytest -q` with `PYTHONPATH` unset and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
    """
    with tempfile.TemporaryDirectory(prefix="invar_fresh_unzip_") as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        # If zip contains a single top-level directory, run pytest inside it
        entries = [p for p in tmp_dir.iterdir() if not p.name.startswith("__MACOSX")]
        run_cwd = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp_dir

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(run_cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            exit_code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            return {
                "export_pytest_exit_code": 124,
                "export_pytest_passed": 0,
                "export_pytest_failed": 1,
                "ok": False,
                "error_code": "EXPORT_FRESH_TEST_FAILED",
                "output": "fresh-unzip pytest timed out",
            }

        m_pass = re.search(r"(\d+)\s+passed", out)
        m_fail = re.search(r"(\d+)\s+failed", out)
        passed = int(m_pass.group(1)) if m_pass else 0
        failed = int(m_fail.group(1)) if m_fail else (0 if exit_code == 0 else 1)
        return {
            "export_pytest_exit_code": exit_code,
            "export_pytest_passed": passed,
            "export_pytest_failed": failed,
            "ok": exit_code == 0,
            "error_code": None if exit_code == 0 else "EXPORT_FRESH_TEST_FAILED",
            "output": out[-2000:],
        }
