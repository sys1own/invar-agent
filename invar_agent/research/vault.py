"""Research Vault: isolated storage for Research Agent outputs.

Every completed research run is archived under
``<workspace>/.invar/research_vault/<run_id>/`` with a machine-readable
``metadata.json``, the full ``report.md``, the verified
``unified_module.py`` and the serialized ``blueprint.json``. The vault is
completely separate from the software codebase so the Builder Agent never
confuses research artifacts with real project files.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.research import ResearchBlueprint
from .reporter import _unit_for

VAULT_DIRNAME = ".invar"
VAULT_SUBDIR = "research_vault"


def _safe_run_id(raw: str) -> str:
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "")).strip("._-")
    return run_id[:80] or "run"


def _verified_functions(code: str, limit: int = 12) -> List[str]:
    names: List[str] = []
    for match in re.finditer(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", code or "", re.MULTILINE):
        names.append(match.group(1))
        if len(names) >= limit:
            break
    return names


def _key_metrics(blueprint: ResearchBlueprint, telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract named observables (``obs_*`` telemetry) as verified metrics."""
    metrics: Dict[str, Any] = {}
    obs = {k[4:]: v for k, v in telemetry.items()
           if k.startswith("obs_") and isinstance(v, (int, float))}
    variable_map = getattr(blueprint, "variable_map", {}) or {}
    for name, value in obs.items():
        unit = ""
        for suffix, u in (("_mw_m2", "MW/m²"), ("_mj_m2", "MJ/m²"),
                          ("_km", "km"), ("_g", "Earth g"),
                          ("_ms", "m/s"), ("_kg", "kg"), ("_pa", "Pa")):
            if name.endswith(suffix):
                unit = u
                break
        metrics[name] = {"value": value, "unit": unit, "verified": True}
    # Fall back to the blueprint's declared observables when telemetry lacks them.
    for name in variable_map:
        key = name if name in metrics else name
        if key not in metrics:
            metrics[key] = {"value": None, "unit": "", "verified": False}
    return metrics


def _extracted_constants(blueprint: ResearchBlueprint) -> Dict[str, Any]:
    """SI-normalized constants: blueprint.extracted_constants first, then the
    structured_evidence parameter table, then the legacy initial-state digest."""
    constants: Dict[str, Any] = {}
    explicit = getattr(blueprint, "extracted_constants", None) or {}
    if isinstance(explicit, dict):
        for key, value in list(explicit.items())[:48]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                constants[str(key)] = value
    evidence = getattr(blueprint, "structured_evidence", None)
    if evidence is not None:
        for name, p in (getattr(evidence, "parameters", None) or {}).items():
            if name not in constants and getattr(p, "si_value", None) is not None:
                try:
                    constants[str(name)] = float(p.si_value)
                except (TypeError, ValueError):
                    continue
    if constants:
        return constants
    initial = getattr(blueprint, "initial_state", None)
    if isinstance(initial, dict):
        for key, value in list(initial.items())[:24]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                constants[str(key)] = value
    return constants


def _structured_evidence_summary(blueprint: ResearchBlueprint) -> Dict[str, Any]:
    """Compact, JSON-safe digest of StructuredResearchEvidence for metadata."""
    evidence = getattr(blueprint, "structured_evidence", None)
    if evidence is None:
        return {}
    try:
        data = evidence.model_dump()
    except Exception:  # noqa: BLE001
        return {}
    params = {}
    for name, p in (data.get("parameters") or {}).items():
        params[name] = {"si_value": p.get("si_value"), "si_unit": p.get("si_unit"),
                        "raw_value": p.get("raw_value"), "raw_unit": p.get("raw_unit"),
                        "source": p.get("source")}
    equations = [{"lhs": e.get("lhs"), "rhs_latex": e.get("rhs_latex"),
                  "rhs_python_hint": e.get("rhs_python_hint"),
                  "variables": e.get("variables") or [], "source": e.get("source")}
                 for e in (data.get("equations") or [])[:24]]
    benchmarks = data.get("benchmarks") or []
    return {
        "domains": data.get("domains") or [],
        "recommended_backend": data.get("recommended_backend"),
        "recommended_stepper": data.get("recommended_stepper"),
        "parameters": params,
        "equations": equations,
        "benchmarks": benchmarks[:12],
        "suggested_invariants": data.get("suggested_invariants") or [],
    }


def _evidence_report_section(blueprint: ResearchBlueprint) -> str:
    """Markdown section: Extracted Literature Equations & SI Constants."""
    evidence = getattr(blueprint, "structured_evidence", None)
    if evidence is None:
        return ""
    table = getattr(evidence, "summary_markdown_table", "") or ""
    if not table:
        return ""
    return ("\n\n## Extracted Literature Equations & SI Constants\n\n"
            + table
            + "\n\n*Deterministic zero-LLM extraction from the ingested literature "
              "and the curated reference registry. SI-normalized; Python hints "
              "are `v9_transpiler_friendly`.*\n")


def _contracts_summary(blueprint: ResearchBlueprint) -> List[Dict[str, Any]]:
    """Per-contract boundary conditions so the Builder never bisects ICs.

    Each entry carries the FULL ``initial_state`` (never truncated), ``dt``,
    ``steps`` and the blueprint ``variable_map`` inverted to a slot-to-name
    mapping (``"3": {"name": "peak_alt_km", "unit": "km"}``).
    """
    slot_map: Dict[str, Dict[str, str]] = {}
    variable_map = getattr(blueprint, "variable_map", {}) or {}
    for name, idx in variable_map.items():
        if isinstance(idx, int) and idx >= 0:
            slot_map[str(idx)] = {"name": str(name), "unit": _unit_for(str(name))}
    summary: List[Dict[str, Any]] = []
    for contract in getattr(blueprint, "contracts", []) or []:
        initial = getattr(contract, "initial_state", None) or []
        summary.append({
            "contract_id": str(getattr(contract, "contract_id", "")),
            "target_backend": str(getattr(contract, "target_backend", "")),
            "dt": float(getattr(contract, "dt", 0.0) or 0.0),
            "steps": int(getattr(contract, "steps", 0) or 0),
            "initial_state": [float(v) for v in initial],
            "variable_map": dict(slot_map),
        })
    return summary


def _governing_equations(blueprint: ResearchBlueprint) -> List[str]:
    """Surface contract descriptions as the governing-equation digest."""
    equations: List[str] = []
    for contract in getattr(blueprint, "contracts", []) or []:
        desc = getattr(contract, "description", "") or getattr(contract, "contract_id", "")
        if desc:
            equations.append(str(desc)[:300])
        if len(equations) >= 12:
            break
    return equations


class ResearchVaultManager:
    """CRUD + search over ``<workspace>/.invar/research_vault/``."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.root = self.workspace_root / VAULT_DIRNAME / VAULT_SUBDIR

    # -- paths ---------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.root / _safe_run_id(run_id)

    def _metadata_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "metadata.json"

    # -- write ----------------------------------------------------------------
    def register_run(self, *, run_id: str, title: str, blueprint: ResearchBlueprint,
                     report_md: str, status: str, telemetry: Optional[Dict[str, Any]] = None,
                     target_backend: str = "") -> Dict[str, Any]:
        """Archive one completed research run into the vault."""
        telemetry = dict(telemetry or {})
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        module_code = blueprint.unified_module_code or blueprint.code_template or ""
        evidence_section = _evidence_report_section(blueprint)
        (directory / "unified_module.py").write_text(module_code, encoding="utf-8")
        (directory / "report.md").write_text(
            (report_md or "") + evidence_section, encoding="utf-8")
        (directory / "blueprint.json").write_text(
            blueprint.model_dump_json(indent=2), encoding="utf-8")
        feasibility = blueprint.feasibility
        verdict = getattr(getattr(feasibility, "verdict", None), "value",
                          getattr(feasibility, "verdict", "")) if feasibility else ""
        stages = len(getattr(blueprint, "verified_layers", []) or [])
        metadata = {
            "run_id": _safe_run_id(run_id),
            "title": title or blueprint.title,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": str(verdict or ""),
            "status": status,
            "target_backend": target_backend,
            "stages_completed": stages,
            "key_metrics": _key_metrics(blueprint, telemetry),
            "extracted_constants": _extracted_constants(blueprint),
            "governing_equations": _governing_equations(blueprint),
            "contracts_summary": _contracts_summary(blueprint),
            "verified_functions": _verified_functions(module_code),
            "structured_evidence": _structured_evidence_summary(blueprint),
        }
        self._metadata_path(run_id).write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return metadata

    def mark_status(self, run_id: str, status: str) -> bool:
        """Update only the status field of an existing vault entry."""
        path = self._metadata_path(run_id)
        if not path.is_file():
            return False
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(meta, dict):
            return False
        meta["status"] = status
        try:
            path.write_text(json.dumps(meta, indent=2, default=str),
                            encoding="utf-8")
        except OSError:
            return False
        return True

    def recover_orphans(self) -> List[str]:
        """Rebuild metadata for vault dirs that have artifacts but no valid
        metadata.json (e.g. a crash between artifact writes)."""
        recovered: List[str] = []
        if not self.root.is_dir():
            return recovered
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            run_id = directory.name
            meta = self._read_metadata(directory)
            has_artifacts = any((directory / name).is_file()
                                for name in ("report.md", "blueprint.json",
                                             "unified_module.py"))
            if meta is not None or not has_artifacts:
                continue
            bp_raw = ""
            bp_path = directory / "blueprint.json"
            stages = 0
            if bp_path.is_file():
                try:
                    bp_raw = bp_path.read_text(encoding="utf-8", errors="replace")
                    bp = ResearchBlueprint.model_validate_json(bp_raw)
                    stages = len(bp.verified_layers or [])
                except Exception:  # noqa: BLE001 - legacy blueprints may predate schema
                    stages = 0
            module_path = directory / "unified_module.py"
            module_code = ""
            if module_path.is_file():
                try:
                    module_code = module_path.read_text(encoding="utf-8",
                                                        errors="replace")
                except OSError:
                    module_code = ""
            metadata: Dict[str, Any] = {
                "run_id": _safe_run_id(run_id),
                "title": run_id.replace("_", " ").title(),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "verdict": "",
                "status": "PARTIALLY_VERIFIED" if stages else "LEGACY",
                "target_backend": "",
                "stages_completed": stages,
                "key_metrics": {},
                "extracted_constants": {},
                "governing_equations": [],
                "verified_functions": _verified_functions(module_code),
                "recovered": True,
            }
            if bp_raw:
                try:
                    bp = ResearchBlueprint.model_validate_json(bp_raw)
                    metadata["title"] = bp.title
                    feasibility = bp.feasibility
                    metadata["verdict"] = str(
                        getattr(getattr(feasibility, "verdict", None), "value",
                                getattr(feasibility, "verdict", "")) or "")
                    metadata["governing_equations"] = _governing_equations(bp)
                    metadata["extracted_constants"] = _extracted_constants(bp)
                except Exception:  # noqa: BLE001
                    pass
            try:
                (directory / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, default=str), encoding="utf-8")
                recovered.append(run_id)
            except OSError:
                continue
        return recovered

    # -- read -----------------------------------------------------------------
    def _read_metadata(self, directory: Path) -> Optional[Dict[str, Any]]:
        path = directory / "metadata.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def list_runs(self) -> List[Dict[str, Any]]:
        """All vault entries, newest first, with a search snippet."""
        self.recover_orphans()
        runs: List[Dict[str, Any]] = []
        if not self.root.is_dir():
            return runs
        for directory in sorted(self.root.iterdir(), key=lambda p: p.name):
            if not directory.is_dir():
                continue
            meta = self._read_metadata(directory) or {"run_id": directory.name}
            meta = dict(meta)
            meta["run_id"] = meta.get("run_id") or directory.name
            meta["has_report"] = (directory / "report.md").is_file()
            meta["has_module"] = (directory / "unified_module.py").is_file()
            runs.append(meta)
        runs.sort(key=lambda m: str(m.get("created_at", "")), reverse=True)
        for meta in runs:
            meta["snippet"] = self._snippet(meta["run_id"], limit=180)
        return runs

    def _snippet(self, run_id: str, limit: int = 180) -> str:
        path = self.run_dir(run_id) / "report.md"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        # First non-heading prose line is the most informative snippet.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "|", "-", "*", ">")):
                continue
            if len(stripped) > 30 and any(ch.isalpha() for ch in stripped):
                return " ".join(stripped.split())[:limit]
        return " ".join(text.split())[:limit]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        directory = self.run_dir(run_id)
        if not directory.is_dir():
            return None
        meta = self._read_metadata(directory) or {"run_id": _safe_run_id(run_id)}
        meta = dict(meta)
        meta["run_id"] = meta.get("run_id") or _safe_run_id(run_id)

        def _read(name: str) -> str:
            path = directory / name
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""

        blueprint: Any = None
        raw_bp = _read("blueprint.json")
        if raw_bp:
            try:
                blueprint = json.loads(raw_bp)
            except json.JSONDecodeError:
                blueprint = None
        stage_files = sorted(p.name for p in directory.glob("stage_*.py"))
        return {
            "metadata": meta,
            "report_md": _read("report.md"),
            "unified_module_code": _read("unified_module.py"),
            "blueprint": blueprint,
            "stage_files": stage_files,
        }

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Case-insensitive full-text search over titles, reports, equations,
        constants and code; returns matching runs with highlighted snippets."""
        needle = (query or "").strip().lower()
        if not needle:
            return []
        results: List[Dict[str, Any]] = []
        for meta in self.list_runs():
            run_id = meta["run_id"]
            detail = self.get_run(run_id)
            if detail is None:
                continue
            haystacks = {
                "title": str(meta.get("title", "")),
                "report": detail["report_md"],
                "module": detail["unified_module_code"],
                "metadata": json.dumps(meta, default=str),
            }
            snippet = ""
            matched_field = ""
            for field, text in haystacks.items():
                idx = text.lower().find(needle)
                if idx >= 0:
                    matched_field = field
                    start = max(0, idx - 80)
                    end = min(len(text), idx + len(needle) + 120)
                    snippet = text[start:end].replace("\n", " ").strip()
                    snippet = ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")
                    break
            if not snippet:
                # Try per-equation / per-constant matches inside metadata.
                for equation in meta.get("governing_equations", []) or []:
                    if needle in str(equation).lower():
                        matched_field = "equations"
                        snippet = str(equation)[:220]
                        break
                if not snippet:
                    for key, value in (meta.get("extracted_constants", {}) or {}).items():
                        if needle in str(key).lower() or needle in str(value).lower():
                            matched_field = "constants"
                            snippet = f"{key} = {value}"
                            break
            if snippet:
                entry = dict(meta)
                entry["matched_field"] = matched_field
                entry["snippet"] = snippet
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def delete_run(self, run_id: str) -> bool:
        directory = self.run_dir(run_id)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True


# ---------------------------------------------------------------------------
# Legacy artifact migration
# ---------------------------------------------------------------------------

def _latest_stage_module(checkpoint_dir: Path) -> str:
    stages = sorted(checkpoint_dir.glob("stage_*_unified_module.py"))
    if stages:
        try:
            return stages[-1].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""


def _migrate_checkpoint_dir(vault: ResearchVaultManager, checkpoint_dir: Path,
                            report_md: str = "") -> Optional[str]:
    """Archive one legacy checkpoint directory as a vault run."""
    run_id = _safe_run_id(checkpoint_dir.name)
    # Never clobber an existing vault entry: a live run registers its own
    # metadata (PARTIALLY_VERIFIED/VERIFIED) and also exports checkpoints/,
    # which startup migration must not overwrite with a LEGACY stub.
    if vault._read_metadata(vault.run_dir(run_id)) is not None:
        return None
    blueprint_raw = ""
    bp_path = checkpoint_dir / "blueprint.json"
    if bp_path.is_file():
        try:
            blueprint_raw = bp_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            blueprint_raw = ""
    module_code = _latest_stage_module(checkpoint_dir)
    if not module_code:
        module_path = checkpoint_dir / "unified_module.py"
        if module_path.is_file():
            try:
                module_code = module_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                module_code = ""
    if not report_md:
        report_path = checkpoint_dir / "research_report.md"
        if report_path.is_file():
            try:
                report_md = report_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                report_md = ""
    if not (report_md or module_code or blueprint_raw):
        return None
    target = vault.run_dir(run_id)
    target.mkdir(parents=True, exist_ok=True)
    if module_code:
        (target / "unified_module.py").write_text(module_code, encoding="utf-8")
    if report_md:
        (target / "report.md").write_text(report_md, encoding="utf-8")
    if blueprint_raw:
        (target / "blueprint.json").write_text(blueprint_raw, encoding="utf-8")
    # Stage files are preserved for the viewer's stage browser.
    for stage_file in checkpoint_dir.glob("stage_*.py"):
        try:
            shutil.copy2(stage_file, target / stage_file.name)
        except OSError:
            pass
    # Metadata is best-effort: parse the blueprint when possible.
    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "title": run_id.replace("_", " ").title(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": "",
        "status": "LEGACY",
        "target_backend": "",
        "stages_completed": 0,
        "key_metrics": {},
        "extracted_constants": {},
        "governing_equations": [],
        "verified_functions": _verified_functions(module_code),
    }
    if blueprint_raw:
        try:
            bp = ResearchBlueprint.model_validate_json(blueprint_raw)
            metadata["title"] = bp.title
            feasibility = bp.feasibility
            metadata["verdict"] = str(getattr(getattr(feasibility, "verdict", None), "value",
                                               getattr(feasibility, "verdict", "")) or "")
            metadata["stages_completed"] = len(bp.verified_layers or [])
            metadata["governing_equations"] = _governing_equations(bp)
            metadata["extracted_constants"] = _extracted_constants(bp)
            if not module_code and bp.unified_module_code:
                (target / "unified_module.py").write_text(bp.unified_module_code, encoding="utf-8")
                module_code = bp.unified_module_code
                metadata["verified_functions"] = _verified_functions(module_code)
        except Exception:  # noqa: BLE001 - legacy blueprints may predate the schema
            pass
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_id


def migrate_legacy_research_artifacts(workspace_dir: str | Path) -> Dict[str, Any]:
    """Move legacy research outputs into the vault and clean the software root.

    Handles ``<workspace>/research_report.md``, ``<workspace>/research/checkpoints/*``,
    ``<workspace>/checkpoints/*`` and ``<workspace>/research/cabi`` scratch. Returns
    a summary of what was migrated.
    """
    workspace = Path(workspace_dir)
    vault = ResearchVaultManager(workspace)
    migrated: List[str] = []

    # 1. Root-level comprehensive report without a checkpoint.
    root_report = workspace / "research_report.md"
    if root_report.is_file():
        try:
            report_md = root_report.read_text(encoding="utf-8", errors="replace")
        except OSError:
            report_md = ""
        target = vault.run_dir("legacy_report")
        target.mkdir(parents=True, exist_ok=True)
        (target / "report.md").write_text(report_md, encoding="utf-8")
        meta_path = target / "metadata.json"
        meta: Dict[str, Any] = {
            "run_id": "legacy_report",
            "title": "Legacy Research Report",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": "", "status": "LEGACY", "target_backend": "",
            "stages_completed": 0, "key_metrics": {}, "extracted_constants": {},
            "governing_equations": [], "verified_functions": [],
        }
        if meta_path.is_file():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    meta.update(existing)
            except json.JSONDecodeError:
                pass
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            root_report.unlink()
        except OSError:
            pass
        migrated.append("research_report.md")

    # 2. Checkpoint directories (both legacy locations).
    for base in (workspace / "research" / "checkpoints", workspace / "checkpoints"):
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                run_id = _migrate_checkpoint_dir(vault, entry)
                if run_id:
                    migrated.append(run_id)

    # 3. Compiler scratch: move cabi artifacts under .invar/scratch/cabi, then
    #    drop unreferenced binaries so the software root stays clean.
    scratch_target = workspace / ".invar" / "scratch" / "cabi"
    for cabi in (workspace / "research" / "cabi", workspace / "cabi"):
        if cabi.is_dir():
            scratch_target.mkdir(parents=True, exist_ok=True)
            for artifact in cabi.iterdir():
                try:
                    if artifact.suffix in (".so", ".dll", ".dylib", ".o", ".a"):
                        artifact.unlink(missing_ok=True)
                    elif artifact.is_file():
                        shutil.move(str(artifact), scratch_target / artifact.name)
                except OSError:
                    pass

    # 4. Remove emptied legacy directories from the software root.
    for directory in (workspace / "research", workspace / "checkpoints"):
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)

    return {"migrated": migrated, "vault_root": str(vault.root)}
