"""Research Vault: isolated storage for Research Agent outputs.

Every completed research run is archived under
``<workspace>/.invar/research_vault/<run_id>/`` with a machine-readable
``metadata.json``, the full ``report.md``, the verified
``unified_module.py`` and the serialized ``blueprint.json``. The vault is
completely separate from the software codebase so the Builder Agent never
confuses research artifacts with real project files.
"""

from __future__ import annotations

import base64
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

    def set_metadata_fields(self, run_id: str, **fields: Any) -> bool:
        """Merge scalar fields into an existing run's metadata.json."""
        path = self._metadata_path(run_id)
        meta = self._read_metadata(self.run_dir(run_id))
        if meta is None:
            return False
        meta.update({k: v for k, v in fields.items() if isinstance(v, (str, int, float, bool))
                     or v is None})
        try:
            path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            return True
        except OSError:
            return False

    # -- user uploads ---------------------------------------------------------

    def restore_research_bundle(self, files: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Restore an exported ``.invar-research.zip`` bundle (metadata.json +
        report.md at the zip root or one level down) into the vault.

        Returns the restored run_id, or None when the files are not a bundle.
        """
        by_path = {f.get("filename", "").replace("\\", "/").lstrip("/"): f
                   for f in files if f.get("filename")}
        base = ""
        if "metadata.json" not in by_path and "report.md" not in by_path:
            dirs = {p.split("/", 1)[0] for p in by_path if "/" in p}
            if len(dirs) == 1:
                candidate = next(iter(dirs)) + "/"
                if candidate + "metadata.json" in by_path:
                    base = candidate
        if base + "metadata.json" not in by_path or base + "report.md" not in by_path:
            return None
        meta_file = by_path.get(base + "metadata.json")
        try:
            meta = json.loads(_decode_content(meta_file))
        except (ValueError, TypeError):
            return None
        if not isinstance(meta, dict) or not meta.get("run_id"):
            return None
        run_id = _safe_run_id(str(meta["run_id"]))
        directory = self.run_dir(run_id)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        restored: List[str] = []
        for path, item in by_path.items():
            if not path.startswith(base):
                continue
            rel = path[len(base):]
            if _unsafe_member_name(rel):
                continue
            target = directory / rel
            if rel.endswith("/"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.write_bytes(_decode_bytes(item))
                restored.append(rel)
            except (OSError, ValueError):
                continue
        meta["run_id"] = run_id
        meta["origin"] = meta.get("origin") or "research_agent"
        (directory / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return {"run_id": run_id, "restored": True, "files": restored,
                "title": meta.get("title", run_id)}

    def ingest_uploaded_research(self, files: List[Dict[str, Any]],
                                 title: Optional[str] = None,
                                 custom_notes: str = "") -> Dict[str, Any]:
        """First-class vault entry for user-uploaded research.

        ``files`` are ``{"filename", "content", "encoding"}`` dicts (UTF-8 or
        base64). A portable ``.invar-research.zip`` bundle is restored
        directly; any other ``.md``/``.tex``/``.txt``/``.py``/``.json``/``.csv``
        files (or a general ``.zip`` of them) are extracted deterministically
        by :class:`QuantitativeInsightExtractor` into a ``UPLOADED_READY`` run.
        """
        normalized = [_normalize_upload_file(f) for f in (files or [])]
        if not normalized:
            raise ValueError("no research files uploaded")
        total = sum(len(n["content"]) for n in normalized)
        if total > _UPLOAD_MAX_TOTAL_BYTES:
            raise ValueError("uploaded research exceeds the 20 MB limit")

        if len(normalized) == 1 and normalized[0]["filename"].lower().endswith(".zip"):
            members = _zip_members(normalized[0]["content"])
            restored = self.restore_research_bundle(members)
            if restored is not None:
                restored["uploaded_files"] = [normalized[0]["filename"]]
                return restored
            normalized = members or normalized

        base_id = "upload_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        run_id, n = base_id, 1
        while self.run_dir(run_id).exists():
            n += 1
            run_id = f"{base_id}_{n}"
        meta = self._ingest_research_files(run_id, normalized, title=title,
                                           custom_notes=custom_notes)
        return meta

    def _ingest_research_files(self, run_id: str,
                               files: List[Dict[str, Any]],
                               title: Optional[str], custom_notes: str) -> Dict[str, Any]:
        from .extractor import QuantitativeInsightExtractor, parse_equation

        directory = self.run_dir(run_id)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        sources_dir = directory / "uploaded_sources"
        sources_dir.mkdir(parents=True, exist_ok=True)

        documents: List[tuple] = []  # (filename, text)
        python_sources: List[str] = []
        saved: List[str] = []
        for item in files:
            name = item["filename"]
            rel = _safe_source_name(name)
            if rel is None:
                continue
            target = sources_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item["content"])
            saved.append(f"uploaded_sources/{rel.as_posix()}")
            suffix = rel.suffix.lower()
            if suffix == ".py":
                text = _decode_text(item)
                if "simulation_step" in text or "def " in text:
                    python_sources.append(text)
            if suffix in (".md", ".tex", ".txt"):
                documents.append((name, _decode_text(item)))
            elif suffix == ".json":
                documents.append((name, _decode_text(item)))
            elif suffix == ".csv":
                documents.append((name, _decode_text(item)))

        corpus_text = "\n\n".join(f"# {n}\n{t}" for n, t in documents)
        title = (title or "").strip() or _upload_title(documents, custom_notes, run_id)
        extractor = QuantitativeInsightExtractor()
        evidence = extractor.extract(" ".join([title, custom_notes, corpus_text])[:20000])

        # LaTeX / <math> equations from the uploaded documents themselves.
        from .extractor import extract_wikitext_math
        latex_equations = []
        for doc_name, doc_text in documents:
            suffix = _path_suffix(doc_name)
            if suffix in (".md", ".tex", ".txt", ".html"):
                for tex in extract_wikitext_math(doc_text):
                    eq = parse_equation(tex, source=f"user_upload:{doc_name}")
                    if eq is not None:
                        latex_equations.append(eq)
            latex_equations.extend(extractor.equations_from_text(
                doc_text, source=f"user_upload:{doc_name}"))
        if latex_equations:
            merged = list(evidence.equations)
            known = {e.rhs_python_hint for e in merged}
            for eq in latex_equations:
                if eq.rhs_python_hint not in known:
                    merged.append(eq)
                    known.add(eq.rhs_python_hint)
            evidence = evidence.model_copy(update={
                "equations": merged[: extractor.max_equations]})

        # Plain-text (non-LaTeX) equations such as ``dT/dt = -k*(T - T_amb)``.
        extra_equations = []
        for line in corpus_text.splitlines():
            line = line.strip().lstrip("-*># ").strip()
            if "=" not in line or len(line) > 240:
                continue
            lhs, _, rhs = line.partition("=")
            lhs = lhs.strip()
            rhs = rhs.strip().rstrip(".")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\s*/\s*d[A-Za-z0-9_]*)?|[A-Za-z][A-Za-z0-9_]*", lhs):
                continue
            eq = parse_equation(f"${lhs} = {rhs}$", source="user_upload")
            if eq is not None:
                extra_equations.append(eq)
        if extra_equations:
            merged = list(evidence.equations)
            for eq in extra_equations:
                if eq.rhs_python_hint not in {e.rhs_python_hint for e in merged}:
                    merged.append(eq)
            evidence = evidence.model_copy(update={
                "equations": merged[: extractor.max_equations]})

        constants = {name: float(p.si_value) for name, p in evidence.parameters.items()
                     if p.si_value is not None}
        benchmarks = _benchmarks_from_uploads(documents, evidence)
        key_metrics = {}
        for i, b in enumerate(benchmarks[:12]):
            try:
                value = float(b.get("expected_value"))
            except (TypeError, ValueError):
                continue
            key_metrics[f"benchmark_{i}_{b.get('name', 'metric')}"] = {
                "value": value, "unit": b.get("unit", ""), "verified": False}

        module_code = _select_upload_kernel(python_sources, constants, evidence, title)
        for name, value in _module_constants(module_code).items():
            constants.setdefault(name.lower(), value)
        (directory / "unified_module.py").write_text(module_code, encoding="utf-8")
        (directory / "report.md").write_text(
            _upload_report(title, custom_notes, evidence, benchmarks, saved, run_id),
            encoding="utf-8")

        metadata = {
            "run_id": run_id,
            "title": title,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "verdict": "OPTIMAL_DIRECT",
            "status": "UPLOADED_READY",
            "target_backend": evidence.recommended_backend,
            "origin": "user_upload",
            "stages_completed": 0,
            "key_metrics": key_metrics,
            "extracted_constants": constants,
            "governing_equations": [
                f"{e.lhs} = {e.rhs_python_hint or e.rhs_latex}"
                for e in evidence.equations[:24]],
            "contracts_summary": [],
            "verified_functions": _verified_functions(module_code),
            "structured_evidence": _structured_evidence_summary(
                type("_B", (), {"structured_evidence": evidence})()),
            "uploaded_sources": saved,
            "source_file_count": len(saved),
            "recommended_stepper": evidence.recommended_stepper,
        }
        self._metadata_path(run_id).write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return metadata


# ---------------------------------------------------------------------------
# User research upload helpers
# ---------------------------------------------------------------------------

_UPLOAD_MAX_TOTAL_BYTES = 20 * 1024 * 1024
_UPLOAD_MAX_FILE_BYTES = 10 * 1024 * 1024
_UPLOAD_TEXT_SUFFIXES = {".md", ".tex", ".txt", ".py", ".json", ".csv"}


def _decode_bytes(item: Dict[str, Any]) -> bytes:
    content = item.get("content", b"")
    if isinstance(content, str):
        if str(item.get("encoding", "utf-8")).lower() == "base64":
            try:
                return base64.b64decode(content, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("uploaded file content is not valid base64") from exc
        return content.encode("utf-8", errors="replace")
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    return b""


def _decode_content(item: Dict[str, Any]) -> bytes:
    return _decode_bytes(item)


def _decode_text(item: Dict[str, Any]) -> str:
    return _decode_bytes(item).decode("utf-8", errors="replace")


def _member_key(name: str) -> str:
    """Zip-Slip-safe relative path (posix, no absolute/``..`` segments)."""
    parts = [p for p in str(name or "").replace("\\", "/").split("/")
             if p not in ("", ".", "..")]
    if str(name or "").startswith(("/", "\\")) or ":" in (parts[0] if parts else ""):
        return ""
    if not parts:
        return ""
    safe = [re.sub(r"[^A-Za-z0-9_.@+\- ]", "_", p)[:80] for p in parts]
    return "/".join(safe)[:200]


def _normalize_upload_file(raw: Dict[str, Any]) -> Dict[str, Any]:
    name = _member_key(str(raw.get("filename") or raw.get("name") or "upload.bin")) \
        or "upload.bin"
    content = raw.get("content", raw.get("content_base64", ""))
    encoding = str(raw.get("encoding") or "").lower()
    if not encoding and raw.get("content_base64"):
        encoding = "base64"
    data = _decode_bytes({"content": content, "encoding": encoding or "utf-8"})
    if len(data) > _UPLOAD_MAX_FILE_BYTES:
        raise ValueError(f"uploaded file {name!r} exceeds the 10 MB limit")
    return {"filename": name or "upload.bin", "content": data,
            "encoding": encoding or "utf-8"}


def _unsafe_member_name(name: str) -> bool:
    """Zip-Slip guard: reject absolute paths and ``..`` traversal."""
    if not name or name.endswith("/"):
        return False
    p = Path(name.replace("\\", "/"))
    return p.is_absolute() or ".." in p.parts


def _safe_source_name(name: str) -> Optional[Path]:
    key = _member_key(name)
    if not key or _path_suffix(key) not in _UPLOAD_TEXT_SUFFIXES:
        return None
    return Path(*key.split("/"))


def _zip_members(data: bytes, *, max_entries: int = 200) -> List[Dict[str, Any]]:
    import io
    import zipfile

    out: List[Dict[str, Any]] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return out
    with zf:
        for info in zf.infolist()[:max_entries]:
            if info.is_dir() or _unsafe_member_name(info.filename):
                continue
            if info.file_size > _UPLOAD_MAX_FILE_BYTES:
                continue
            if _path_suffix(info.filename) not in _UPLOAD_TEXT_SUFFIXES:
                continue
            raw = zf.read(info)
            out.append({"filename": info.filename,
                        "content": base64.b64encode(raw).decode("ascii"),
                        "encoding": "base64"})
    return out


def _path_suffix(name: str) -> str:
    return Path(str(name).replace("\\", "/")).suffix.lower()


def _upload_title(documents: List[tuple], custom_notes: str, run_id: str) -> str:
    for _, text in documents:
        for line in (text or "").splitlines():
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()[:160] or run_id
    if custom_notes.strip():
        return custom_notes.strip().splitlines()[0][:160]
    return run_id


def _benchmarks_from_uploads(documents: List[tuple], evidence: Any) -> List[Dict[str, Any]]:
    """Benchmark rows from Markdown tables, JSON objects and CSV headers."""
    rows: List[Dict[str, Any]] = [b.model_dump() for b in (evidence.benchmarks or [])]
    for name, text in documents:
        suffix = _path_suffix(name)
        if suffix == ".md" or suffix == ".tex" or suffix == ".txt":
            rows.extend(_markdown_table_benchmarks(text))
        elif suffix == ".json":
            rows.extend(_json_benchmarks(text))
        elif suffix == ".csv":
            rows.extend(_csv_benchmarks(text))
    # De-duplicate by (name, expected_value).
    seen = set()
    unique = []
    for row in rows:
        key = (str(row.get("name", "")), str(row.get("expected_value", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique[:24]


def _markdown_table_benchmarks(text: str) -> List[Dict[str, Any]]:
    rows = []
    for line in (text or "").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        numeric_idx = next((i for i, c in enumerate(cells)
                            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", c)),
                           None)
        if numeric_idx is None or numeric_idx == 0:
            continue
        name = cells[0]
        if name.lower() in ("name", "metric", "quantity", "parameter"):
            continue
        unit = ""
        for cand in cells[numeric_idx + 1:]:
            if cand and not re.fullmatch(
                    r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cand):
                unit = cand
                break
        rows.append({"name": name, "expected_value": cells[numeric_idx],
                     "unit": unit, "source": "user_upload_markdown",
                     "input_regime": "uploaded research", "tolerance": 0.05})
    return rows[:12]


def _json_benchmarks(text: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []
    rows: List[Dict[str, Any]] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 5 or len(rows) >= 12:
            return
        if isinstance(node, dict):
            name = node.get("name") or node.get("metric")
            value = node.get("expected_value", node.get("value"))
            if name is not None and value is not None \
                    and isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append({"name": str(name), "expected_value": float(value),
                             "unit": str(node.get("unit", "") or ""),
                             "source": "user_upload_json",
                             "input_regime": str(node.get("regime", "") or ""),
                             "tolerance": float(node.get("tolerance", 0.05) or 0.05)})
                return
            for v in node.values():
                walk(v, depth + 1)
        elif isinstance(node, list):
            for item in node[:50]:
                walk(item, depth + 1)

    walk(data)
    return rows


def _csv_benchmarks(text: str) -> List[Dict[str, Any]]:
    import csv
    import io

    try:
        rows = list(csv.reader(io.StringIO(text)))
    except Exception:  # noqa: BLE001
        return []
    if len(rows) < 2:
        return []
    header = [h.strip().lower() for h in rows[0]]
    name_col = next((i for i, h in enumerate(header) if h in ("name", "metric")), None)
    value_col = next((i for i, h in enumerate(header)
                      if h in ("value", "expected_value", "expected")), None)
    if name_col is None or value_col is None:
        return []
    unit_col = next((i for i, h in enumerate(header) if h == "unit"), None)
    out = []
    for row in rows[1:12]:
        try:
            value = float(row[value_col])
        except (TypeError, ValueError, IndexError):
            continue
        out.append({"name": row[name_col].strip(), "expected_value": value,
                    "unit": row[unit_col].strip() if unit_col is not None
                    and unit_col < len(row) else "",
                    "source": "user_upload_csv", "input_regime": "",
                    "tolerance": 0.05})
    return out


def _module_constants(code: str) -> Dict[str, float]:
    """Top-level ``UPPER_NAME = <number>`` assignments of an uploaded kernel."""
    import ast
    out: Dict[str, float] = {}
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return out
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.isupper()):
            value = node.value
            sign = 1.0
            if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                value, sign = value.operand, -1.0
            if (isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
                    and not isinstance(value.value, bool)):
                out[node.targets[0].id] = sign * float(value.value)
    return out


def _select_upload_kernel(python_sources: List[str], constants: Dict[str, float],
                          evidence: Any, title: str) -> str:
    """Preserve an uploaded kernel that defines ``simulation_step``; otherwise
    synthesize a deterministic baseline module seeded with the extracted
    constants and equation hints."""
    for src in sorted(python_sources, key=len, reverse=True):
        if "simulation_step" not in src:
            continue
        try:
            compile(src, "<uploaded_kernel>", "exec")
        except SyntaxError:
            continue
        header = (f'"""Verified kernel uploaded by the user (title: {title})."""\n\n'
                  "# invar-agent: preserved verbatim from uploaded_sources.\n\n")
        return header + src.strip() + "\n"

    lines = [
        '"""Baseline unified module synthesized by invar-agent from user-uploaded research."""',
        "",
        "# Extracted SI constants (normalized by the zero-token extractor; use verbatim).",
    ]
    for name, value in sorted(constants.items()):
        py_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper() or "CONST"
        if py_name[0].isdigit():
            py_name = "C_" + py_name
        lines.append(f"{py_name} = {value!r}")
    lines.append("")
    lines.append("# Governing equations (v9_transpiler_friendly Python hints):")
    for eq in (evidence.equations or [])[:12]:
        hint = eq.rhs_python_hint or eq.rhs_latex
        lines.append(f"#   {eq.lhs} = {hint}")
    lines += [
        "",
        "# Placeholder dynamics: the Research Agent (Verify & Extend) or the",
        "# Builder must replace this with the verified numerical integration of",
        "# the governing equations above. The buffer layout follows v9.",
        "",
        "class SimulationBuffer:",
        "    def __init__(self, data, dt):",
        "        self.data = list(data)",
        "        self.size = len(self.data)",
        "        self.dt = dt",
        "        self.steps = 0",
        "",
        "    def __getitem__(self, index):",
        "        return self.data[index]",
        "",
        "    def __setitem__(self, index, value):",
        "        self.data[index] = value",
        "",
        "",
        "def simulation_step(buf, dt):",
        "    # TODO(research): implement d(state)/dt from the governing equations.",
        "    buf.steps += 1",
        "",
        "",
        "def run_simulation(initial_state=None, dt=0.01, steps=1000):",
        "    state = [float(v) for v in (initial_state or [0.0])]",
        "    buf = SimulationBuffer(state, dt)",
        "    for _ in range(int(steps)):",
        "        simulation_step(buf, dt)",
        "    final = [float(v) for v in buf.data]",
        "    telemetry = {\"steps\": float(buf.steps), \"finite\": 1.0,",
        "                  \"final_l2\": sum(v * v for v in final) ** 0.5}",
        "    return {\"state\": final, \"steps\": int(buf.steps), \"finite\": True,",
        "            \"telemetry\": telemetry}",
        "",
    ]
    return "\n".join(lines)


def _upload_report(title: str, custom_notes: str, evidence: Any,
                   benchmarks: List[Dict[str, Any]], saved: List[str],
                   run_id: str) -> str:
    lines = [
        f"# {title}",
        "",
        "> **Origin:** user-uploaded research (zero-token extraction).",
        "> **Status:** UPLOADED_READY — not yet computationally verified.",
        "",
    ]
    if custom_notes.strip():
        lines += ["## Researcher Notes", "", custom_notes.strip(), ""]
    lines += ["## Uploaded Sources", ""]
    lines.extend(f"- `{p}`" for p in saved) or ["- (none)"]
    lines.append("")
    table = getattr(evidence, "summary_markdown_table", "") or ""
    if table:
        lines += ["## Extracted Literature Equations & SI Constants", "", table, ""]
    if evidence.equations:
        lines += ["## Governing Equations", ""]
        for eq in evidence.equations[:16]:
            lines.append(f"- `{eq.lhs} = {eq.rhs_python_hint or eq.rhs_latex}`")
        lines.append("")
    if benchmarks:
        lines += ["## Benchmark Anchors", "", "| name | expected | unit |", "|---|---|---|"]
        for b in benchmarks[:12]:
            lines.append(f"| {b.get('name', '')} | {b.get('expected_value', '')} "
                         f"| {b.get('unit', '')} |")
        lines.append("")
    lines += [
        "## Next Steps",
        "",
        f"- Attach `@research:{run_id}` in the Research Agent to computationally "
        "verify and accrete this research (status becomes VERIFIED).",
        f"- Use `[Build Software from this Research]` (`@research:{run_id}`) to "
        "generate a software archetype with Research-Parity tests.",
        "",
    ]
    return "\n".join(lines)


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
