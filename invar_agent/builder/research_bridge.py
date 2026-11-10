"""Research-to-Builder Bridge.

Connects completed (or user-uploaded) Research Vault runs to the Builder Agent
in both Direct Mode and Plan Mode:

* :func:`classify_software_archetype` maps a build prompt onto one of seven
  research-to-software archetypes.
* :func:`resolve_relevant_research_runs` binds vault runs to a build prompt
  (explicit ``@research:<id>`` -> keyword/symbol overlap -> implicit "from our
  research" fallback; unrelated greenfield prompts resolve to ``[]``).
* :func:`compile_research_build_packet` compiles a
  :class:`ResearchToBuildPacket` (SI constants, equations, verified kernel,
  contracts, ground-truth metrics) and its
  ``<research_to_software_specification>`` prompt block.
* :func:`generate_research_parity_test_code` emits a self-contained
  ``tests/test_research_parity.py`` that re-executes the built software (or the
  embedded verified kernel) and asserts numerical parity with the vault.

Everything here is deterministic and makes zero LLM calls.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..research.vault import ResearchVaultManager
from .archetypes import (ARCHETYPE_REGISTRY, ArchetypeClassificationResult,
                         SoftwareArchetype, archetype_scaffold_text,
                         classify_software_archetype as _classify_archetype)

RESEARCH_REF_RE = re.compile(r"@research:([a-zA-Z0-9_\-]+)")
PARITY_TEST_RELPATH = "tests/test_research_parity.py"
PROVENANCE_FILENAME = ".invar_research_provenance.json"

# ---------------------------------------------------------------------------
# Software archetypes
# ---------------------------------------------------------------------------


class SoftwareArchetypeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    archetype_id: str
    title: str
    recommended_file_layout: List[str] = Field(default_factory=list)
    architecture_guidelines: List[str] = Field(default_factory=list)
    verification_requirements: List[str] = Field(default_factory=list)


_COMMON_VERIFICATION = [
    "tests/test_research_parity.py must pass: constants preserved, outputs finite, "
    "every research invariant holds, key_metrics reproduced within tolerance.",
]

ARCHETYPES: Dict[str, SoftwareArchetypeSpec] = {
    "SCIENTIFIC_SDK_LIBRARY": SoftwareArchetypeSpec(
        archetype_id="SCIENTIFIC_SDK_LIBRARY",
        title="Scientific SDK / Library",
        recommended_file_layout=[
            "pyproject.toml", "src/{pkg}/__init__.py", "src/{pkg}/constants.py",
            "src/{pkg}/kernel.py", "src/{pkg}/api.py", "cpp_core/ (optional C-ABI)",
            "rust_core/ (optional PyO3)", "tests/test_api.py",
            "tests/test_research_parity.py", "README.md"],
        architecture_guidelines=[
            "Expose a typed public API (dataclasses / type hints) over the verified kernel.",
            "Keep SI constants in constants.py and re-export RESEARCH_CONSTANTS.",
            "Native bindings (C++ C-ABI or Rust PyO3) are optional accelerators that "
            "must be numerically identical to the pure-Python kernel.",
        ],
        verification_requirements=_COMMON_VERIFICATION + [
            "Unit tests for every public API function."],
    ),
    "CLI_SIMULATOR_SWEEP": SoftwareArchetypeSpec(
        archetype_id="CLI_SIMULATOR_SWEEP",
        title="CLI Simulator & Parameter Sweep Tool",
        recommended_file_layout=[
            "pyproject.toml", "src/{pkg}/__init__.py", "src/{pkg}/constants.py",
            "src/{pkg}/kernel.py", "src/{pkg}/cli.py", "src/{pkg}/sweep.py",
            "tests/test_cli.py", "tests/test_research_parity.py", "README.md"],
        architecture_guidelines=[
            "argparse CLI with flags for every model parameter (defaults = research constants).",
            "Batch sweep over parameter grids with invariant checks per run.",
            "JSON / CSV / Markdown output writers.",
        ],
        verification_requirements=_COMMON_VERIFICATION + [
            "CLI default invocation reproduces the research key_metrics.",
            "Sweep runs flag invariant violations instead of crashing."],
    ),
    "INTERACTIVE_WEB_SIMULATOR": SoftwareArchetypeSpec(
        archetype_id="INTERACTIVE_WEB_SIMULATOR",
        title="Interactive Web Simulator / Dashboard",
        recommended_file_layout=[
            "pyproject.toml", "src/{pkg}/kernel.py", "src/{pkg}/constants.py",
            "src/{pkg}/server.py (FastAPI)", "src/{pkg}/static/index.html",
            "src/{pkg}/static/app.js", "tests/test_api.py",
            "tests/test_research_parity.py", "README.md"],
        architecture_guidelines=[
            "FastAPI REST endpoints: POST /simulate (parameters -> trajectory), GET /constants.",
            "HTML5/JS dashboard with parameter controls and trajectory / phase-space plots.",
            "The server must call the same verified kernel used by the parity test.",
        ],
        verification_requirements=_COMMON_VERIFICATION + [
            "TestClient: /simulate with research defaults returns the vault metrics."],
    ),
    "INVERSE_DESIGN_OPTIMIZER": SoftwareArchetypeSpec(
        archetype_id="INVERSE_DESIGN_OPTIMIZER",
        title="Inverse-Design / Optimizer Engine",
        recommended_file_layout=[
            "pyproject.toml", "src/{pkg}/kernel.py", "src/{pkg}/constants.py",
            "src/{pkg}/objective.py", "src/{pkg}/optimizer.py",
            "src/{pkg}/constraints.py", "tests/test_optimizer.py",
            "tests/test_research_parity.py", "README.md"],
        architecture_guidelines=[
            "Wrap the verified forward kernel as the objective evaluator.",
            "Encode the research invariants as hard constraints on candidate designs.",
            "Deterministic optimizer (seeded) with convergence history output.",
        ],
        verification_requirements=_COMMON_VERIFICATION + [
            "Optimized designs satisfy every research invariant."],
    ),
    "MULTI_SOLVER_BENCHMARK": SoftwareArchetypeSpec(
        archetype_id="MULTI_SOLVER_BENCHMARK",
        title="Multi-Solver Benchmark Suite",
        recommended_file_layout=[
            "pyproject.toml", "src/{pkg}/kernel.py", "src/{pkg}/solvers/euler.py",
            "src/{pkg}/solvers/rk4.py", "src/{pkg}/solvers/symplectic.py",
            "src/{pkg}/benchmark.py", "src/{pkg}/pareto.py",
            "tests/test_benchmark.py", "tests/test_research_parity.py", "README.md"],
        architecture_guidelines=[
            "Common solver interface; the verified research stepper is the reference.",
            "Measure accuracy (vs reference), dt-convergence order and latency.",
            "Emit a Pareto front (error vs latency) as JSON/Markdown.",
        ],
        verification_requirements=_COMMON_VERIFICATION + [
            "Reference solver reproduces the vault metrics exactly."],
    ),
    "TELEMETRY_DIGITAL_TWIN": SoftwareArchetypeSpec(
        archetype_id="TELEMETRY_DIGITAL_TWIN",
        title="Streaming Telemetry / Digital Twin Filter",
        recommended_file_layout=[
            "pyproject.toml", "src/{pkg}/kernel.py", "src/{pkg}/constants.py",
            "src/{pkg}/stream.py", "src/{pkg}/estimator.py",
            "src/{pkg}/anomaly.py", "tests/test_twin.py",
            "tests/test_research_parity.py", "README.md"],
        architecture_guidelines=[
            "Streaming CSV/JSON ingestion with bounded memory.",
            "State estimator propagating the verified research equations.",
            "Real-time anomaly detector built on the research invariant bounds.",
        ],
        verification_requirements=_COMMON_VERIFICATION + [
            "Noise-free synthetic telemetry from the kernel raises zero anomalies."],
    ),
    "STANDALONE_NATIVE_CRATE": SoftwareArchetypeSpec(
        archetype_id="STANDALONE_NATIVE_CRATE",
        title="Standalone Native C++20 / Rust Crate",
        recommended_file_layout=[
            "CMakeLists.txt + include/{pkg}/kernel.hpp + src/kernel.cpp + tests/test_kernel.cpp",
            "or Cargo.toml + src/lib.rs + tests/parity.rs",
            "reference/kernel.py (verified Python reference for parity)",
            "tests/test_research_parity.py", "README.md"],
        architecture_guidelines=[
            "Zero-Python runtime: constexpr SI constants / Rust const items.",
            "C-ABI `extern \"C\"` entry point mirroring simulation_step(buf, dt).",
            "Native test binary asserting the research key_metrics.",
        ],
        verification_requirements=_COMMON_VERIFICATION + [
            "Native test binary reproduces the vault metrics within tolerance."],
    ),
}

# Ordered by specificity: ties resolve to the earlier (more specific) archetype.
_ARCHETYPE_PATTERNS: List[tuple] = [
    ("STANDALONE_NATIVE_CRATE", [
        r"\bnative\s+(?:c\+\+|rust)", r"\bc\+\+\s*20\b", r"\bc\+\+", r"\brust\s+crate\b",
        r"\bcrate\b", r"\bcmake", r"\bcargo\b", r"\bzero[- ]python\b",
        r"\bstandalone\s+native\b", r"\bheader[- ]only\b"]),
    ("INVERSE_DESIGN_OPTIMIZER", [
        r"\boptimi[sz]", r"\binverse[- ]design\b", r"\bcalibrat", r"\bminimi[sz]",
        r"\bmaximi[sz]", r"\bparameter\s+fit", r"\bdesign\s+search\b"]),
    ("MULTI_SOLVER_BENCHMARK", [
        r"\bbenchmark", r"\bcompare\s+solvers?\b", r"\bsolver\s+comparison\b",
        r"\bpareto\b", r"\bconvergence\s+stud", r"\bmulti[- ]solver\b"]),
    ("TELEMETRY_DIGITAL_TWIN", [
        r"\bdigital[- ]twin\b", r"\btelemetry\b", r"\bstreaming\b", r"\bkalman\b",
        r"\banomal", r"\bstate\s+estimat", r"\bfilter\b"]),
    ("INTERACTIVE_WEB_SIMULATOR", [
        r"\bweb\b", r"\bdashboard\b", r"\binteractive\b", r"\bfastapi\b",
        r"\bbrowser\b", r"\bhtml", r"\bweb\s*app\b"]),
    ("CLI_SIMULATOR_SWEEP", [
        r"\bcli\b", r"\bcommand[- ]line\b", r"\bargparse\b", r"\bparameter\s+sweep\b",
        r"\bsweep", r"\bterminal\s+tool\b"]),
    ("SCIENTIFIC_SDK_LIBRARY", [
        r"\blibrary\b", r"\bpackage\b", r"\bsdk\b", r"\bmodule\b", r"\bapi\b"]),
]


def classify_software_archetype(prompt: str,
                                research_runs: Optional[List[Dict[str, Any]]] = None,
                                has_research: bool = False
                                ) -> ArchetypeClassificationResult:
    """Deterministic archetype classification with safe fallback.

    Delegates to :mod:`invar_agent.builder.archetypes`: explicit
    ``[TARGET_ARCHETYPE: <NAME>]`` tags win, weighted keyword scoring handles
    the rest, and ambiguous prompts fall back to SCIENTIFIC_SDK_LIBRARY.
    """
    del research_runs  # retained for backward compatibility
    return _classify_archetype(prompt, has_research=has_research)


# ---------------------------------------------------------------------------
# 3-tier research resolution
# ---------------------------------------------------------------------------

_IMPLICIT_RESEARCH_PHRASES = (
    "from research", "from the research", "from our research", "from my research",
    "completed research", "the research", "this research", "our research",
    "the simulation", "verified model", "our study", "uploaded research",
    "turn this into", "build a simulator for it", "research run",
    "verified research", "latest research",
)

_STOPWORDS = {
    "build", "create", "make", "write", "generate", "implement", "develop", "design",
    "software", "program", "application", "app", "tool", "tools", "project", "code",
    "python", "production", "simple", "small", "basic", "full", "complete", "new",
    "with", "from", "for", "into", "that", "this", "the", "and", "our", "your", "using",
    "based", "model", "models", "simulator", "simulation", "library", "package",
    "module", "sdk", "cli", "web", "interactive", "dashboard", "engine", "suite",
    "research", "completed", "verified", "study", "run", "runs", "data", "system",
    "native", "crate", "rust", "c++", "api", "test", "tests", "version", "file", "files",
}


def _tokens(text: str) -> set:
    raw = str(text or "").lower()
    out = set()
    for tok in re.split(r"[^a-z0-9\-]+", raw):
        tok = tok.strip("-")
        if not tok:
            continue
        parts = [tok] + [p for p in tok.split("-") if p]
        for part in parts:
            if len(part) >= 3 and part not in _STOPWORDS and not part.isdigit():
                out.add(part)
    return out


def _run_tokens(run: Dict[str, Any]) -> set:
    fields: List[str] = [str(run.get("run_id", "")).replace("_", " "),
                         str(run.get("title", ""))]
    for key in (run.get("extracted_constants") or {}):
        fields.append(str(key).replace("_", " "))
    fields.extend(str(e) for e in (run.get("governing_equations") or []))
    for key in (run.get("key_metrics") or {}):
        fields.append(str(key).replace("_", " "))
    evidence = run.get("structured_evidence") or {}
    fields.extend(str(d) for d in (evidence.get("domains") or []))
    for key in (evidence.get("parameters") or {}):
        fields.append(str(key).replace("_", " "))
    fields.extend(str(k) for k in (run.get("domain_keywords") or []))
    return _tokens(" ".join(fields))


def _has_research_intent(prompt: str) -> bool:
    low = (prompt or "").lower()
    return any(phrase in low for phrase in _IMPLICIT_RESEARCH_PHRASES)


def resolve_relevant_research_runs(workspace_dir: str | Path, prompt: str,
                                   attached_research_ids: Optional[List[str]] = None,
                                   *, min_overlap: int = 2) -> List[Dict[str, Any]]:
    """Resolve vault runs relevant to a build prompt (each tagged with
    ``resolution_tier``: ``explicit`` / ``keyword`` / ``implicit``)."""
    vault = ResearchVaultManager(workspace_dir)
    runs = vault.list_runs()
    if not runs:
        return []
    by_id = {str(r.get("run_id")): r for r in runs}

    explicit: List[str] = []
    for rid in list(attached_research_ids or []) + RESEARCH_REF_RE.findall(prompt or ""):
        if rid and rid not in explicit:
            explicit.append(rid)
    if explicit:
        resolved = [dict(by_id[rid], resolution_tier="explicit")
                    for rid in explicit if rid in by_id]
        if resolved:
            return resolved

    prompt_tokens = _tokens(RESEARCH_REF_RE.sub(" ", prompt or ""))
    scored: List[tuple] = []
    for idx, run in enumerate(runs):
        overlap = prompt_tokens & _run_tokens(run)
        if len(overlap) >= min_overlap:
            scored.append((-len(overlap), idx, run, sorted(overlap)))
    if scored:
        scored.sort(key=lambda t: (t[0], t[1]))
        return [dict(run, resolution_tier="keyword", matched_terms=terms)
                for _, _, run, terms in scored]

    if _has_research_intent(prompt):
        # list_runs() is newest-first.
        return [dict(runs[0], resolution_tier="implicit")]
    return []


# ---------------------------------------------------------------------------
# ResearchToBuildPacket
# ---------------------------------------------------------------------------


class ResearchConstant(BaseModel):
    name: str
    python_name: str
    value: float
    unit: str = ""
    source: str = ""


class ResearchContractSpec(BaseModel):
    contract_id: str
    run_id: str = ""
    dt: float = 0.01
    steps: int = 1000
    initial_state: List[float] = Field(default_factory=list)
    invariants: List[str] = Field(default_factory=list)


class ResearchMetric(BaseModel):
    name: str
    value: float
    unit: str = ""
    slot: Optional[int] = None
    run_id: str = ""


class ResearchToBuildPacket(BaseModel):
    run_ids: List[str]
    titles: Dict[str, str] = Field(default_factory=dict)
    origins: Dict[str, str] = Field(default_factory=dict)
    archetype: SoftwareArchetypeSpec
    classification_result: Optional[Dict[str, Any]] = None
    package_name: str = "research_software"
    constants: List[ResearchConstant] = Field(default_factory=list)
    governing_equations: List[str] = Field(default_factory=list)
    equation_hints: List[Dict[str, str]] = Field(default_factory=list)
    reference_kernels: Dict[str, str] = Field(default_factory=dict)
    entry_function: str = "simulation_step"
    variable_map: Dict[str, int] = Field(default_factory=dict)
    contracts: List[ResearchContractSpec] = Field(default_factory=list)
    key_metrics: List[ResearchMetric] = Field(default_factory=list)
    metric_contract_id: str = ""
    metric_rel_tolerance: float = 1e-6
    metric_abs_tolerance: float = 1e-9
    prompt_block: str = ""

    @property
    def primary_run_id(self) -> str:
        return self.run_ids[0] if self.run_ids else ""

    @property
    def primary_kernel(self) -> str:
        return self.reference_kernels.get(self.primary_run_id, "")


_NAME_FILLER = {"the", "and", "for", "with", "via", "under", "over", "from", "into",
                "using", "two", "stage", "multi", "deep"}


def _python_const_name(name: str) -> str:
    ident = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").upper()
    if not ident:
        ident = "CONST"
    if ident[0].isdigit():
        ident = "C_" + ident
    return ident


def _package_name(title: str, run_id: str) -> str:
    words = [w for w in re.split(r"[^a-z0-9]+", (title or run_id or "").lower())
             if len(w) >= 3 and w not in _STOPWORDS and w not in _NAME_FILLER
             and not w.isdigit()][:3]
    name = "_".join(words) or "research_software"
    if name[0].isdigit():
        name = "r_" + name
    return name[:40]


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    return None


def _contracts_from_detail(run_id: str, detail: Dict[str, Any]) -> List[ResearchContractSpec]:
    """Contracts verified against the FINAL unified kernel: the last pinned
    layer's contracts when available, else the blueprint's contracts, else
    the metadata contracts_summary (no invariants)."""
    blueprint = detail.get("blueprint") or {}
    contracts_raw: List[Dict[str, Any]] = []
    layers = blueprint.get("verified_layers") or []
    if layers and (layers[-1].get("contracts") or []):
        contracts_raw = list(layers[-1]["contracts"])
    elif blueprint.get("contracts"):
        contracts_raw = list(blueprint["contracts"])
    specs: List[ResearchContractSpec] = []
    for c in contracts_raw:
        initial = c.get("initial_state") or []
        specs.append(ResearchContractSpec(
            contract_id=str(c.get("contract_id", "contract")), run_id=run_id,
            dt=float(c.get("dt", 0.01) or 0.01), steps=int(c.get("steps", 1000) or 1000),
            initial_state=[float(v) for v in initial],
            invariants=[str(i.get("assertion", "")) for i in (c.get("invariants") or [])
                        if i.get("assertion")]))
    if specs:
        return specs
    meta = detail.get("metadata") or {}
    for c in meta.get("contracts_summary") or []:
        specs.append(ResearchContractSpec(
            contract_id=str(c.get("contract_id", "contract")), run_id=run_id,
            dt=float(c.get("dt", 0.01) or 0.01), steps=int(c.get("steps", 1000) or 1000),
            initial_state=[float(v) for v in (c.get("initial_state") or [])]))
    return specs


def _load_detail(vault: Optional[ResearchVaultManager], run: Dict[str, Any]) -> Dict[str, Any]:
    if vault is not None:
        detail = vault.get_run(str(run.get("run_id", "")))
        if detail is not None:
            return detail
    return {"metadata": run, "unified_module_code": run.get("unified_module_code", ""),
            "blueprint": run.get("blueprint"), "report_md": run.get("report_md", "")}


def compile_research_build_packet(prompt: str, research_runs: List[Dict[str, Any]],
                                  workspace_dir: Optional[str | Path] = None,
                                  archetype: Optional[SoftwareArchetypeSpec] = None
                                  ) -> ResearchToBuildPacket:
    """Compile the structured research-to-software specification."""
    vault = ResearchVaultManager(workspace_dir) if workspace_dir is not None else None
    classification = archetype or classify_software_archetype(
        prompt, research_runs, has_research=bool(research_runs))
    if isinstance(classification, ArchetypeClassificationResult):
        archetype_spec = ARCHETYPES[classification.archetype_id]
        classification_result: Optional[Dict[str, Any]] = {
            "archetype": classification.archetype_id,
            "confidence": classification.confidence,
            "is_fallback": classification.is_fallback,
            "rationale": classification.rationale,
            "recommended_manifest": list(classification.recommended_manifest),
            "toolchains": list(classification.toolchains),
        }
    else:
        # Backward compatibility: a raw SoftwareArchetypeSpec was passed.
        archetype_spec = classification
        classification_result = None
    archetype = archetype_spec
    run_ids: List[str] = []
    titles: Dict[str, str] = {}
    origins: Dict[str, str] = {}
    constants: Dict[str, ResearchConstant] = {}
    equations: List[str] = []
    hints: List[Dict[str, str]] = []
    kernels: Dict[str, str] = {}
    variable_map: Dict[str, int] = {}
    contracts: List[ResearchContractSpec] = []
    metrics: List[ResearchMetric] = []
    metric_contract_id = ""

    for run in research_runs:
        detail = _load_detail(vault, run)
        meta = dict(detail.get("metadata") or {})
        meta.update({k: v for k, v in run.items() if k not in meta})
        run_id = str(meta.get("run_id") or run.get("run_id") or "")
        if not run_id or run_id in run_ids:
            continue
        run_ids.append(run_id)
        titles[run_id] = str(meta.get("title") or run_id)
        origins[run_id] = str(meta.get("origin") or "research_agent")

        evidence = meta.get("structured_evidence") or {}
        ev_params = evidence.get("parameters") or {}
        for name, value in (meta.get("extracted_constants") or {}).items():
            f = _as_float(value)
            if f is None or name in constants:
                continue
            p = ev_params.get(name) or {}
            constants[name] = ResearchConstant(
                name=str(name), python_name=_python_const_name(name), value=f,
                unit=str(p.get("si_unit") or ""), source=str(p.get("source") or run_id))
        for name, p in ev_params.items():
            f = _as_float((p or {}).get("si_value"))
            if f is None or name in constants:
                continue
            constants[name] = ResearchConstant(
                name=str(name), python_name=_python_const_name(name), value=f,
                unit=str(p.get("si_unit") or ""), source=str(p.get("source") or run_id))

        for eq in meta.get("governing_equations") or []:
            if eq and str(eq) not in equations:
                equations.append(str(eq))
        for eq in evidence.get("equations") or []:
            hint = {"lhs": str(eq.get("lhs") or ""),
                    "rhs_latex": str(eq.get("rhs_latex") or ""),
                    "rhs_python_hint": str(eq.get("rhs_python_hint") or ""),
                    "source": str(eq.get("source") or "")}
            if hint not in hints:
                hints.append(hint)

        code = detail.get("unified_module_code") or ""
        if code.strip():
            kernels[run_id] = code

        blueprint = detail.get("blueprint") or {}
        for name, idx in (blueprint.get("variable_map") or {}).items():
            if isinstance(idx, int) and name not in variable_map:
                variable_map[str(name)] = idx
        if not variable_map:
            for c in meta.get("contracts_summary") or []:
                for slot, entry in (c.get("variable_map") or {}).items():
                    name = entry.get("name") if isinstance(entry, dict) else entry
                    if name and str(slot).isdigit():
                        variable_map.setdefault(str(name), int(slot))

        run_contracts = _contracts_from_detail(run_id, detail)
        contracts.extend(run_contracts)
        if run_contracts and not metric_contract_id:
            metric_contract_id = run_contracts[-1].contract_id
        for name, entry in (meta.get("key_metrics") or {}).items():
            if not isinstance(entry, dict):
                continue
            # Verified runs: only verified metrics are ground truth. User uploads
            # carry benchmark anchors marked verified=False — still surfaced so
            # the built software can be checked against them.
            if not entry.get("verified", True) and origins[run_id] != "user_upload":
                continue
            f = _as_float(entry.get("value"))
            if f is None:
                continue
            metrics.append(ResearchMetric(
                name=str(name), value=f, unit=str(entry.get("unit") or ""),
                slot=variable_map.get(str(name)), run_id=run_id))

    primary = run_ids[0] if run_ids else ""
    packet = ResearchToBuildPacket(
        run_ids=run_ids, titles=titles, origins=origins, archetype=archetype,
        package_name=_package_name(titles.get(primary, ""), primary),
        constants=list(constants.values()), governing_equations=equations,
        equation_hints=hints, reference_kernels=kernels,
        classification_result=classification_result,
        variable_map=variable_map, contracts=contracts, key_metrics=metrics,
        metric_contract_id=metric_contract_id)
    return packet.model_copy(update={"prompt_block": render_packet_prompt_block(packet)})


def render_packet_prompt_block(packet: ResearchToBuildPacket,
                               max_kernel_chars: int = 6000) -> str:
    """Markdown/XML ``<research_to_software_specification>`` prompt block."""
    arch = packet.archetype
    pkg = packet.package_name
    lines: List[str] = [
        "<research_to_software_specification>",
        f"SOURCE_RESEARCH: {', '.join('@research:' + r for r in packet.run_ids)}",
        f"ARCHETYPE: {arch.archetype_id} ({arch.title})",
        f"PACKAGE_NAME: {pkg}",
        "",
        "## Target File Layout",
    ]
    lines.extend(f"- {item.replace('{pkg}', pkg)}" for item in arch.recommended_file_layout)
    lines.append("")
    lines.append("## Architecture Guidelines")
    lines.extend(f"- {g}" for g in arch.architecture_guidelines)
    lines.append("")
    lines.append("## Archetype Scaffold")
    lines.append(archetype_scaffold_text(SoftwareArchetype(arch.archetype_id)))
    lines.append("")
    lines.append("## Exact SI Constants & Parameters (use VERBATIM — never guess or alter)")
    if packet.constants:
        lines.append("| name | value (SI) | unit | Python | C++ | Rust |")
        lines.append("|---|---|---|---|---|---|")
        for c in packet.constants:
            v = repr(c.value)
            lines.append(
                f"| {c.name} | {v} | {c.unit or '-'} | `{c.python_name} = {v}` | "
                f"`constexpr double {c.python_name} = {v};` | "
                f"`pub const {c.python_name}: f64 = {v};` |")
        lines.append("Expose all constants as module attributes AND as a "
                     "`RESEARCH_CONSTANTS` dict keyed by the Python names above.")
    else:
        lines.append("- (no extracted constants; reuse the literals inside the verified kernel)")
    lines.append("")
    lines.append("## Verified Governing Equations & Symbolic Hints")
    for eq in packet.governing_equations[:16]:
        lines.append(f"- {eq}")
    for h in packet.equation_hints[:16]:
        lines.append(f"- `{h['lhs']} = {h['rhs_python_hint'] or h['rhs_latex']}`"
                     + (f" (source: {h['source']})" if h.get("source") else ""))
    if not packet.governing_equations and not packet.equation_hints:
        lines.append("- (see verified reference kernel)")
    lines.append("")
    lines.append("## Verified Reference Kernel & variable_map")
    lines.append(
        f"Port, wrap or modularize `{packet.entry_function}(buf, dt)` (and its stage "
        "helpers) into the target architecture while preserving mathematical "
        "equivalence: same update order, same constants, same state-slot layout. "
        f"The core module must expose `{packet.entry_function}(buf, dt)` where "
        "`buf.data` is the mutable state vector.")
    if packet.variable_map:
        lines.append(f"variable_map (observable -> state slot): {json.dumps(packet.variable_map)}")
    for run_id, code in packet.reference_kernels.items():
        lines.append(f"### unified_module.py (@research:{run_id})")
        lines.append("```python")
        lines.append(code[:max_kernel_chars].rstrip())
        lines.append("```")
    lines.append("")
    lines.append("## Ground-Truth Regression Targets (Research-Parity Test Contract)")
    for c in packet.contracts:
        lines.append(f"- contract `{c.contract_id}`: dt={c.dt} steps={c.steps} "
                     f"initial_state={c.initial_state}")
        for inv in c.invariants:
            lines.append(f"  - invariant: `{inv}`")
    for m in packet.key_metrics:
        lines.append(f"- key_metric `{m.name}` = {m.value!r} {m.unit}".rstrip()
                     + (f" (state slot {m.slot})" if m.slot is not None else "")
                     + f" — rel_tol {packet.metric_rel_tolerance}")
    lines.append(f"The build MUST include `{PARITY_TEST_RELPATH}` (auto-generated) and it "
                 "must pass in Sandbox_Execute.")
    lines.append("</research_to_software_specification>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Research-parity regression test generator
# ---------------------------------------------------------------------------

_PARITY_TEMPLATE = '''"""Research-Parity Regression Test (auto-generated by invar-agent).

Source research: {sources}
Archetype: {archetype}

Verifies the built software reproduces the Research Vault's verified SI
constants, invariants and key metrics. Set INVAR_PARITY_MODULE to point at the
built kernel module explicitly; otherwise the candidates below are tried and
the embedded verified reference kernel is used as the last resort.
"""

import importlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

PACKET = json.loads({packet_json!r})
REFERENCE_KERNEL = {kernel!r}
CANDIDATE_MODULES = {candidates!r}
ENTRY = PACKET["entry_function"]

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class _Buffer:
    def __init__(self, data, dt):
        self.data = [float(v) for v in data]
        self.size = len(self.data)
        self.dt = dt
        self.steps = 0

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value


def _load_built_module():
    names = []
    env_name = os.environ.get("INVAR_PARITY_MODULE", "").strip()
    if env_name:
        names.append(env_name)
    names.extend(CANDIDATE_MODULES)
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - candidate not present in this build
            if name == env_name:
                raise
            continue
        if callable(getattr(module, ENTRY, None)):
            return module
    return None


def _entry():
    module = _load_built_module()
    if module is not None:
        return module, getattr(module, ENTRY)
    namespace = {{"__name__": "invar_reference_kernel"}}
    exec(compile(REFERENCE_KERNEL, "<reference_kernel>", "exec"), namespace)  # noqa: S102
    return None, namespace[ENTRY]


def _simulate(step, contract):
    buf = _Buffer(contract["initial_state"], contract["dt"])
    trace_min = list(buf.data)
    trace_max = list(buf.data)
    for _ in range(int(contract["steps"])):
        step(buf, contract["dt"])
        for i, v in enumerate(buf.data):
            trace_min[i] = min(trace_min[i], v)
            trace_max[i] = max(trace_max[i], v)
    return buf.data, trace_min, trace_max


def _telemetry(state, initial):
    l2 = math.sqrt(sum(x * x for x in state)) if state else 0.0
    tel = {{"steps": 0.0, "finite": 1.0 if all(math.isfinite(v) for v in state) else 0.0,
           "initial_l2": math.sqrt(sum(x * x for x in initial)) if initial else 0.0}}
    if state:
        change = max(abs(a - b) for a, b in zip(state, initial)) if initial else 0.0
        tel.update(final_min=min(state), final_max=max(state),
                   final_mean=sum(state) / len(state), final_l2=l2, final_norm=l2,
                   max_abs_change=change, energy_drift=change)
    for name, idx in PACKET["variable_map"].items():
        if 0 <= idx < len(state):
            tel["obs_" + name] = float(state[idx])
    return tel


class _BufView:
    def __init__(self, state):
        self.data = state
        self.size = len(state)


_SAFE = {{"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "all": all,
         "any": any, "round": round, "float": float, "int": int, "range": range,
         "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "sin": math.sin,
         "cos": math.cos, "tan": math.tan, "isfinite": math.isfinite,
         "isnan": math.isnan, "fabs": math.fabs, "hypot": math.hypot,
         "pi": math.pi}}
_UNREPRODUCIBLE = ("variants", "variant_", "hin_", "pareto", "execution_time",
                   "cabi_fallback", "run_ms", "compile_ms")


def _eval_invariant(expr, state, trace_min, trace_max, telemetry, initial):
    env = dict(_SAFE)
    env.update({{"state": state, "data": state, "buf": _BufView(state),
                "initial": initial, "initial_state": initial,
                "trace_min": trace_min, "trace_max": trace_max,
                "telemetry": telemetry, "math": math, "tol": 0.0}})
    for name, idx in PACKET["variable_map"].items():
        if name not in env and 0 <= idx < len(state):
            env[name] = float(state[idx])
    return bool(eval(expr, {{"__builtins__": {{}}}}, env))  # noqa: S307


def test_research_constants_preserved():
    assert PACKET["run_ids"], "parity test must reference at least one research run"
    module = _load_built_module()
    if module is None:
        pytest.skip("no built module found; constants checked against embedded kernel only")
    table = getattr(module, "RESEARCH_CONSTANTS", None)
    missing, mismatched = [], []
    for c in PACKET["constants"]:
        value = None
        if isinstance(table, dict):
            value = table.get(c["python_name"], table.get(c["name"]))
        if value is None:
            value = getattr(module, c["python_name"], None)
        if value is None:
            missing.append(c["python_name"])
        elif not math.isclose(float(value), c["value"], rel_tol=1e-12, abs_tol=1e-300):
            mismatched.append((c["python_name"], value, c["value"]))
    assert not missing, f"research constants missing from built package: {{missing}}"
    assert not mismatched, f"research constants altered: {{mismatched}}"


@pytest.mark.parametrize("contract", PACKET["contracts"],
                         ids=[c["contract_id"] for c in PACKET["contracts"]])
def test_research_kernel_parity(contract):
    _, step = _entry()
    initial = [float(v) for v in contract["initial_state"]]
    state, trace_min, trace_max = _simulate(step, contract)
    assert all(math.isfinite(v) for v in state), f"non-finite state: {{state}}"
    telemetry = _telemetry(state, initial)
    for expr in contract["invariants"]:
        if any(tok in expr for tok in _UNREPRODUCIBLE):
            continue
        try:
            ok = _eval_invariant(expr, state, trace_min, trace_max, telemetry, initial)
        except KeyError:
            continue  # telemetry channel only available inside the research sandbox
        assert ok, f"research invariant violated: {{expr}} (state={{state}})"
    if contract["contract_id"] != PACKET["metric_contract_id"]:
        return
    for metric in PACKET["key_metrics"]:
        if metric["slot"] is None or metric["slot"] >= len(state):
            continue
        got = float(state[metric["slot"]])
        assert math.isclose(got, metric["value"], rel_tol=PACKET["metric_rel_tolerance"],
                            abs_tol=PACKET["metric_abs_tolerance"]), (
            f"key_metric {{metric['name']}}: built={{got!r}} research={{metric['value']!r}}")
'''


def _candidate_modules(packet: ResearchToBuildPacket) -> List[str]:
    pkg = packet.package_name
    names = [f"{pkg}.kernel", f"{pkg}.core", pkg, f"{pkg}.simulator",
             "simulator", "research_kernel", "kernel", "reference.kernel"]
    out: List[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def generate_research_parity_test_code(packet: ResearchToBuildPacket) -> str:
    """Self-contained ``tests/test_research_parity.py`` source."""
    payload = {
        "run_ids": packet.run_ids,
        "entry_function": packet.entry_function,
        "variable_map": packet.variable_map,
        "constants": [c.model_dump() for c in packet.constants],
        "contracts": [c.model_dump() for c in packet.contracts],
        "key_metrics": [m.model_dump() for m in packet.key_metrics],
        "metric_contract_id": packet.metric_contract_id,
        "metric_rel_tolerance": packet.metric_rel_tolerance,
        "metric_abs_tolerance": packet.metric_abs_tolerance,
    }
    return _PARITY_TEMPLATE.format(
        sources=", ".join("@research:" + r for r in packet.run_ids) or "(none)",
        archetype=packet.archetype.archetype_id,
        packet_json=json.dumps(payload, sort_keys=True),
        kernel=packet.primary_kernel,
        candidates=_candidate_modules(packet),
    )


# ---------------------------------------------------------------------------
# Direct-mode binding helpers
# ---------------------------------------------------------------------------


def write_research_build_artifacts(workspace_dir: str | Path,
                                   packet: ResearchToBuildPacket) -> Dict[str, str]:
    """Write the parity test and the provenance manifest into the workspace."""
    root = Path(workspace_dir)
    test_path = root / PARITY_TEST_RELPATH
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(generate_research_parity_test_code(packet), encoding="utf-8")
    provenance = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "research_runs": [{"run_id": r, "title": packet.titles.get(r, r),
                           "origin": packet.origins.get(r, "")} for r in packet.run_ids],
        "archetype": packet.archetype.archetype_id,
        "package_name": packet.package_name,
        "parity_test": PARITY_TEST_RELPATH,
        "constants": {c.python_name: c.value for c in packet.constants},
        "key_metrics": {m.name: m.value for m in packet.key_metrics},
    }
    prov_path = root / PROVENANCE_FILENAME
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {"parity_test": str(test_path), "provenance": str(prov_path)}


def run_research_parity_tests(workspace_dir: str | Path, *, timeout: float = 120.0,
                              module: Optional[str] = None,
                              extra_args: Sequence[str] = ()) -> Dict[str, Any]:
    """Run the parity test in an isolated pytest subprocess (Sandbox_Execute)."""
    root = Path(workspace_dir)
    test_path = root / PARITY_TEST_RELPATH
    if not test_path.is_file():
        return {"ran": False, "passed": False, "returncode": None,
                "output": f"{PARITY_TEST_RELPATH} not found"}
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)
    if module:
        env["INVAR_PARITY_MODULE"] = module
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--rootdir", str(root), str(test_path), *extra_args]
    try:
        proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"ran": True, "passed": False, "returncode": None,
                "output": f"parity test timed out after {timeout}s: {exc}"}
    output = (proc.stdout or "") + (proc.stderr or "")
    return {"ran": True, "passed": proc.returncode == 0, "returncode": proc.returncode,
            "output": output[-4000:]}


def bind_research_context(workspace_dir: str | Path, prompt: str,
                          attached_research_ids: Optional[Iterable[str]] = None
                          ) -> Optional[ResearchToBuildPacket]:
    """Resolve + compile in one call; ``None`` for greenfield prompts."""
    runs = resolve_relevant_research_runs(workspace_dir, prompt,
                                          list(attached_research_ids or []))
    if not runs:
        return None
    return compile_research_build_packet(prompt, runs, workspace_dir=workspace_dir)
