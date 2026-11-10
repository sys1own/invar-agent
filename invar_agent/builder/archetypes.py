"""Formalized software build archetypes for the invar-agent Builder Agent.

Seven target software shapes, each with toolchains, a default file manifest,
Plan-Mode scaffold instructions and test requirements. Classification is
deterministic: an explicit ``[TARGET_ARCHETYPE: <NAME>]`` / ``[ARCHETYPE: <NAME>]``
tag wins outright, otherwise weighted keyword scoring applies, and ambiguous
prompts fall back safely to ``SCIENTIFIC_SDK_LIBRARY``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class SoftwareArchetype(str, Enum):
    SCIENTIFIC_SDK_LIBRARY = "SCIENTIFIC_SDK_LIBRARY"
    CLI_SIMULATOR_SWEEP = "CLI_SIMULATOR_SWEEP"
    MULTI_SOLVER_BENCHMARK = "MULTI_SOLVER_BENCHMARK"
    INVERSE_DESIGN_OPTIMIZER = "INVERSE_DESIGN_OPTIMIZER"
    STANDALONE_NATIVE_CRATE = "STANDALONE_NATIVE_CRATE"
    INTERACTIVE_WEB_SIMULATOR = "INTERACTIVE_WEB_SIMULATOR"
    TELEMETRY_DIGITAL_TWIN = "TELEMETRY_DIGITAL_TWIN"


# Tie-break priority (most specific first); SCIENTIFIC_SDK_LIBRARY is last.
_ARCHEYPE_PRIORITY = (
    SoftwareArchetype.STANDALONE_NATIVE_CRATE,
    SoftwareArchetype.INVERSE_DESIGN_OPTIMIZER,
    SoftwareArchetype.MULTI_SOLVER_BENCHMARK,
    SoftwareArchetype.TELEMETRY_DIGITAL_TWIN,
    SoftwareArchetype.INTERACTIVE_WEB_SIMULATOR,
    SoftwareArchetype.CLI_SIMULATOR_SWEEP,
    SoftwareArchetype.SCIENTIFIC_SDK_LIBRARY,
)


@dataclass(frozen=True)
class ArchetypeBlueprint:
    archetype: SoftwareArchetype
    toolchains: List[str]
    default_manifest: List[str]
    prompt_scaffold_instructions: str
    test_requirements: List[str]


ARCHETYPE_REGISTRY: Dict[SoftwareArchetype, ArchetypeBlueprint] = {
    SoftwareArchetype.SCIENTIFIC_SDK_LIBRARY: ArchetypeBlueprint(
        archetype=SoftwareArchetype.SCIENTIFIC_SDK_LIBRARY,
        toolchains=["python", "pytest", "clang++ (optional C-ABI)", "cargo (optional PyO3)"],
        default_manifest=[
            "pyproject.toml", "src/{pkg}/__init__.py", "src/{pkg}/constants.py",
            "src/{pkg}/kernel.py", "src/{pkg}/api.py", "tests/test_api.py",
            "tests/test_research_parity.py", "README.md"],
        prompt_scaffold_instructions=(
            "Reusable Python package with a typed public API over the verified "
            "kernel. Keep SI constants in constants.py and re-export "
            "RESEARCH_CONSTANTS. Native C++ C-ABI or Rust PyO3 accelerators are "
            "optional and must be numerically identical to the pure-Python kernel."),
        test_requirements=[
            "tests/test_research_parity.py (constants preserved, finite outputs, "
            "invariants, key_metrics within tolerance)",
            "Unit tests for every public API function."],
    ),
    SoftwareArchetype.CLI_SIMULATOR_SWEEP: ArchetypeBlueprint(
        archetype=SoftwareArchetype.CLI_SIMULATOR_SWEEP,
        toolchains=["python", "pytest", "clang++ (optional)"],
        default_manifest=[
            "pyproject.toml", "src/{pkg}/__init__.py", "src/{pkg}/constants.py",
            "src/{pkg}/kernel.py", "src/{pkg}/cli.py", "src/{pkg}/sweep.py",
            "tests/test_cli.py", "tests/test_research_parity.py", "README.md"],
        prompt_scaffold_instructions=(
            "Standalone command-line application with argparse flags for every "
            "model parameter (defaults = research constants). Batch parameter "
            "sweeps with per-run invariant checks and JSON / CSV / Markdown "
            "output writers. Headless throughput matters: avoid per-step I/O."),
        test_requirements=[
            "tests/test_research_parity.py",
            "CLI default invocation reproduces the research key_metrics.",
            "Sweep runs flag invariant violations instead of crashing."],
    ),
    SoftwareArchetype.MULTI_SOLVER_BENCHMARK: ArchetypeBlueprint(
        archetype=SoftwareArchetype.MULTI_SOLVER_BENCHMARK,
        toolchains=["python", "pytest", "clang++ (optional)"],
        default_manifest=[
            "pyproject.toml", "src/{pkg}/kernel.py", "src/{pkg}/solvers/euler.py",
            "src/{pkg}/solvers/rk4.py", "src/{pkg}/solvers/symplectic.py",
            "src/{pkg}/benchmark.py", "src/{pkg}/pareto.py",
            "tests/test_benchmark.py", "tests/test_research_parity.py", "README.md"],
        prompt_scaffold_instructions=(
            "Benchmarking harness comparing solvers on accuracy (vs the verified "
            "reference), dt-convergence order and execution latency. Common "
            "solver interface; the verified research stepper is the reference. "
            "Emit a Pareto front (error vs latency) as JSON/Markdown."),
        test_requirements=[
            "tests/test_research_parity.py",
            "Reference solver reproduces the vault metrics exactly."],
    ),
    SoftwareArchetype.INVERSE_DESIGN_OPTIMIZER: ArchetypeBlueprint(
        archetype=SoftwareArchetype.INVERSE_DESIGN_OPTIMIZER,
        toolchains=["python", "pytest", "numpy/scipy (optional)"],
        default_manifest=[
            "pyproject.toml", "src/{pkg}/kernel.py", "src/{pkg}/constants.py",
            "src/{pkg}/objective.py", "src/{pkg}/optimizer.py",
            "src/{pkg}/constraints.py", "tests/test_optimizer.py",
            "tests/test_research_parity.py", "README.md"],
        prompt_scaffold_instructions=(
            "Constrained parameter-optimization harness (Nelder-Mead / Brent / "
            "gradient methods) wrapping the verified forward kernel as the "
            "objective evaluator. Encode research invariants as hard constraints "
            "on candidate designs. Deterministic optimizer (seeded) with "
            "convergence history output."),
        test_requirements=[
            "tests/test_research_parity.py",
            "Optimized designs satisfy every research invariant."],
    ),
    SoftwareArchetype.STANDALONE_NATIVE_CRATE: ArchetypeBlueprint(
        archetype=SoftwareArchetype.STANDALONE_NATIVE_CRATE,
        toolchains=["cargo", "rustc", "clang++", "cmake"],
        default_manifest=[
            "CMakeLists.txt + include/{pkg}/kernel.hpp + src/kernel.cpp + tests/test_kernel.cpp",
            "or Cargo.toml + src/lib.rs + tests/parity.rs",
            "reference/kernel.py (verified Python reference for parity)",
            "tests/test_research_parity.py", "README.md"],
        prompt_scaffold_instructions=(
            "Zero-Python standalone C++20 or Rust crate/binary. constexpr SI "
            "constants, C-ABI `extern \"C\"` entry point mirroring "
            "simulation_step(buf, dt), native test binary asserting the "
            "research key_metrics."),
        test_requirements=[
            "tests/test_research_parity.py",
            "Native test binary reproduces the vault metrics within tolerance."],
    ),
    SoftwareArchetype.INTERACTIVE_WEB_SIMULATOR: ArchetypeBlueprint(
        archetype=SoftwareArchetype.INTERACTIVE_WEB_SIMULATOR,
        toolchains=["python", "pytest", "fastapi", "uvicorn", "html5/canvas js"],
        default_manifest=[
            "pyproject.toml", "src/{pkg}/kernel.py", "src/{pkg}/constants.py",
            "src/{pkg}/server.py (FastAPI)", "src/{pkg}/static/index.html",
            "src/{pkg}/static/app.js", "tests/test_api.py",
            "tests/test_research_parity.py", "README.md"],
        prompt_scaffold_instructions=(
            "FastAPI backend + HTML5/Canvas interactive dashboard. REST "
            "endpoints: POST /simulate (parameters -> trajectory), GET "
            " /constants. Real-time parameter controls and trajectory / "
            "phase-space plots. The server must call the same verified kernel "
            "used by the parity test."),
        test_requirements=[
            "tests/test_research_parity.py",
            "TestClient: /simulate with research defaults returns the vault metrics."],
    ),
    SoftwareArchetype.TELEMETRY_DIGITAL_TWIN: ArchetypeBlueprint(
        archetype=SoftwareArchetype.TELEMETRY_DIGITAL_TWIN,
        toolchains=["python", "pytest"],
        default_manifest=[
            "pyproject.toml", "src/{pkg}/kernel.py", "src/{pkg}/constants.py",
            "src/{pkg}/stream.py", "src/{pkg}/estimator.py",
            "src/{pkg}/anomaly.py", "tests/test_twin.py",
            "tests/test_research_parity.py", "README.md"],
        prompt_scaffold_instructions=(
            "Streaming telemetry processor and real-time digital twin. Bounded-"
            "memory CSV/JSON ingestion, a state estimator (e.g. Kalman filter) "
            "propagating the verified research equations, and an anomaly "
            "detector built on the research invariant bounds."),
        test_requirements=[
            "tests/test_research_parity.py",
            "Noise-free synthetic telemetry from the kernel raises zero anomalies."],
    ),
}

# Weighted keyword sets (substring match on lowercased prompt).
_ARCHETYPE_KEYWORDS: Dict[SoftwareArchetype, List[str]] = {
    SoftwareArchetype.CLI_SIMULATOR_SWEEP: [
        "cli", "command-line", "command line", "sweep", "grid search", "batch",
        "csv export", "csv", "argparse", "click", "headless", "parameter sweep",
        "terminal tool"],
    SoftwareArchetype.INTERACTIVE_WEB_SIMULATOR: [
        "web", "browser", "html5", "canvas", "interactive", "dashboard",
        "fastapi", "gui", "visualization", "web app", "web simulator"],
    SoftwareArchetype.MULTI_SOLVER_BENCHMARK: [
        "benchmark", "multi-solver", "multi solver", "compare solvers",
        "pareto", "accuracy vs latency", "symplectic vs rk4", "solver comparison",
        "versus", "solvers", "convergence study"],
    SoftwareArchetype.INVERSE_DESIGN_OPTIMIZER: [
        "optimizer", "inverse design", "parameter estimation", "nelder-mead",
        "nelder mead", "objective function", "fitting", "minimize", "optimise",
        "optimize", "calibrat"],
    SoftwareArchetype.STANDALONE_NATIVE_CRATE: [
        "standalone crate", "pure rust", "cargo crate", "zero-python",
        "zero python", "native binary", "no python", "rust crate", "c++20",
        "cmake", "header-only", "standalone native"],
    SoftwareArchetype.TELEMETRY_DIGITAL_TWIN: [
        "digital twin", "digital-twin", "telemetry", "kalman", "sensor stream",
        "state observer", "live streaming", "anomaly detection", "streaming",
        "estimator"],
    SoftwareArchetype.SCIENTIFIC_SDK_LIBRARY: [
        "sdk", "library", "python package", "pyo3", "c-abi", "bindings",
        "importable", "python library", "module", "api", "package"],
}

_EXPLICIT_TAG_RE = re.compile(r"\[(?:TARGET_)?ARCHETYPE:\s*([A-Za-z_]+)\]", re.IGNORECASE)
_FALLBACK_CONFIDENCE = 0.35
_FALLBACK_RATIONALE = (
    "Prompt did not match specialized tooling requirements; defaulted to "
    "standard SCIENTIFIC_SDK_LIBRARY.")


@dataclass
class ArchetypeClassificationResult:
    archetype: SoftwareArchetype
    confidence: float
    is_fallback: bool
    rationale: str
    recommended_manifest: List[str] = field(default_factory=list)
    toolchains: List[str] = field(default_factory=list)

    @property
    def archetype_id(self) -> str:
        return self.archetype.value


def _keyword_score(archetype: SoftwareArchetype, text: str) -> float:
    score = 0.0
    for kw in _ARCHETYPE_KEYWORDS[archetype]:
        if kw in text:
            # Multi-word / hyphenated phrases are stronger signals than
            # single tokens.
            words = [w for w in re.split(r"[\s-]+", kw) if w]
            score += min(2.0, float(len(words)))
    return score


def classify_software_archetype(prompt: str,
                                has_research: bool = False
                                ) -> ArchetypeClassificationResult:
    """Deterministic archetype classification with safe fallback.

    1. Explicit ``[TARGET_ARCHETYPE: <NAME>]`` / ``[ARCHETYPE: <NAME>]`` tag
       (or a bare archetype id) wins with confidence 1.0.
    2. Otherwise weighted keyword scoring; the best archetype wins ties by
       specificity priority.
    3. If the best confidence is below 0.35, fall back to
       SCIENTIFIC_SDK_LIBRARY with ``is_fallback=True``.
    """
    text = (prompt or "").lower()

    def _result(archetype: SoftwareArchetype, confidence: float,
                is_fallback: bool, rationale: str) -> ArchetypeClassificationResult:
        bp = ARCHETYPE_REGISTRY[archetype]
        return ArchetypeClassificationResult(
            archetype=archetype, confidence=round(confidence, 4),
            is_fallback=is_fallback, rationale=rationale,
            recommended_manifest=list(bp.default_manifest),
            toolchains=list(bp.toolchains))

    tag = _EXPLICIT_TAG_RE.search(prompt or "")
    if tag:
        name = tag.group(1).upper()
        try:
            archetype = SoftwareArchetype(name)
        except ValueError:
            archetype = None
        if archetype is not None:
            return _result(archetype, 1.0, False,
                           f"Explicit [{tag.group(0).strip('[]')}] tag in prompt.")

    # Legacy: a bare archetype id anywhere in the prompt also pins the target.
    for archetype in _ARCHEYPE_PRIORITY:
        if archetype.value.lower() in text:
            return _result(archetype, 1.0, False,
                           f"Explicit archetype id {archetype.value} in prompt.")

    scored = [(a, _keyword_score(a, text)) for a in _ARCHEYPE_PRIORITY]
    best_archetype, best_score = max(
        scored, key=lambda item: (item[1], -_ARCHEYPE_PRIORITY.index(item[0])))
    confidence = min(1.0, best_score / 2.5)
    if confidence < _FALLBACK_CONFIDENCE or best_score <= 0:
        rationale = _FALLBACK_RATIONALE
        if has_research:
            rationale += " Research context is present, so the SDK pattern " \
                "is used as the foundation."
        return _result(SoftwareArchetype.SCIENTIFIC_SDK_LIBRARY,
                       _FALLBACK_CONFIDENCE, True, rationale)

    return _result(best_archetype, confidence, False,
                   f"Selected {best_archetype.value} from keyword signals "
                   f"(score {best_score:.1f}).")


def archetype_scaffold_text(archetype: SoftwareArchetype) -> str:
    """Plan-Mode scaffold block for one archetype (manifest + tests)."""
    bp = ARCHETYPE_REGISTRY[archetype]
    lines = [
        f"[TARGET_ARCHETYPE: {archetype.value}]",
        "",
        f"Target archetype: {archetype.value}",
        "Toolchains: " + ", ".join(bp.toolchains),
        "",
        "Architecture guidance:",
        bp.prompt_scaffold_instructions,
        "",
        "Default file manifest:",
    ]
    lines += [f"- {p}" for p in bp.default_manifest]
    lines += ["", "Test requirements:"]
    lines += [f"- {t}" for t in bp.test_requirements]
    return "\n".join(lines)


def has_explicit_archetype_tag(text: str) -> bool:
    """True when ``[TARGET_ARCHETYPE: <NAME>]`` / ``[ARCHETYPE: <NAME>]`` is present."""
    return _EXPLICIT_TAG_RE.search(text or "") is not None


def ensure_archetype_tag(prompt: str, archetype: SoftwareArchetype) -> str:
    """Prepend ``[TARGET_ARCHETYPE: <NAME>]`` unless already tagged."""
    tag = f"[TARGET_ARCHETYPE: {archetype.value}]"
    if _EXPLICIT_TAG_RE.search(prompt or ""):
        return prompt
    return f"{tag}\n\n{prompt}"
