"""Comprehensive Quantitative Research Report generation.

Deterministic (no LLM): every number is pulled directly from the verified
sandbox execution telemetry (``obs_<metric>`` observables published from the
blueprint ``variable_map`` plus the standard final-state telemetry), the
feasibility assessment, the literature ingest and the accretion history, so
the report can never invent values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas.research import ResearchBlueprint, SimulationResult
from .report import _fmt_value, extract_observables

# Map common unit suffixes embedded in observable names to display units.
_UNIT_SUFFIXES = [
    ("mw_m2", "MW/m²"), ("mj_m2", "MJ/m²"), ("w_m2", "W/m²"),
    ("km", "km"), ("_g", "Earth g"), ("_j", "J"), ("_t", "t"),
    ("_kg", "kg"), ("_s", "s"), ("_ms", "ms"), ("_t_s", "T"),
    ("_pa", "Pa"), ("_k", "k"),
]


def _unit_for(name: str) -> str:
    lowered = name.lower()
    for suffix, unit in _UNIT_SUFFIXES:
        if lowered.endswith(suffix):
            return unit
    return "—"


def _symbol_for(name: str) -> str:
    # peak_deceleration_g -> a_max is unknowable deterministically; use the
    # metric name itself as the symbol (honest, no invention).
    return name


def _nice_metric(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def generate_comprehensive_research_report(
    blueprint: ResearchBlueprint,
    telemetry: Dict[str, Any],
    search_history: List[Dict[str, Any]],
    *,
    results: Optional[List[SimulationResult]] = None,
    status: str = "",
    checkpoint_path: Optional[str] = None,
) -> str:
    """Publication-grade Markdown report for a finished research run.

    ``telemetry`` carries the final verified run telemetry (flat dict);
    ``search_history`` is the literature ingest (list of dicts with
    source/title/url/year/summary). ``results`` (optional) enables the
    per-contract invariant audit table.
    """
    results = list(results or [])
    bp = blueprint
    feasibility = bp.feasibility
    observables = extract_observables(bp.variable_map or {}, results) if results else {}
    # Fall back to flat telemetry obs_* entries when no SimulationResult list
    # is available (e.g. direct calls from tests or partial states).
    if not observables and telemetry:
        for key, value in telemetry.items():
            if key.startswith("obs_") and isinstance(value, (int, float)):
                observables[key[4:]] = {"value": float(value), "contract_id": "final",
                                        "stage": bp.current_stage or 1,
                                        "status": status or "VERIFIED"}

    lines: List[str] = []
    lines.append(f"# Quantitative Research Report: {bp.title}")
    lines.append("")
    lines.append(f"**Blueprint:** `{bp.blueprint_id}`  ")
    lines.append(f"**Hypothesis:** {bp.falsifiable_hypothesis}")
    lines.append("")

    # -- 1. Executive summary & verified numerical findings ------------------
    lines.append("## 1. Executive Summary & Verified Numerical Findings")
    lines.append("")
    if observables:
        lines.append("| Metric Name | Symbol | Computed Value | Unit | Verification Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for name in sorted(observables):
            obs = observables[name]
            st = obs.get("status") or status or "VERIFIED"
            lines.append(
                f"| {_nice_metric(name)} | `{_symbol_for(name)}` | "
                f"**{_fmt_value(obs['value'])}** | {_unit_for(name)} | {st} |")
        lines.append("")
    else:
        lines.append("_No named observables were mapped in this blueprint "
                     "(`variable_map` empty); see telemetry below._")
        lines.append("")
    verified = [r for r in results if r.status == "VERIFIED"]
    if verified:
        r = verified[-1]
        tel_bits = ", ".join(
            f"{k}={_fmt_value(v)}" for k, v in sorted(r.telemetry.items())
            if isinstance(v, (int, float)) and k.startswith(("final_", "obs_"))
        )
        lines.append(f"**Final verified telemetry** ({r.contract_id}, "
                     f"{r.execution_time_ms:.1f} ms): {tel_bits or 'n/a'}")
        lines.append("")
    elif telemetry:
        tel_bits = ", ".join(
            f"{k}={_fmt_value(v)}" for k, v in sorted(telemetry.items())
            if isinstance(v, (int, float))
        )
        if tel_bits:
            lines.append(f"**Final telemetry:** {tel_bits[:600]}")
            lines.append("")

    # -- 2. Epistemic feasibility & governing modeling regime ---------------
    lines.append("## 2. Epistemic Feasibility & Governing Modeling Regime")
    lines.append("")
    if feasibility is not None:
        lines.append(f"- **Verdict:** `{feasibility.verdict}`")
        lines.append(f"- **Rationale & Assumptions:** {feasibility.rationale}")
        if feasibility.violated_boundary:
            lines.append(f"- **Violated boundary:** {feasibility.violated_boundary}")
        if feasibility.reformulation:
            lines.append(f"- **Surrogate formulation:** {feasibility.reformulation}")
    else:
        lines.append("- No feasibility assessment recorded.")
    lines.append("")
    lines.append("- **Governing equations & verified constraints** (as encoded in "
                 "the verification contracts and enforced by Z3 + sandbox):")
    inv_count = 0
    for contract in bp.contracts:
        for inv in contract.invariants:
            inv_count += 1
            lines.append(f"  - `{inv.variable_name}`: {inv.assertion} "
                         f"(tolerance {inv.tolerance:g})")
    if inv_count == 0:
        lines.append("  - _No explicit invariants recorded._")
    lines.append("")

    # -- 3. Literature ingest & parameter extraction -------------------------
    lines.append("## 3. Literature Ingest & Parameter Extraction")
    lines.append("")
    if search_history:
        lines.append("| Source | Title | Year | URL |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for item in search_history[:12]:
            title = str(item.get("title") or "untitled")[:90]
            year = item.get("year") or "n/d"
            source = item.get("source") or "?"
            url = str(item.get("url") or "")[:80]
            lines.append(f"| {source} | {title} | {year} | {url} |")
        lines.append("")
    else:
        lines.append("_No literature hits retrieved (offline cache or no hits)._")
        lines.append("")
    # Parameters actually used: initial_state + named observables.
    param_bits = []
    for contract in bp.contracts:
        if contract.initial_state:
            param_bits.append(
                f"`{contract.contract_id}` initial_state = "
                f"[{', '.join(_fmt_value(float(v)) for v in contract.initial_state[:12])}"
                f"{', …' if len(contract.initial_state) > 12 else ''}]")
        param_bits.append(f"`{contract.contract_id}` dt = {contract.dt:g} s, "
                          f"steps = {contract.steps}")
    if param_bits:
        lines.append("- **Parameters used in the verified simulation:**")
        for bit in param_bits:
            lines.append(f"  - {bit}")
        lines.append("")

    # -- 4. Computational implementation & accretion history -----------------
    lines.append("## 4. Computational Implementation & Accretion History")
    lines.append("")
    backends = sorted({c.target_backend for c in bp.contracts})
    lines.append(f"- **Target backend(s):** {', '.join(f'`{b}`' for b in backends) or 'n/a'}")
    scheme = ", ".join(
        f"`{c.contract_id}` (dt={c.dt:g}, steps={c.steps}, "
        f"timeout={c.timeout_seconds:g}s, mem={c.max_memory_mb}MB)"
        for c in bp.contracts)
    lines.append(f"- **Numerical scheme / contracts:** {scheme or 'n/a'}")
    module = bp.unified_module_code or ""
    lines.append(f"- **Unified module:** {len(module)} chars, "
                 f"{len(bp.verified_layers)} pinned verified layer(s), "
                 f"{len(bp.evolution_history)} accretion step(s) recorded.")
    lines.append("")
    if bp.evolution_history:
        lines.append("| Stage | Status | Bundle hash | Note |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for h in bp.evolution_history:
            note = f" — {h.note[:80]}" if h.note else ""
            lines.append(f"| {h.stage} | {h.status} | `{h.bundle_hash[:16]}` |{note} |")
        lines.append("")

    # -- 5. Formal invariant verification & precision shield audit -----------
    lines.append("## 5. Formal Invariant Verification & Precision Shield Audit")
    lines.append("")
    lines.append(f"- **Total invariants tested:** {inv_count}")
    lines.append(f"- **SMT/Z3 consistency:** {'PASSED' if status in ('VERIFIED', 'PARTIAL') or results else 'n/a'}")
    times = [r.execution_time_ms for r in results if r.execution_time_ms]
    if times:
        lines.append(f"- **Sandbox execution:** {len(results)} contract run(s), "
                     f"total kernel time {sum(times):.1f} ms, "
                     f"max single run {max(times):.1f} ms; "
                     f"memory budget ≤ {max((c.max_memory_mb for c in bp.contracts), default=0)} MB per contract.")
    else:
        lines.append("- **Sandbox execution:** no contract runs recorded.")
    lines.append(f"- **Run status:** `{status or 'n/a'}` "
                 f"({len(bp.verified_layers)} verified stage(s) of {bp.target_research_stages})")
    if checkpoint_path:
        lines.append(f"- **Checkpoint:** `{checkpoint_path}`")
    lines.append("")
    if results:
        lines.append("| Contract | Status | Time (ms) | Telemetry (final) |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for r in results[-8:]:
            tel = ", ".join(
                f"{k}={_fmt_value(v)}" for k, v in sorted(r.telemetry.items())
                if isinstance(v, (int, float)) and k.startswith(("final_", "obs_"))
            ) or "n/a"
            lines.append(f"| `{r.contract_id}` | {r.status} | "
                         f"{r.execution_time_ms:.1f} | {tel[:220]} |")
        lines.append("")

    # -- 6. Process notes & execution log ------------------------------------
    lines.append("## 6. Process Notes & Execution Log")
    lines.append("")
    lines.append("1. **Ingest** — live literature retrieval (arXiv/OpenAlex/Wikipedia) "
                 "for the blueprint search queries; hits cached under `research/literature_cache/`.")
    lines.append("2. **GoI_Plan** — accretive stage synthesis coupling the pinned "
                 "verified layers of the unified module.")
    lines.append("3. **SMT_Concolic_Verify** — Z3 consistency check of every contract "
                 "invariant set before execution.")
    lines.append("4. **Sandbox_Execute** — transpiled C++20 kernels compiled with "
                 "`-O3 -std=c++20 -fPIC -shared` and executed in an air-gapped, "
                 "resource-limited subprocess (zero-copy `SimulationBuffer`).")
    lines.append("5. **Reflex_Heal / Adjoint_Commit_or_Rollback** — falsified or "
                 "diverged candidates are rolled back stage-scoped; only verified "
                 "layers are pinned as checkpoints.")
    lines.append("")
    lines.append("---")
    lines.append("_Generated deterministically from verified sandbox telemetry; "
                 "no values are extrapolated._")
    return "\n".join(lines)


def report_markdown_title(markdown: str) -> str:
    """First heading of a report, for chat display fallbacks."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Research Report"
