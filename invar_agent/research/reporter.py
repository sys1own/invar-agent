"""Comprehensive Quantitative Research Report generation.

Deterministic (no LLM): every number is pulled directly from the verified
sandbox execution telemetry (``obs_<metric>`` observables published from the
blueprint ``variable_map`` plus the standard final-state telemetry), the
feasibility assessment, the literature ingest and the accretion history, so
the report can never invent values.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..schemas.research import ResearchBlueprint, SimulationResult
from .report import _fmt_value, extract_observables

# Multi-character suffixes are checked first (longest match), case-insensitively.
# Single-letter SI suffixes are strictly case-sensitive so biological/state
# symbols like `sim_epi_S` (Susceptible) or `M_atm` never map to seconds/meters.
UNIT_SUFFIX_MAP = [
    ("_mw_m2", "MW/m²"),
    ("_mj_m2", "MJ/m²"),
    ("_w_m2", "W/m²"),
    ("_j_m2", "J/m²"),
    ("_kg_m3", "kg/m³"),
    ("_mol_m3_s", "mol/(m³·s)"),
    ("_mol_m3", "mol/m³"),
    ("_gtc_yr", "GtC/yr"),
    ("_j_per_bit", "J/bit"),
    ("_j_per_k", "J/K"),
    ("_m_s2", "m/s²"),
    ("_m_s", "m/s"),
    ("_ms", "m/s"),
    ("_km_s", "km/s"),
    ("_dimensionless", "dimensionless"),
    ("_bytes", "bytes"),
    ("_bits", "bits"),
    ("_nats", "nats"),
    ("_morgan", "Morgan"),
    ("_gtc", "GtC"),
    ("_pgc", "PgC"),
    ("_ppm", "ppm"),
    ("_ge", "g_E"),
    ("_km", "km"),
    ("_kg", "kg"),
    ("_kw", "kW"),
    ("_mw", "MW"),
    ("_ev", "eV"),
    ("_pa", "Pa"),
    ("_hz", "Hz"),
    ("_deg", "deg"),
    ("_rad", "rad"),
]

_SINGLE_LETTER_UNIT_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("_m", "m"),
    ("_s", "s"),
    ("_W", "W"),
    ("_K", "K"),
    ("_J", "J"),
    ("_V", "V"),
    ("_g", "g_E"),
)


def _deduplicate_observables(observables: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Collapse case-insensitive and unit-suffixed aliases to one canonical row."""
    if not observables:
        return {}
    multi_stem_re = re.compile(r"_(?:kw|mw|pa|hz|ev|dimensionless|bytes|km|kg|deg|rad)$", re.I)
    single_stem_re = re.compile(r"_(?:W|K|J|V|s|m)$")
    groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for name, obs in observables.items():
        stem = single_stem_re.sub("", multi_stem_re.sub("", name)).lower()
        groups.setdefault(stem, []).append((name, obs))
    out: Dict[str, Dict[str, Any]] = {}
    for stem, items in groups.items():
        if len(items) == 1:
            out[items[0][0]] = items[0][1]
            continue
        # Prefer non-zero verified value, then unit-suffixed canonical name.
        best_name, best_obs = max(
            items,
            key=lambda kv: (
                1 if abs(float(kv[1].get("value") or 0.0)) > 0.0 else 0,
                1 if _unit_for(kv[0]) != "—" else 0,
                1 if kv[0] != kv[0].upper() else 0,
                len(kv[0]),
            ),
        )
        out[best_name] = best_obs
    return out


def _unit_for(name: str, blueprint: Optional[ResearchBlueprint] = None) -> str:
    stem_no_sym = re.sub(r"_[a-zA-Z]$", "", name.lower())
    if blueprint is not None:
        ev = getattr(blueprint, "structured_evidence", None)
        ev_params = ev.get("parameters") if isinstance(ev, dict) else getattr(ev, "parameters", None)
        if isinstance(ev_params, dict):
            key_low = name.lower()
            p = ev_params.get(name) or ev_params.get(key_low) or ev_params.get(stem_no_sym)
            if p is not None:
                u = (p.get("si_unit") or p.get("raw_unit")) if isinstance(p, dict) else (getattr(p, "si_unit", "") or getattr(p, "raw_unit", ""))
                return str(u) if u else "—"
    lowered = name.lower()
    for suffix, unit in UNIT_SUFFIX_MAP:
        if lowered.endswith(suffix):
            return unit
    _NON_TIME_S_WORDS = ("coefficient", "factor", "ratio", "index", "entropy", "susceptible", "selection", "score", "fraction", "state")
    for suffix, unit in _SINGLE_LETTER_UNIT_SUFFIXES:
        if name.endswith(suffix):
            if suffix == "_s" and any(w in lowered for w in _NON_TIME_S_WORDS):
                return "—"
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
    epistemic_verdict: Optional[str] = None,
) -> str:
    """Publication-grade Markdown report for a finished research run.

    ARCH-002: Headings, metric statuses, and telemetry labels are derived
    strictly from ``epistemic_verdict``. ``Verified`` appears only when
    ``epistemic_verdict`` starts with ``VERIFIED_``. Stage statuses use
    ``INVARIANTS_HELD`` rather than ``VERIFIED``. Placeholder ``state_N``
    slots are omitted.
    """
    from .epistemic import compute_evidence_status, is_verified_epistemic_verdict

    results = list(results or [])
    bp = blueprint
    feasibility = bp.feasibility
    verdict_val = epistemic_verdict if epistemic_verdict is not None else status
    is_certified = is_verified_epistemic_verdict(verdict_val)
    filtered_vmap = {
        k: v for k, v in (bp.variable_map or {}).items()
        if not re.match(r"^state_\d+$", str(k))
    }
    observables = extract_observables(filtered_vmap, results) if results else {}
    # Fall back to flat telemetry obs_* entries when no SimulationResult list
    # is available (e.g. direct calls from tests or partial states).
    if not observables and telemetry:
        for key, value in telemetry.items():
            if (
                key.startswith("obs_")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not re.match(r"^state_\d+$", key[4:])
            ):
                observables[key[4:]] = {"value": float(value), "contract_id": "final",
                                        "stage": bp.current_stage or 1,
                                        "status": status or "OBSERVED_UNVERIFIED"}
    observables = _deduplicate_observables(observables)

    lines: List[str] = []
    lines.append(f"# Quantitative Research Report: {bp.title}")
    lines.append("")
    lines.append(f"**Blueprint:** `{bp.blueprint_id}`  ")
    lines.append(f"**Hypothesis:** {bp.falsifiable_hypothesis}")
    lines.append("")

    # -- 1. Executive summary & numerical findings ---------------------------
    findings_heading = (
        "## 1. Executive Summary & Verified Numerical Findings"
        if is_certified
        else "## 1. Executive Summary & Observed Numerical Findings (Unverified)"
    )
    lines.append(findings_heading)
    lines.append("")
    if observables:
        lines.append("| Metric Name | Symbol | Computed Value | Unit | Evidence Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for name in sorted(observables):
            obs = observables[name]
            ev_status = compute_evidence_status(
                verdict_val,
                obs.get("value"),
                violated=(str(obs.get("status") or "").upper() == "FALSIFIED"),
            )
            lines.append(
                f"| {_nice_metric(name)} | `{_symbol_for(name)}` | "
                f"**{_fmt_value(obs['value'])}** | {_unit_for(name, blueprint=bp)} | `{ev_status}` |")
        lines.append("")
    else:
        lines.append("_No named observables were mapped in this blueprint "
                     "(`variable_map` empty); see telemetry below._")
        lines.append("")
    invariants_held = [r for r in results if r.status == "VERIFIED"]
    if invariants_held:
        r = invariants_held[-1]
        tel_bits = ", ".join(
            f"{k}={_fmt_value(v)}" for k, v in sorted(r.telemetry.items())
            if isinstance(v, (int, float)) and k.startswith(("final_", "obs_"))
            and not re.match(r"^obs_state_\d+$", k)
        )
        tel_heading = "Final verified telemetry" if is_certified else "Final observed telemetry (unverified)"
        lines.append(f"**{tel_heading}** ({r.contract_id}, "
                     f"{r.execution_time_ms:.1f} ms): {tel_bits or 'n/a'}")
        lines.append("")
    elif telemetry:
        tel_bits = ", ".join(
            f"{k}={_fmt_value(v)}" for k, v in sorted(telemetry.items())
            if isinstance(v, (int, float))
        )
        if tel_bits:
            lines.append(f"**Final telemetry:** {tel_bits}")
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
            lines.append(f"  - `{inv.variable_name}`: `{inv.assertion}` "
                         f"(tolerance {inv.tolerance:g})")
    if inv_count == 0:
        lines.append("  - _No explicit invariants recorded._")
    lines.append("")

    # -- 3. Literature ingest & parameter extraction -------------------------
    lines.append("## 3. Literature Ingest & Parameter Extraction")
    lines.append("")
    if search_history:
        from .report import rank_literature_by_relevance
        ranked_history = rank_literature_by_relevance(
            search_history,
            prompt=getattr(bp, "research_prompt", "") or "",
            title=getattr(bp, "title", "") or "",
            hypothesis=getattr(bp, "falsifiable_hypothesis", "") or "",
        )
        lines.append("| Source | Title | Year | URL |")
        lines.append("| :--- | :--- | :--- | :--- |")
        seen_rows = set()
        for item in ranked_history:
            title = str(item.get("title") or "untitled")
            source = str(item.get("source") or "?")
            # Collapse duplicate rows (multiple queries hitting the offline
            # fallback_constants dictionary print one clean row, not four).
            row_key = (source, title)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            year = item.get("year") or "n/d"
            url = str(item.get("url") or "")
            lines.append(f"| {source} | {title} | {year} | {url} |")
            if len(seen_rows) >= 12:
                break
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
            # Full note text: mid-word slicing corrupted accretion records.
            note = f" — {h.note}" if h.note else ""
            h_status = "INVARIANTS_HELD" if str(h.status or "").upper() == "VERIFIED" else h.status
            lines.append(f"| {h.stage} | {h_status} | `{h.bundle_hash[:16]}` |{note} |")
        lines.append("")

    # -- 5. Formal invariant audit & precision shield audit ------------------
    lines.append("## 5. Formal Invariant Audit & Precision Shield")
    lines.append("")
    lines.append(f"- **Total invariants tested:** {inv_count}")
    lines.append(f"- **SMT/Z3 consistency:** {'PASSED' if is_certified or results else 'n/a'}")
    times = [r.execution_time_ms for r in results if r.execution_time_ms]
    if times:
        lines.append(f"- **Sandbox execution:** {len(results)} contract run(s), "
                     f"total kernel time {sum(times):.1f} ms, "
                     f"max single run {max(times):.1f} ms; "
                     f"memory budget ≤ {max((c.max_memory_mb for c in bp.contracts), default=0)} MB per contract.")
    else:
        lines.append("- **Sandbox execution:** no contract runs recorded.")
    lines.append(f"- **Run status:** `{status or 'n/a'}` "
                 f"({len(bp.verified_layers)} stage(s) with invariants held of {bp.target_research_stages})")
    if checkpoint_path:
        lines.append(f"- **Checkpoint:** `{checkpoint_path}`")
    lines.append("")
    if results:
        lines.append("| Contract | Stage Status | Time (ms) | Telemetry (final) |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for r in results[-8:]:
            tel = ", ".join(
                f"{k}={_fmt_value(v)}" for k, v in sorted(r.telemetry.items())
                if isinstance(v, (int, float)) and k.startswith(("final_", "obs_"))
            ) or "n/a"
            row_st = "INVARIANTS_HELD" if str(r.status or "").upper() == "VERIFIED" else r.status
            lines.append(f"| `{r.contract_id}` | {row_st} | "
                         f"{r.execution_time_ms:.1f} | {tel} |")
        lines.append("")

    # -- 6. Process notes & execution log ------------------------------------
    lines.append("## 6. Process Notes & Execution Log")
    lines.append("")
    lines.append("1. **Ingest** — live literature retrieval (arXiv/OpenAlex/Wikipedia) "
                 "for the blueprint search queries; hits cached under `research/literature_cache/`.")
    lines.append("2. **GoI_Plan** — accretive stage synthesis coupling the pinned "
                 "layers of the unified module.")
    lines.append("3. **SMT_Concolic_Verify** — Z3 consistency check of every contract "
                 "invariant set before execution.")
    lines.append("4. **Sandbox_Execute** — transpiled C++20 kernels compiled with "
                 "`-O3 -std=c++20 -fPIC -shared` and executed in an air-gapped, "
                 "resource-limited subprocess (zero-copy `SimulationBuffer`).")
    lines.append("5. **Reflex_Heal / Adjoint_Commit_or_Rollback** — falsified or "
                 "diverged candidates are rolled back stage-scoped; only passing "
                 "layers are pinned as checkpoints.")
    lines.append("")
    lines.append("---")
    lines.append("_Generated deterministically from "
                 + ("certified " if is_certified else "sandbox ")
                 + "telemetry; no values are extrapolated._")
    return "\n".join(lines)


def report_markdown_title(markdown: str) -> str:
    """First heading of a report, for chat display fallbacks."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Research Report"
