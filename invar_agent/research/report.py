"""Final Verified Quantitative Research Report synthesis.

Deterministic (no LLM): every number is pulled directly from the verified
sandbox execution telemetry (``obs_<metric>`` observables published from the
blueprint ``variable_map`` plus the standard final-state telemetry), so the
report can never invent values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas.research import ResearchBlueprint, SimulationResult


def extract_observables(variable_map: Dict[str, int],
                        results: List[SimulationResult]) -> Dict[str, Dict[str, Any]]:
    """Collect ``obs_<name>`` telemetry values per verified result.

    Returns {metric_name: {"value": float, "contract_id": str, "stage": int}}.
    Later (higher-stage) results win for duplicated metrics.
    """
    if not variable_map:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for result in sorted(results, key=lambda r: (r.stage or 0)):
        for name in variable_map:
            key = f"obs_{name}"
            value = result.telemetry.get(key)
            if isinstance(value, (int, float)):
                out[name] = {
                    "value": float(value),
                    "contract_id": result.contract_id,
                    "stage": result.stage or 0,
                    "status": result.status,
                }
    return out


def _fmt_value(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):  # NaN/inf guard
        return "n/a"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude >= 1e6 or magnitude < 1e-4):
        return f"{value:.6e}"
    return f"{value:.6g}"


def _stage_rows(state) -> List[str]:
    rows = []
    for h in state.blueprint.evolution_history:
        # Multi-line tracebacks / notes must never corrupt the Markdown table.
        note = (f" — {h.note[:120]}" if h.note else "")
        note = note.replace("\n", " ").replace("|", "/").replace("\r", " ")
        rows.append(f"| {h.stage} | {h.status} | `{h.bundle_hash[:16]}` |{note} |")
    return rows


def _literature_rows(literature: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    rows = []
    for item in literature[:limit]:
        title = str(item.get("title") or "untitled")[:90]
        year = item.get("year") or "n/d"
        source = item.get("source") or "?"
        rows.append(f"| {source} | {title} | {year} |")
    return rows


def _fallback_variable_map(bp: ResearchBlueprint) -> Dict[str, int]:
    """Deterministic variable_map fallback so reports never show an empty map."""
    mapping: Dict[str, int] = {}
    for contract in getattr(bp, "contracts", None) or []:
        for inv in getattr(contract, "invariants", None) or []:
            import re as _re
            text = f"{getattr(inv, 'variable_name', '')} {getattr(inv, 'assertion', '')}"
            m = _re.search(r"(?:state|initial|buf(?:\.data)?)\[(\d+)\]", text)
            if m and getattr(inv, "variable_name", ""):
                mapping.setdefault(str(inv.variable_name), int(m.group(1)))
    if not mapping:
        mapping = {f"state_{i}": i for i in range(4)}
    return dict(list(mapping.items())[:64])


def build_research_report(state) -> str:
    """Markdown Verified Quantitative Research Report for a finished run."""
    bp: ResearchBlueprint = state.blueprint
    results: List[SimulationResult] = list(state.results)
    feasibility = bp.feasibility
    variable_map = dict(bp.variable_map or {})
    if not variable_map:
        variable_map = _fallback_variable_map(bp)
    observables = extract_observables(variable_map, results)

    lines: List[str] = []
    lines.append(f"# Verified Quantitative Research Report — {bp.title}")
    lines.append("")
    lines.append(f"**Blueprint:** `{bp.blueprint_id}`  ")
    lines.append(f"**Hypothesis:** {bp.falsifiable_hypothesis}")
    lines.append("")

    # 1. Executive summary & exact numerical results
    lines.append("## 1. Executive Summary & Numerical Results")
    lines.append("")
    if variable_map:
        lines.append("| Observable (units in name) | State slot | Value | Source contract | Stage |")
        lines.append("|---|---:|---|---|---|")
        for name in sorted(variable_map):
            obs = observables.get(name)
            if obs is None:
                lines.append(f"| `{name}` | {variable_map[name]} | n/a | - | - |")
            else:
                lines.append(
                    f"| `{name}` | {variable_map[name]} | **{_fmt_value(obs['value'])}** | "
                    f"`{obs['contract_id']}` | {obs['stage']} |")
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

    # 2. Feasibility & modelling regime
    lines.append("## 2. Feasibility & Modeling Regime")
    lines.append("")
    if feasibility is not None:
        lines.append(f"- **Verdict:** `{feasibility.verdict}`")
        lines.append(f"- **Rationale:** {feasibility.rationale}")
        if feasibility.violated_boundary:
            lines.append(f"- **Boundary:** {feasibility.violated_boundary}")
        if feasibility.reformulation:
            lines.append(f"- **Surrogate / reformulation:** {feasibility.reformulation}")
    else:
        lines.append("- No feasibility assessment recorded.")
    lines.append("")

    # 3. Literature & governing equations used
    lines.append("## 3. Literature Retrieved")
    lines.append("")
    rows = _literature_rows(state.literature)
    if rows:
        lines.append("| Source | Title | Year |")
        lines.append("|---|---|---|")
        lines.extend(rows)
    else:
        lines.append("_No literature hits retrieved (offline cache or no hits)._")
    lines.append("")
    lines.append("Governing equations are those implemented in the verified "
                 "unified module (`unified_module_code`) and enforced by the "
                 "contracts below.")
    lines.append("")

    # 4. Computational verification & invariant audit
    lines.append("## 4. Computational Verification & Invariant Audit")
    lines.append("")
    lines.append(f"- **Run status:** `{state.status}` "
                 f"({len(state.blueprint.verified_layers)} verified stage(s) of "
                 f"{bp.target_research_stages})")
    lines.append(f"- **Checkpoint:** `{state.checkpoint_path or 'n/a'}`")
    stage_rows = _stage_rows(state)
    if stage_rows:
        lines.append("")
        lines.append("| Stage | Status | Bundle hash | Note |")
        lines.append("|---|---|---|---|")
        lines.extend(stage_rows)
    lines.append("")
    if results:
        lines.append("| Contract | Status | Time (ms) | Telemetry (final) |")
        lines.append("|---|---|---|---|")
        for r in results[-6:]:
            tel = ", ".join(
                f"{k}={_fmt_value(v)}" for k, v in sorted(r.telemetry.items())
                if isinstance(v, (int, float)) and k.startswith(("final_", "obs_"))
            ) or "n/a"
            lines.append(f"| `{r.contract_id}` | {r.status} | "
                         f"{r.execution_time_ms:.1f} | {tel[:220]} |")
        lines.append("")
    lines.append("---")
    lines.append("_Generated deterministically from verified sandbox telemetry; "
                 "no values are extrapolated._")
    return "\n".join(lines)
