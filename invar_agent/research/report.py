"""Final Verified Quantitative Research Report synthesis.

Deterministic (no LLM): every number is pulled directly from the verified
sandbox execution telemetry (``obs_<metric>`` observables published from the
blueprint ``variable_map`` plus the standard final-state telemetry), so the
report can never invent values.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..latex_normalizer import format_contract_constraint_math, normalize_report_latex
from ..schemas.research import ResearchBlueprint, SimulationResult


def extract_observables(variable_map: Dict[str, int],
                        results: List[SimulationResult]) -> Dict[str, Dict[str, Any]]:
    """Collect ``obs_<name>`` telemetry values per verified result.

    Returns {metric_name: {"value": float, "contract_id": str, "stage": int, "status": str}}.
    Later (higher-stage) VERIFIED results win; a FALSIFIED result never
    downgrades a metric that was already VERIFIED.
    """
    if not variable_map:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for result in sorted(results, key=lambda r: (r.stage or 0)):
        if result.status != 'VERIFIED':
            continue
        for name in variable_map:
            if name.lower().endswith(('_positivity', '_nonnegative', '_nonneg', '_conservation', '_window', '_bound', '_finite', '_check')):
                continue
            key = f"obs_{name}"
            value = result.telemetry.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                prev = out.get(name)
                if prev is not None and prev.get("status") == "VERIFIED" and result.status != "VERIFIED":
                    continue
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


def rank_literature_by_relevance(
    literature: List[Dict[str, Any]],
    *,
    prompt: str = "",
    title: str = "",
    hypothesis: str = "",
) -> List[Dict[str, Any]]:
    """Rank deduplicated literature hits by prompt/domain relevance and Stage 1 priority."""
    import re as _re
    stop = {
        "with", "from", "that", "this", "into", "over", "under", "between",
        "using", "model", "models", "system", "state", "stage", "stages",
        "verify", "verification", "simulate", "simulation", "sweep", "protocol",
        "across", "explicit", "euler", "step", "steps", "time", "initial",
        "final", "range", "values", "bound", "bounds", "finite", "nonneg",
    }
    ctx_tokens = {
        tok for tok in _re.findall(r"[a-z]{4,}", f"{prompt} {title} {hypothesis}".lower())
        if tok not in stop
    }
    off_re = _re.compile(
        r"\b(?:covid[- ]?19|sars[- ]?cov[- ]?2|burn\s+contracture|post[- ]?burn|"
        r"wound\s+healing|randomized\s+controlled\s+trial|case\s+report)\b",
        _re.I,
    )
    clinical_ctx = bool(off_re.search(prompt or "") or off_re.search(hypothesis or ""))
    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    seen: set = set()
    for idx, item in enumerate(literature or []):
        if not isinstance(item, dict):
            continue
        title_s = str(item.get("title") or "untitled")
        source_s = str(item.get("source") or "?")
        url_s = str(item.get("url") or "").strip().lower().rstrip("/")
        norm_t = _re.sub(r"[^a-z0-9]+", " ", title_s.lower()).strip()
        key = ("url", url_s) if url_s else ("title", source_s, norm_t or title_s)
        if key in seen:
            continue
        seen.add(key)
        summary_s = str(item.get("summary") or "")
        t_toks = set(_re.findall(r"[a-z]{4,}", title_s.lower()))
        s_toks = set(_re.findall(r"[a-z]{4,}", summary_s.lower()))
        score = 3.0 * len(ctx_tokens & t_toks) + 1.0 * len(ctx_tokens & s_toks)
        if int(item.get("stage", 1) or 1) == 1:
            score += 4.0
        if item.get("equations"):
            score += 1.5
        if item.get("constants"):
            score += 1.0
        if not clinical_ctx and off_re.search(f"{title_s} {summary_s}"):
            score -= 8.0
        scored.append((score, -idx, item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in scored]


def _literature_rows(
    literature: List[Dict[str, Any]],
    limit: int = 12,
    *,
    prompt: str = "",
    title: str = "",
    hypothesis: str = "",
) -> List[str]:
    ranked = rank_literature_by_relevance(
        literature, prompt=prompt, title=title, hypothesis=hypothesis
    )
    rows = []
    for item in ranked[:limit]:
        t = str(item.get("title") or "untitled")[:90]
        year = item.get("year") or "n/d"
        source = item.get("source") or "?"
        rows.append(f"| {source} | {t} | {year} |")
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


def render_meaningfulness_certificate(certificate, preserved=None) -> str:
    from .epistemic import TRUST_BOUNDARY, parse_run_certificate
    certificate = parse_run_certificate(certificate)
    lines = ["## Epistemic Meaningfulness Certificate", "", f"**Epistemic verdict:** `{certificate.epistemic_verdict.value}`",
             f"**BLAKE3 receipt:** `{certificate.blake3_receipt}`", ""]
    if hasattr(certificate, "ihcg_anchor"):
        from .meaningfulness import meaningfulness_gates
        lines += ["| Gate | Passed |", "|---|---|"]
        lines += [f"| `{gate}` | {passed} |" for gate, passed in meaningfulness_gates(certificate).items()]
    else: lines += [f"Unverified: {certificate.failure_reason}. No formal hypothesis was fabricated for this legacy run."]
    audits = preserved or getattr(certificate, "preserved_vs_enforced", {})
    if audits:
        lines += ["", "| Stage/domain | Total | Preserved | Projected | Stabilized | Enforcement ratio |", "|---|---:|---:|---:|---:|---:|"]
        for domain, audit in sorted(audits.items()):
            lines.append(f"| {domain} | {audit.total_steps} | {audit.preserved_steps} | {audit.projected_steps} | {audit.stabilized_steps} | {audit.enforcement_ratio:.6g} |")
    lines += ["", "**Trust Boundary (TCB):** " + "; ".join(TRUST_BOUNDARY) + "."]
    lines += ["", "_Formal conclusions are conditional on the TCB. Rewrite sampling and differential/FD checks are empirical; libm bounds are assumed, and cooperative deadlines are not hard preemption. Unsupported interval operations fail closed._"]
    return "\n".join(lines)


def build_research_report(state) -> str:
    """Markdown Verified Quantitative Research Report for a finished run."""
    bp: ResearchBlueprint = state.blueprint
    results: List[SimulationResult] = list(state.results)
    feasibility = bp.feasibility
    variable_map = dict(bp.variable_map or {})
    if not variable_map:
        variable_map = _fallback_variable_map(bp)
    observables = extract_observables(variable_map, results)
    variable_map = {name:slot for name,slot in variable_map.items() if name in observables}

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
            if isinstance(v, (int, float)) and math.isfinite(v)
            and (k.startswith('final_') or (k.startswith('obs_') and k[4:] in variable_map))
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
    if getattr(bp, "contracts", None):
        lines.append("")
        lines.append("**Governing Contract Constraints:**")
        for c in bp.contracts:
            for inv in (getattr(c, "invariants", None) or []):
                math_expr = format_contract_constraint_math(inv.assertion, variable_map)
                lines.append(f"- `{c.contract_id}` / `{inv.variable_name}`: {math_expr} (`{inv.assertion}`)")
    lines.append("")

    # 3. Literature & governing equations used
    lines.append("## 3. Literature Retrieved")
    lines.append("")
    rows = _literature_rows(
        state.literature,
        prompt=getattr(bp, "research_prompt", "") or "",
        title=getattr(bp, "title", "") or "",
        hypothesis=getattr(bp, "falsifiable_hypothesis", "") or "",
    )
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
    certificate = getattr(state, "meaningfulness_certificate", None)
    if certificate is not None: lines += ["", render_meaningfulness_certificate(certificate, getattr(state, "preserved_vs_enforced", {}))]
    return normalize_report_latex("\n".join(lines))
