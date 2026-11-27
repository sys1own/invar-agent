"""Epistemic self-awareness & feasibility gate for the Research Agent.

Runs before any blueprint is synthesised or simulated. Classifies every
research request into one of four verdicts:

- ``OPTIMAL_DIRECT`` — quantitative/mathematical/dynamical research suited for
  literature search + deterministic native compute verification.
- ``REFORMULATED_SURROGATE`` — target too large for local execution; the agent
  must define a rigorous reduced-order surrogate and explain the reduction.
- ``REFUSED_IMPOSSIBLE`` — mathematically/physically/logically impossible
  (conservation-law violations, undecidable problems, contradictory invariants).
- ``REFUSED_UNSUITABLE`` — subjective/qualitative/historical/clinical tasks
  without a falsifiable mathematical model, or tasks requiring forbidden
  sandbox operations; simulation would produce "math-washing".
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional

from ..schemas.research import FeasibilityAssessment, FeasibilityVerdict

FEASIBILITY_PROMPT = """You are the Epistemic Feasibility Gate of a deterministic computational
research engine. The engine can ONLY: search scientific literature, synthesise
numerical simulation modules in a restricted Python subset, and verify them with
native Rust/C++ kernels, Z3 SMT and concolic execution in an air-gapped sandbox
(no network, no disk I/O, no imports beyond math helpers).

Classify the research request into EXACTLY ONE verdict:
- OPTIMAL_DIRECT: quantitative, mathematical, physical, chemical, biological,
  epidemiological (SIR/SEIR), pharmacological (PK/PD/Hill), statistical/causal
  inference (Monte Carlo/OLS/2SLS/IV/Cramer-Rao), game-theoretic/evolutionary
  (replicator-mutator/Nash), climate/biogeochemical, dynamical or algorithmic
  research that can be falsified by simulation.
- REFORMULATED_SURROGATE: too large for full-scale local execution (e.g. 3D DNS
  turbulence, ab-initio, macro-scale) but admits a rigorous reduced-order /
  1D-2D asymptotic / scaling-law surrogate.
- REFUSED_IMPOSSIBLE: mathematically, physically or logically impossible
  (violates conservation laws while demanding gains, undecidable, or mutually
  contradictory invariants).
- REFUSED_UNSUITABLE: purely subjective, qualitative, historical, clinical or
  literary research without falsifiable governing equations, or requires
  forbidden sandbox operations.

Output strict JSON only: {"verdict": ..., "rationale": ..., "violated_boundary": ...,
"reformulation": ...}. No markdown fences.

Research request: {question}
"""

# -- deterministic heuristics (offline, fast, conservative) -------------------

_IMPOSSIBLE_PATTERNS = [
    (r"net positive energy", "demanding net energy gain"),
    (r"net positive (?:energy|work)", "demanding net energy/work gain"),
    (r"perpetual motion", "perpetual motion"),
    (r"overunity|over-unity|free energy device", "over-unity claim"),
    (r"100\s*%\s*efficien", "impossible efficiency bound"),
    # Explicit over-unity claims only: bare "efficiency"/"eta" would false-
    # positive on normal numbers (viscosity eta = 1.2e-3, speedup ratios).
    (r"\b(?:cop|coefficient\s+of\s+performance)\s*(?:=|>=|>|of)\s*"
     r"(?:1\.[0-9]+|[2-9][0-9]*(?:\.[0-9]+)?)",
     "coefficient of performance above unity"),
    (r"\b(?:thermal|heat[- ]transfer|energy|thermodynamic)\s+efficiency\s*"
     r"(?:=|>=|>|of)\s*(?:1\.[0-9]+|[2-9][0-9]*(?:\.[0-9]+)?)",
     "thermodynamic efficiency above unity"),
    (r"violat\w*\s+(energy\s+)?conservation", "conservation-law violation"),
    (r"create energy from nothing|energy from nothing", "energy creation"),
    (r"halting problem|undecidable", "undecidable problem"),
    (r"contradict\w*\s+(invariant|constraint|assumption)", "contradictory constraints"),
]

_CONSERVATION_WORDS = re.compile(r"conserv\w+|hamiltonian|energy", re.I)
_GAIN_WORDS = re.compile(r"gain|generat\w+|produc\w+|increas\w+|positive energy|net positive", re.I)

_UNSUITABLE_PATTERNS = [
    # NOTE: bare "literature"/"history"/"book"/"paper"/"reading" are
    # intentionally excluded — quantitative research routinely says "ingest
    # literature", "literature scaling constants", "historical atmospheric
    # density data" or "re-entry decelerations". Only strict non-computational
    # collocations refuse.
    (r"\bpoetry\b|\bpoem\b", "poetry/verse subject"),
    (r"\b(?:literary|romantic|victorian|renaissance)\s+"
     r"(?:criticism|poetry|literature|prose|novels?|drama)\b",
     "literary/humanities subject"),
    (r"\bliterary\s+(?:criticism|analysis|theory|review)\b", "literary criticism"),
    (r"\btheological\s+(?:debate|argument|doctrine|controversy)\b",
     "theological debate"),
    (r"\bmoral\s+(?:superiority|philosophy|debate|dilemma)\b", "moral philosophy"),
    (r"causal effect on|effect on .*legislation|influence on .*policy", "non-quantifiable causal claim"),
    (r"\bbeaut(y|iful)\b|\bmeaning of life\b|\bmorality\b|\bethics\b", "subjective value judgement"),
    (r"\bprove\b.*\b(poetry|art|music|culture|society)\b", "unfalsifiable cultural claim"),
]

_TASKS_REQUIRING_MODEL = [
    (r"\bclinical\b", "clinical/wet-lab task without a model"),
]

_SURROGATE_PATTERNS = [
    (r"\b3\s*-?\s*d\b[^.;]*\bturbulen", "3D turbulence simulation"),
    (r"3\s*-?\s*d(irect)?\s*(numerical\s+)?simulation of turbulence|dns turbulence", "3D DNS turbulence"),
    (r"ab\s*-?\s*initio", "ab-initio simulation"),
    (r"full[- ]scale|entire (atmosphere|ocean|climate|genome|brain)", "full-scale system"),
    (r"macro[- ]scale|planetary scale|whole[- ]genome", "macro-scale system"),
    (r"turbulent flow at high reynolds", "high-Reynolds turbulence"),
]

_QUANTITATIVE_HINTS = re.compile(
    r"\b(simulat\w+|model\w*|equation|dynamics?|oscillat\w+|soliton|wave|"
    r"converg\w+|algorithm\w+|numerical|analytic\w+|scaling|invariant|"
    r"conserv\w+|energy|entropy|rk4|euler|newton|lu decomposition|"
    r"pde|ode|spectrum|eigen\w+|matrix|tensor|field theory|quantum|"
    r"thermodynamic\w*|kinetic\w*|fluid\w*|acoustic\w*)\b", re.I)


def _smt_contradiction_core(question: str) -> Optional[list[str]]:
    """Detect direct scalar inequality contradictions (e.g. `M > 10` and `M < 5`)
    and certify emptiness via Z3 SMT Q1 with a minimal UNSAT core."""
    text = str(question or "")
    # Extract simple scalar comparisons like "M > 10", "x <= 5.0", "mass >= 10"
    pat = re.compile(
        r"\b([A-Za-z][A-Za-z0-9_]{0,24})\s*(<=|>=|<|>|==)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b"
    )
    grouped: Dict[str, list[tuple[str, float, str]]] = {}
    for m_obj in pat.finditer(text):
        sym, op, num_s = m_obj.group(1), m_obj.group(2), m_obj.group(3)
        if sym.lower() in {"if", "and", "or", "is", "to", "at", "in", "of", "for", "by", "on"}:
            continue
        try:
            val = float(num_s)
        except ValueError:
            continue
        grouped.setdefault(sym, []).append((op, val, f"{sym} {op} {num_s}"))
    candidates = {sym: clauses for sym, clauses in grouped.items() if len(clauses) >= 2}
    if not candidates:
        return None
    from ..runtime.isolation import run_on_z3_thread
    import z3

    def _check():
        for sym, clauses in candidates.items():
            solver = z3.Solver()
            solver.set("smt.core.minimize", True)
            solver.set(timeout=300)
            var = z3.Real(sym)
            labels: Dict[str, str] = {}
            for idx, (op, val, raw) in enumerate(clauses):
                rv = z3.RealVal(str(val))
                expr = {
                    "<": var < rv,
                    "<=": var <= rv,
                    ">": var > rv,
                    ">=": var >= rv,
                    "==": var == rv,
                }[op]
                lbl = z3.Bool(f"$q1_{idx}")
                labels[str(lbl)] = raw
                solver.assert_and_track(expr, lbl)
            if solver.check() == z3.unsat:
                core = [labels[str(b)] for b in solver.unsat_core() if str(b) in labels]
                if len(core) >= 2:
                    return sorted(core)
        return None

    return run_on_z3_thread(_check)


def _heuristic_verdict(question: str) -> FeasibilityAssessment:
    q = question.lower()
    # ARCH-003 (T-R5): Certify direct scalar contradictions via Z3 SMT Q1 UNSAT core
    # BEFORE any SHBT or surrogate check.
    smt_core = _smt_contradiction_core(question)
    if smt_core:
        return FeasibilityAssessment(
            verdict="REFUSED_IMPOSSIBLE",
            rationale=(
                f"Direct mathematical contradiction certified by SMT Q1 UNSAT core: "
                f"{' AND '.join(smt_core)} has an empty feasible set."
            ),
            violated_boundary="SMT Q1 UNSAT (contradictory scalar constraints)",
            reformulation="Remove or reconcile the contradictory inequality bounds so the feasible set is non-empty.",
            smt_unsat_core=smt_core,
            assessed_by="heuristic_offline",
        )
    # Impossible: explicit pattern, or conservation + gain co-occurrence.
    # ARCH-011: Impossibility checks ALWAYS run before SHBT (SHBT NEVER overrides refusals).
    is_comp_bench = bool(_COMPUTATIONAL_BENCHMARK_HINTS.search(q))
    for pattern, what in _IMPOSSIBLE_PATTERNS:
        if what == "conservation-law violation" and is_comp_bench:
            continue
        if re.search(pattern, q):
            return FeasibilityAssessment(
                verdict="REFUSED_IMPOSSIBLE",
                rationale=(f"The request demands {what}, which is axiomatically "
                           "impossible for a conservative deterministic system."),
                violated_boundary="axiomatic impossibility (conservation laws / decidability)",
                reformulation=("Reformulate as a consistency check (e.g. verify that "
                               "invariants hold and bound the achievable gain) or ask "
                               "for the maximum permissible rate within the constraints."),
                assessed_by="heuristic_offline")
    if _CONSERVATION_WORDS.search(q) and _GAIN_WORDS.search(q) \
            and re.search(r"105%|>\s*100%|exceed\w*.*consumption|self[- ]sustain", q):
        return FeasibilityAssessment(
            verdict="REFUSED_IMPOSSIBLE",
            rationale=("The request simultaneously demands strict conservation of a "
                       "Hamiltonian/energy invariant and net energy gain beyond 100%, "
                       "which is a thermodynamic contradiction."),
            violated_boundary="contradictory SMT constraints (conservation vs. net gain)",
            reformulation=("Split into two verifiable sub-problems: (1) verify "
                           "conservation/invariant bounds, (2) maximise efficiency "
                           "subject to those bounds."),
            assessed_by="heuristic_offline")
    from .shbt_federation import resolve_shbt_solvability
    shbt_res = resolve_shbt_solvability(question)
    if shbt_res.should_activate:
        repos = ", ".join(s.repo for s in shbt_res.matched_repos)
        return FeasibilityAssessment(
            verdict="OPTIMAL_DIRECT",
            rationale=(f"SHBT Frontier Solvability Federation available ({repos}): "
                       "provides open-ledger/WZW/LANR/ZPE governing equations and REGISTRY_ASSUMPTION constants."),
            assessed_by="heuristic_offline")
    unsuitable = _UNSUITABLE_PATTERNS + ([] if _QUANTITATIVE_HINTS.search(q) else _TASKS_REQUIRING_MODEL)
    for pattern, what in unsuitable:
        if re.search(pattern, q):
            return FeasibilityAssessment(
                verdict="REFUSED_UNSUITABLE",
                rationale=(f"The request concerns {what}: it has no falsifiable "
                           "mathematical governing equations, so a deterministic "
                           "simulation would produce fabricated 'math-washing'."),
                violated_boundary="absence of falsifiable mathematical model",
                reformulation=("Recast as a quantitative proxy (e.g. a measurable "
                               "metric with an explicit mathematical model) or use "
                               "the Builder Agent for qualitative artefacts."),
                assessed_by="heuristic_offline")
    for pattern, what in _SURROGATE_PATTERNS:
        if re.search(pattern, q):
            return FeasibilityAssessment(
                verdict="REFORMULATED_SURROGATE",
                rationale=(f"The target ({what}) exceeds what the air-gapped local "
                           "sandbox can execute faithfully; a reduced-order surrogate "
                           "is required."),
                violated_boundary="compute envelope of rust_hin/cpp sandbox",
                reformulation=("Define a 1D/2D reduced-order or asymptotic model "
                               "(e.g. scaling laws, moment closures) and verify its "
                               "invariants instead of the full system."),
                assessed_by="heuristic_offline")
    return FeasibilityAssessment(
        verdict="OPTIMAL_DIRECT",
        rationale=("Quantitative research question compatible with literature "
                   "search + deterministic native compute verification."),
        assessed_by="heuristic_offline")


def format_advisory_hint(heuristic: FeasibilityAssessment) -> str:
    """Format an offline heuristic pre-check as a non-binding advisory hint for the LLM interpreter."""
    verdict = str(getattr(heuristic.verdict, "value", heuristic.verdict))
    if verdict == "OPTIMAL_DIRECT":
        return ""
    return (
        "\n\nADVISORY HEURISTIC PRE-CHECK (non-binding hint; you are the interpreter): "
        f"offline keyword check suggested verdict={verdict} ({heuristic.rationale}). "
        "If the user makes a concrete falsifiable mathematical/physical claim (even a potentially false or "
        "contradictory one like T>=500 and T<=300 or Carnot eta>=0.65), formalize it faithfully as "
        "OPTIMAL_DIRECT so the deterministic formal checker can prove or formally falsify it. "
        "Only issue REFUSED_UNSUITABLE for non-mathematical prose/gibberish or REFUSED_IMPOSSIBLE for "
        "undecidable/non-formalizable requests."
    )


# -- LLM refinement ------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    # Strip reasoning-model chain-of-thought wrappers before fence/JSON parsing.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        from ..latex_normalizer import repair_json_strings
        return repair_json_strings(data)
    return None


def truncate_at_sentence(text: str, limit: int = 1200) -> str:
    """Bound ``text`` to ``limit`` chars, cutting at a complete sentence boundary.

    Mid-word/mid-sentence slicing produced truncated rationales like
    "...Sutton-Graves stagnation heating correlation. No" in reports; this
    prefers the last sentence end within the window, then a word boundary.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "),
              window.rfind(".\n"), window.rfind("!\n"), window.rfind("?\n"))
    if cut > limit // 3:
        return window[:cut + 1].strip()
    space = window.rfind(" ")
    if space > limit // 3:
        return window[:space].strip()
    return window.strip()


def _assessment_from_llm(data: Dict[str, Any]) -> Optional[FeasibilityAssessment]:
    try:
        verdict = FeasibilityVerdict(str(data.get("verdict") or ""))
    except ValueError:
        return None
    reformulation = str(data.get("reformulation") or "").strip()
    if not reformulation and verdict != FeasibilityVerdict.OPTIMAL_DIRECT:
        reformulation = ("Reformulate as a falsifiable quantitative question "
                         "with explicit governing equations and measurable "
                         "invariants that a deterministic sandbox can verify.")
    questions = [str(x).strip() for x in (data.get("questions") or []) if str(x).strip()]
    missing_fields = [str(x).strip() for x in (data.get("missing_fields") or []) if str(x).strip()]
    smt_core = data.get("smt_unsat_core")
    if isinstance(smt_core, list):
        smt_core = [str(x).strip() for x in smt_core if str(x).strip()] or None
    else:
        smt_core = None
    try:
        return FeasibilityAssessment(
            verdict=verdict,
            rationale=truncate_at_sentence(str(data.get("rationale") or ""), 1200) or "LLM assessment",
            violated_boundary=str(data.get("violated_boundary") or "")[:400] or None,
            reformulation=reformulation[:600] or None,
            questions=questions[:8],
            missing_fields=missing_fields[:12],
            smt_unsat_core=smt_core,
            assessed_by="llm")
    except Exception:  # noqa: BLE001
        return None


_COMPUTATIONAL_BENCHMARK_HINTS = re.compile(
    r"\b(integrat\w+|discrete\s+update|update\s+operator|ode|pde|"
    r"simulat\w+|invariant\w*|turbulen\w*|vorticity|hasegawa|wakatani|"
    r"drift[- ]wave|lindblad|kinetics|patankar|oregonator|cayley|"
    r"midpoint|crank[- ]nicolson|symplectic|poisson[- ]bracket|"
    r"casimir|enstrophy|hamiltonian|lyapunov|conservation|stiff\w*|"
    r"dynamical\s+system\w*|reaction\s+network|quantum|plasma|"
    r"seir\w*|epidemiolog\w*|pharmacokinetic\w*|hill\s+inhibition|"
    r"monte\s+carlo|2sls|instrumental\s+variable|cramer[- ]rao|cramér[- ]rao|"
    r"replicator|nash\s+regret|carbon\s+cycle|revelle|arrhenius)\b",
    re.I,
)


def is_computational_dynamical_benchmark(question: str) -> bool:
    """Return True when ``question`` asks for a numerical integrator / dynamical benchmark
    and is NOT a clear-cut heuristic refusal (perpetual motion, COP > 1, poetry, etc.)."""
    q = str(question or "")
    if _heuristic_verdict(q).verdict.startswith("REFUSED"):
        return False
    return bool(_COMPUTATIONAL_BENCHMARK_HINTS.search(q))


_DIRECT_CONTRADICTION_RE = re.compile(
    r"\b(contradict\w*|mutually\s+exclusive|empty\s+(?:feasible\s+)?set|"
    r"disjoint\s+intervals?|intersection\s+is\s+empty|infeasible\s+domain|"
    r"unsatisfiable|inconsistent\s+bounds|cannot\s+(?:both|simultaneously)|"
    r"logically\s+impossible|no\s+(?:real\s+|scalar\s+)?solution\s+exists|"
    r"simultaneously\s+satisfy)\b",
    re.I,
)
_SURROGATE_BENCHMARK_RESCUE_RE = re.compile(
    r"\b(hasegawa|wakatani|drift[- ]wave|triad|patankar|oregonator|stiff\w*|lindblad)\b",
    re.I,
)


def override_refusal_for_computational_benchmark(
    question: str,
    assessment: FeasibilityAssessment,
) -> FeasibilityAssessment:
    """Convert false-positive LLM refusals on dynamical/integrator benchmarks to
    ``REFORMULATED_SURROGATE`` with symmetric/compatible surrogate guidance.

    ARCH-011: SHBT activation NEVER overrides or elevates a feasibility refusal.
    """
    verdict_str = str(getattr(assessment.verdict, "value", assessment.verdict) or "").upper()
    if verdict_str == "REFUSED_IMPOSSIBLE" and not _SURROGATE_BENCHMARK_RESCUE_RE.search(str(question or "")):
        refusal_text = f"{assessment.violated_boundary or ''} {assessment.rationale or ''}"
        if _DIRECT_CONTRADICTION_RE.search(refusal_text):
            return assessment
    if verdict_str in ("REFUSED_IMPOSSIBLE", "REFUSED_UNSUITABLE") and is_computational_dynamical_benchmark(question):
        return FeasibilityAssessment(
            verdict=FeasibilityVerdict.REFORMULATED_SURROGATE,
            rationale=(
                "Converted over-constrained ideal/triad tension to a compatible reduced-order "
                "surrogate dynamical system where all requested invariants hold cleanly. "
                + (assessment.rationale or "")
            )[:1200],
            reformulation=(
                "Adopt a symmetric/compatible reduced-order surrogate formulation (e.g. "
                "symmetric triad coupling satisfying exact conservation laws or regularized "
                "stiff production-destruction operator) so all stated invariants are "
                "mathematically consistent and falsifiable."
            ),
            assessed_by=assessment.assessed_by,
        )
    return assessment


def compile_ingest_ihcg(prompt, evidence, domain, archetype):
    from .meaningfulness import IhcgCompiler, classify_contract_vacuity, verify_entailment_lock
    graph = IhcgCompiler().compile_from_prompt(prompt, evidence, domain, archetype)
    obligations = [o.expression for o in graph.obligations]
    lock = verify_entailment_lock(graph, obligations)
    vacuity = classify_contract_vacuity(graph)
    # ARCH-008: Separate claim truth from model admissibility.
    # 1. If a subset of HYPOTHESIS_TARGET conjuncts are implied (redundant) while
    #    other targets remain non-trivial, demote only the implied conjuncts to
    #    CONSERVATION_GUARD rather than rejecting the entire claim set.
    hyp_obs = [o for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET"]
    if vacuity.redundant_conjuncts and len(vacuity.redundant_conjuncts) < len(hyp_obs):
        red_idxs = {int(i) for i in vacuity.redundant_conjuncts if str(i).isdigit()}
        new_obs = []
        hyp_i = 0
        for o in graph.obligations:
            if o.obligation_class == "HYPOTHESIS_TARGET":
                if hyp_i in red_idxs:
                    new_obs.append(o.model_copy(update={"obligation_class": "CONSERVATION_GUARD"}))
                else:
                    new_obs.append(o)
                hyp_i += 1
            else:
                new_obs.append(o)
        ev_copy = dict(evidence)
        ev_copy["obligations"] = [o.model_dump(mode="json") for o in new_obs]
        graph = IhcgCompiler().compile_from_prompt(prompt, ev_copy, domain, archetype)
        obligations = [o.expression for o in graph.obligations]
        lock = verify_entailment_lock(graph, obligations)
        vacuity = classify_contract_vacuity(graph)

    # 2. ARCH-008: Ingest checks ONLY model/domain satisfiability (Q1=SAT) and
    #    non-tautology (Q3!=UNSAT). A claim that is INFEASIBLE (Q2=UNSAT) or
    #    hard-nonlinear UNKNOWN (Q2=UNKNOWN) over a satisfiable model/domain
    #    MUST seal cleanly so it routes to Mode 1 / 2A / 2B falsification or verification.
    claim_admissible_for_testing = (
        vacuity.classification in ("INFEASIBLE", "UNKNOWN")
        and vacuity.queries.get("Q1") is not None
        and vacuity.queries["Q1"].status == "SAT"
        and (vacuity.queries.get("Q3") is None or vacuity.queries["Q3"].status != "UNSAT")
    )
    if not lock.entailment_locked or (not vacuity.passed and not claim_admissible_for_testing):
        extra = ""
        if vacuity.redundant_conjuncts:
            hyp_exprs = [o.expression for o in graph.obligations if o.obligation_class == "HYPOTHESIS_TARGET"]
            extra = f"; redundant_hint=remove or move to environmental_assumptions/CONSERVATION_GUARD any HYPOTHESIS_TARGET already implied by domain/environment; hypothesis_targets={hyp_exprs}"
        raise ValueError(f"IHCG_INGEST_REJECTED: {vacuity.classification}; entailment_locked={lock.entailment_locked}; redundant_conjuncts={vacuity.redundant_conjuncts}; query_statuses={dict((k,v.status) for k,v in vacuity.queries.items())}{extra}")
    return graph


async def evaluate_research_feasibility(
        question: str,
        llm_complete: Optional[Callable[[str], Any]] = None,
        use_llm: bool = True, *, ihcg_input: Optional[Dict[str, Any]] = None) -> FeasibilityAssessment:
    """Classify a research request.

    When an LLM is configured (``llm_complete is not None and use_llm``), heuristic
    keyword checks are passed strictly as non-binding ADVISORY HINTS to the LLM
    interpreter. Offline heuristic refusals (``assessed_by='heuristic_offline'``)
    apply only when no LLM is configured.
    """
    if ihcg_input is not None:
        try: compile_ingest_ihcg(question, **ihcg_input)
        except ValueError as error:
            return FeasibilityAssessment(verdict="REFUSED_UNSUITABLE", rationale=str(error), violated_boundary="Phi_lock/Phi_nonvac")
    heuristic = _heuristic_verdict(question)
    if llm_complete is not None and use_llm:
        try:
            prompt = FEASIBILITY_PROMPT.replace("{question}", question) + format_advisory_hint(heuristic)
            raw = await llm_complete(prompt)
            data = _extract_json(str(raw))
            if data is not None:
                llm_assessment = _assessment_from_llm(data)
                if llm_assessment is not None:
                    return override_refusal_for_computational_benchmark(question, llm_assessment)
        except Exception:  # noqa: BLE001 - offline heuristic stands on LLM failure
            pass
    return heuristic
