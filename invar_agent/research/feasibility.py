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

from ..schemas.research import FeasibilityAssessment

FEASIBILITY_PROMPT = """You are the Epistemic Feasibility Gate of a deterministic computational
research engine. The engine can ONLY: search scientific literature, synthesise
numerical simulation modules in a restricted Python subset, and verify them with
native Rust/C++ kernels, Z3 SMT and concolic execution in an air-gapped sandbox
(no network, no disk I/O, no imports beyond math helpers).

Classify the research request into EXACTLY ONE verdict:
- OPTIMAL_DIRECT: quantitative, mathematical, physical, dynamical or algorithmic
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
    (r"perpetual motion", "perpetual motion"),
    (r"overunity|over-unity|free energy device", "over-unity claim"),
    (r"100\s*%\s*efficien", "impossible efficiency bound"),
    (r"violat\w*\s+(energy\s+)?conservation", "conservation-law violation"),
    (r"create energy from nothing|energy from nothing", "energy creation"),
    (r"halting problem|undecidable", "undecidable problem"),
    (r"contradict\w*\s+(invariant|constraint|assumption)", "contradictory constraints"),
]

_CONSERVATION_WORDS = re.compile(r"conserv\w+|hamiltonian|energy", re.I)
_GAIN_WORDS = re.compile(r"gain|generat\w+|produc\w+|increas\w+|positive energy|net positive", re.I)

_UNSUITABLE_PATTERNS = [
    # NOTE: bare "literature" is intentionally excluded — quantitative research
    # requests legitimately say "search the literature".
    (r"\bpoetry\b|\bpoem\b|\bliterary\b|\brhym(e|es|ing)\b", "literary/subjective subject"),
    (r"\bhistor(y|ical)\b", "historical interpretation"),
    (r"causal effect on|effect on .*legislation|influence on .*policy", "non-quantifiable causal claim"),
    (r"\bbeaut(y|iful)\b|\bmeaning of life\b|\bmorality\b|\bethics\b", "subjective value judgement"),
    (r"\bclinical\b(?!.*(trial|endpoint|biomarker|dose))", "clinical/wet-lab task without a model"),
    (r"\bprove\b.*\b(poetry|art|music|culture|society)\b", "unfalsifiable cultural claim"),
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


def _heuristic_verdict(question: str) -> FeasibilityAssessment:
    q = question.lower()
    # Impossible: explicit pattern, or conservation + gain co-occurrence.
    for pattern, what in _IMPOSSIBLE_PATTERNS:
        if re.search(pattern, q):
            return FeasibilityAssessment(
                verdict="REFUSED_IMPOSSIBLE",
                rationale=(f"The request demands {what}, which is axiomatically "
                           "impossible for a conservative deterministic system."),
                violated_boundary="axiomatic impossibility (conservation laws / decidability)",
                reformulation=("Reformulate as a consistency check (e.g. verify that "
                               "invariants hold and bound the achievable gain) or ask "
                               "for the maximum permissible rate within the constraints."),
                assessed_by="heuristic")
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
            assessed_by="heuristic")
    for pattern, what in _UNSUITABLE_PATTERNS:
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
                assessed_by="heuristic")
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
                assessed_by="heuristic")
    return FeasibilityAssessment(
        verdict="OPTIMAL_DIRECT",
        rationale=("Quantitative research question compatible with literature "
                   "search + deterministic native compute verification."),
        assessed_by="heuristic")


# -- LLM refinement ------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
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
    return data if isinstance(data, dict) else None


def _assessment_from_llm(data: Dict[str, Any]) -> Optional[FeasibilityAssessment]:
    verdict = data.get("verdict")
    if verdict not in ("OPTIMAL_DIRECT", "REFORMULATED_SURROGATE",
                       "REFUSED_IMPOSSIBLE", "REFUSED_UNSUITABLE"):
        return None
    try:
        return FeasibilityAssessment(
            verdict=verdict,
            rationale=str(data.get("rationale") or "")[:1000] or "LLM assessment",
            violated_boundary=str(data.get("violated_boundary") or "")[:400],
            reformulation=str(data.get("reformulation") or "")[:600],
            assessed_by="llm")
    except Exception:  # noqa: BLE001
        return None


async def evaluate_research_feasibility(
        question: str,
        llm_complete: Optional[Callable[[str], Any]] = None,
        use_llm: bool = True) -> FeasibilityAssessment:
    """Classify a research request; heuristic first, optional Tier-2 refinement.

    The deterministic heuristic is authoritative for clear-cut refusals; the LLM
    refinement can override OPTIMAL_DIRECT / SURROGATE verdicts and sharpen
    rationales, but can never *downgrade* a heuristic REFUSAL into a run.
    """
    heuristic = _heuristic_verdict(question)
    if heuristic.verdict in ("REFUSED_IMPOSSIBLE", "REFUSED_UNSUITABLE"):
        return heuristic
    if llm_complete is not None and use_llm:
        try:
            raw = await llm_complete(FEASIBILITY_PROMPT.replace("{question}", question))
            data = _extract_json(str(raw))
            if data is not None:
                llm_assessment = _assessment_from_llm(data)
                if llm_assessment is not None:
                    # LLM may escalate to a refusal or surrogate, never un-refuse.
                    return llm_assessment
        except Exception:  # noqa: BLE001 - heuristic stands on LLM failure
            pass
    return heuristic
