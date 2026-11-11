"""Hermetic zero-LLM-token verification of the literature-optimization upgrade.

Covers: SI unit normalization, LaTeX/wikitext -> v9 python hints, multi-channel
connector routing + offline parsers, execution-strategy recommender, closed-loop
diagnostic re-search (unit scaling + DIVERGED), and end-to-end evidence-grounded
vault commit. NO LLM API calls anywhere.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from invar_agent.research.extractor import (
    QuantitativeInsightExtractor,
    classify_registry_domains,
    extract_ar5iv_equations,
    extract_infobox_parameters,
    extract_wikitext_math,
    latex_to_python,
    normalize_to_si,
    parse_equation,
    recommend_execution_strategy,
)
from invar_agent.research.search import (
    LiteratureResult,
    LiteratureSearchClient,
    _channels_for_prompt,
    classify_research_domain,
)
from invar_agent.research.state_machine import (
    ResearchStateMachine,
    formulate_diagnostic_search_query,
    unit_scaling_diagnostic,
    validate_blueprint_grounding,
)
from invar_agent.research.vault import ResearchVaultManager
from invar_agent.context_compactor import build_research_vault_context
from invar_agent.schemas.research import (
    ResearchBlueprint,
    StructuredResearchEvidence,
    VerificationContract,
    InvariantConstraint,
)


# ---------------------------------------------------------------------------
# Test 1: SI unit normalizer
# ---------------------------------------------------------------------------

def test_si_unit_normalizer():
    assert normalize_to_si(5.8, "km/s") == (5800.0, "m/s")
    v, u = normalize_to_si(0.25, "c")
    assert u == "m/s" and math.isclose(v, 74948114.5, rel_tol=1e-9)
    assert normalize_to_si(0.020, "kg/m^3") == (0.020, "kg/m^3")
    v, u = normalize_to_si(15.6, "g/cm^3")
    assert u == "kg/m^3" and math.isclose(v, 15600.0, rel_tol=1e-9)
    v, u = normalize_to_si(2.84, "MW/m^2")
    assert u == "W/m^2" and math.isclose(v, 2.84e6, rel_tol=1e-9)
    v, u = normalize_to_si(350, "MPa")
    assert u == "Pa" and math.isclose(v, 3.5e8, rel_tol=1e-9)
    v, u = normalize_to_si(150, "mT")
    assert u == "T" and math.isclose(v, 0.15, rel_tol=1e-9)
    v, u = normalize_to_si(-12.5, "deg")
    assert u == "rad" and math.isclose(v, -12.5 * math.pi / 180.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 2: LaTeX & wikitext <math> -> v9_transpiler_friendly python hints
# ---------------------------------------------------------------------------

def test_latex_equation_extractor():
    # Wikipedia wikitext <math> block (Sutton-Graves stagnation heat flux).
    wikitext = ("Mars entry heating. <math>q_s = k \\sqrt{\\frac{\\rho}{R_n}} v^3</math> "
                "more text {{math|g = G\\frac{M}{r^2}}} tail")
    math_blocks = extract_wikitext_math(wikitext)
    assert any("sqrt" in b for b in math_blocks)
    extractor = QuantitativeInsightExtractor()
    eqs = extractor.equations_from_wikitext(wikitext, source="Wikipedia-Math")
    sg = next((e for e in eqs if "sqrt" in e.rhs_latex), None)
    assert sg is not None, f"no Sutton-Graves equation extracted: {eqs}"
    assert "math.sqrt" in sg.rhs_python_hint
    assert "math.pow" in sg.rhs_python_hint
    assert set(sg.variables) >= {"k", "rho", "R_n", "v"}

    # ar5iv HTML5 equation extractor.
    ar5iv_html = (
        '<div class="ltx_equation"><math alttext="\\frac{dv}{dt} = -\\frac{1}{2}'
        '\\rho v^2 \\frac{C_D A}{m}" display="block">x</math></div>'
        '<p>text</p><math alttext="q = k \\sqrt{\\rho} v^3" class="ltx_equation">y</math>')
    ar5iv_eqs = extract_ar5iv_equations(ar5iv_html)
    assert ar5iv_eqs, "ar5iv extractor returned nothing"
    aeq = extractor.equations_from_ar5iv(ar5iv_html, source="ar5iv")
    assert aeq and all(isinstance(e.rhs_python_hint, str) for e in aeq)
    dv = next((e for e in aeq if "dv" in e.lhs), None)
    assert dv is not None
    assert "math.pow" in dv.rhs_python_hint or "rho" in dv.rhs_python_hint

    # Direct LaTeX -> python hint spot checks.
    assert "math.sqrt" in latex_to_python(r"\sqrt{\frac{a}{b}}")
    assert "math.pow" in latex_to_python(r"v^3")
    assert "math.exp" in latex_to_python(r"e^{-h/H}")
    eq = parse_equation(r"q_s = k \sqrt{\frac{\rho}{R_n}} v^3")
    assert eq is not None and eq.lhs == "q_s"

    # Infobox parameter extraction.
    infobox = "| mass = 6.4171e23 kg\n| scale height = 11.1 km\n| not a param = hello"
    params = extract_infobox_parameters(infobox, source="Wikipedia-Math")
    names = {p.name for p in params}
    assert "mass" in names and "scale_height" in names
    mass = next(p for p in params if p.name == "mass")
    assert mass.si_unit == "kg" and math.isclose(mass.si_value, 6.4171e23, rel_tol=1e-6)
    h = next(p for p in params if p.name == "scale_height")
    assert h.si_unit == "m" and math.isclose(h.si_value, 11100.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: multi-channel routing & offline connector parsers
# ---------------------------------------------------------------------------

def test_multi_channel_search_and_oeis_pubchem_parsers():
    # Domain classification routing.
    aero = classify_research_domain(
        "Simulate Mars entry interface aerothermodynamics with heat flux")
    assert {"equations", "peer_reviewed", "constants"} <= aero
    assert "discrete_oeis" not in aero and "chemical_pubchem" not in aero

    discrete = classify_research_domain(
        "Verify the Catalan recurrence and Fibonacci integer sequence")
    assert "discrete_oeis" in discrete

    chem = classify_research_domain(
        "Methane combustion kinetics reaction rates Arrhenius chemical species")
    assert "chemical_pubchem" in chem

    # Source lists route to the right connectors.
    assert "oeis" in _channels_for_prompt("Catalan recurrence integer sequence")
    assert "pubchem" in _channels_for_prompt("Methane combustion chemical kinetics")

    # Offline client: only cached results, never touches the network.
    client = LiteratureSearchClient(cache_dir=Path("/nonexistent-cache"), offline=True)
    assert client.search("anything", 3) == []

    # Mocked-HTTP parsers for every new connector (fetch injection, offline).
    def fake_fetch(url: str, timeout: float) -> str:
        if "semanticscholar" in url:
            return json.dumps({"data": [{
                "title": "Stiff Robertson kinetics", "abstract": "stiff ODE",
                "tldr": {"text": "Robertson problem needs implicit solvers"},
                "year": 2020, "citationCount": 42,
                "externalIds": {"ArXiv": "2001.00001"}}]})
        if "wikidata.org" in url and "wbsearchentities" in url:
            return json.dumps({"search": [{"id": "Q42"}]})
        if "wikidata.org" in url and "wbgetentities" in url:
            import urllib.parse as _up
            qs = _up.parse_qs(_up.urlparse(url).query)
            ids = (qs.get("ids") or [""])[0]
            if "Q42" in ids:
                return json.dumps({"entities": {"Q42": {
                    "labels": {"en": {"value": "Test Constant"}},
                    "claims": {"P1": [{"mainsnak": {"datavalue": {"type": "quantity",
                        "value": {"amount": "+1500000",
                                  "unit": "http://x/Q11573"}}}}]}}}})
            # unit/property label lookups: return a label per requested id
            ents = {i: {"labels": {"en": {"value":
                "metre" if i == "Q11573" else "test constant"}}}
                    for i in ids.split("|") if i}
            return json.dumps({"entities": ents})
        if "oeis.org" in url:
            return json.dumps({"results": [{
                "number": "45", "name": "Fibonacci numbers",
                "data": "0,1,1,2,3,5,8,13,21,34",
                "formula": "a(n) = a(n-1) + a(n-2)"}]})
        if "pubchem" in url:
            return json.dumps({"PropertyTable": {"Properties": [{
                "CID": 297, "MolecularWeight": 44.0095, "ExactMass": 43.9898,
                "XLogP": -0.474, "TPSA": 34.1, "Complexity": 3.5,
                "IUPACName": "methane"}]}})
        return "{}"

    client = LiteratureSearchClient(offline=False, fetch=fake_fetch)

    # Semantic Scholar: tldr + citation count.
    ss = client._search_semantic_scholar("stiff robertson kinetics", 3)
    assert ss and ss[0].badge == "SemanticScholar"
    assert ss[0].tldr.startswith("Robertson") and ss[0].citation_count == 42

    # Wikidata: quantity claims -> constants dict.
    wd = client._search_wikidata_constants("test constant", 3)
    assert wd and "test_constant" in wd[0].constants
    assert wd[0].constants["test_constant"]["value"] == 1500000.0

    # OEIS: sequence id, initial terms benchmark, formula.
    oeis = client._search_oeis("fibonacci", 3)
    assert oeis and oeis[0].badge == "OEIS"
    assert oeis[0].benchmarks[0]["expected_value"][:5] == [0, 1, 1, 2, 3]
    assert "a(n-1)" in oeis[0].constants.get("formula", "")

    # PubChem: molecular properties.
    pc = client._search_pubchem("methane combustion", 3)
    assert pc and pc[0].badge == "PubChem"
    assert abs(pc[0].constants["MolecularWeight"]["value"] - 44.0095) < 1e-6

    # Wikipedia wikitext math extraction through the connector path.
    def fake_fetch_wiki(url: str, timeout: float) -> str:
        if "opensearch" in url:
            return json.dumps(["Fibonacci", ["Fibonacci"], ["x"],
                               ["https://en.wikipedia.org/wiki/Fibonacci"]])
        if "page/summary" in url:
            return json.dumps({"title": "Fibonacci", "extract": "Fibonacci sequence",
                               "content_urls": {"desktop": {"page": "https://x"}}})
        if "prop=revisions" in url:
            page = {"revisions": [{"slots": {"main": {"content":
                "The Fibonacci numbers: <math>F_n = F_{n-1} + F_{n-2}</math> "
                "| infobox_value = 1.6180339887"}}}]}
            return json.dumps({"query": {"pages": {"1": page}}})
        return "{}"

    wiki_client = LiteratureSearchClient(offline=False, fetch=fake_fetch_wiki)
    wiki = wiki_client._search_wikipedia("fibonacci", 2)
    assert wiki and any("F_n" in eq or "F_{n" in eq for eq in wiki[0].equations)
    assert wiki[0].badge == "Wikipedia"


# ---------------------------------------------------------------------------
# Test 4: deterministic backend & stepper recommender
# ---------------------------------------------------------------------------

def test_execution_strategy_recommender():
    ham = recommend_execution_strategy(
        "Simulate a Hamiltonian system with energy conservation for a Kepler orbit")
    assert ham["target_backend"] == "cpp" and ham["stepper"] == "symplectic_verlet"

    discrete = recommend_execution_strategy(
        "Compute the Fibonacci recurrence as an integer sequence")
    assert discrete["target_backend"] == "rust_hin"

    multi = recommend_execution_strategy(
        "Compare two competing closures for the entry heating model")
    assert multi["target_backend"] == "multi_variant"

    stiff = recommend_execution_strategy(
        "Solve the stiff Robertson chemical kinetics ODE")
    assert stiff["target_backend"] == "cpp" and stiff["stepper"] == "semi_implicit_euler"

    # Evidence-domain routing: discrete-only evidence -> rust_hin.
    evidence = QuantitativeInsightExtractor(use_registry=True).extract(
        "Fibonacci recurrence integer sequence")
    strat = recommend_execution_strategy("", evidence)
    assert strat["target_backend"] in ("rust_hin", "multi_variant", "cpp")


# ---------------------------------------------------------------------------
# Test 5: closed-loop diagnostic re-search & unit scaling diagnostic
# ---------------------------------------------------------------------------

def _bp_with_contract(assertion: str = "peak_heat_flux_mw_m2 <= 10.0") -> ResearchBlueprint:
    return ResearchBlueprint(
        blueprint_id="mars_entry_diag", title="Mars entry heat flux diagnostic",
        falsifiable_hypothesis="Peak stagnation heat flux stays below 10 MW/m^2",
        search_queries=["Mars Sutton-Graves heat flux"],
        contracts=[VerificationContract(
            contract_id="c1", target_backend="cpp",
            invariants=[InvariantConstraint(variable_name="peak_heat_flux",
                                            assertion=assertion, tolerance=0.0)],
            initial_state=[1.0, 2.0, 3.0, 4.0], dt=0.01, steps=10)],
        code_template="def simulation_step(buf, dt):\n    pass\n",
        variable_map={"peak_heat_flux_mw_m2": 3})


def test_closed_loop_diagnostic_search():
    bp = _bp_with_contract()
    # FALSIFIED: observed heat flux 2.84e6 (W/m^2) vs bound 10.0 (MW/m^2).
    failure = {
        "status": "FALSIFIED",
        "telemetry": {"obs_peak_heat_flux_mw_m2": 2.84e6},
        "counterexample": ("`peak_heat_flux_mw_m2 <= 10.0` (peak_heat_flux) "
                           "observed {'obs_peak_heat_flux_mw_m2': 2840000.0}"),
    }
    diag = unit_scaling_diagnostic(bp, failure["telemetry"], failure["counterexample"])
    assert diag is not None, "unit scaling diagnostic missed the 1e6 factor"
    assert math.isclose(diag["factor"], 1e6, rel_tol=1e-6)
    assert diag["direction"] == "divide"
    assert diag["si_unit"] == "W/m^2"

    queries = formulate_diagnostic_search_query(bp, failure)
    assert queries, "no diagnostic queries formulated"
    assert any("unit" in q.lower() or "scaling" in q.lower() or "conversion" in q.lower()
               for q in queries) or any("breakdown" in q or "regime" in q for q in queries)

    # DIVERGED (NaN): triggers stability/CFL diagnostic query.
    diverged = {"status": "DIVERGED", "telemetry": {"finite": 0.0},
                "counterexample": "non-finite state at step 42 (NaN)"}
    dqueries = formulate_diagnostic_search_query(bp, diverged)
    assert any("stability" in q.lower() or "cfl" in q.lower() for q in dqueries), dqueries


# ---------------------------------------------------------------------------
# Test 6: end-to-end evidence-grounded vault commit
# ---------------------------------------------------------------------------

_STAGE1_CODE = '''
import math

def simulation_step(buf, dt):
    data = buf.data
    data[0] = data[0] * math.exp(-dt / 11100.0)
'''

_STAGE2_CODE = '''
import math

def simulation_step(buf, dt):
    layer_1_step(buf, dt)
    data = buf.data
    data[1] = data[1] + dt * 9.81
'''


class _FakeSearch:
    def search(self, q: str, limit: int = 3, *args, **kwargs):
        return [LiteratureResult(
            source="arxiv", title=f"Paper: {q}", summary="Sutton-Graves k=1.83e-4",
            url="https://arxiv.org/abs/2401.00001", year=2024, badge="arXiv")]


async def _synth_stage2(**kw):
    return {"code": _STAGE2_CODE, "contracts": [],
            "rationale": "accrete stage 2"}


def _blueprint() -> ResearchBlueprint:
    evidence = QuantitativeInsightExtractor().extract(
        "Mars entry heat shield trajectory deceleration",
        [LiteratureResult(source="arxiv", title="Mars entry", summary="g=3.71 m/s^2",
                          url="https://x", badge="arXiv")])
    return ResearchBlueprint(
        blueprint_id="evidence_vault_run", title="Evidence grounded vault run",
        falsifiable_hypothesis="Density decays exponentially with 11.1 km scale height",
        search_queries=["Mars exponential atmosphere scale height"],
        contracts=[
            VerificationContract(
                contract_id="c1_atmosphere", target_backend="cpp",
                invariants=[InvariantConstraint(
                    variable_name="density_decay",
                    assertion="state[0] < initial_state[0] + 1e-9", tolerance=1e-6)],
                initial_state=[0.020, 1000.0], dt=0.05, steps=40),
            VerificationContract(
                contract_id="c2_energy", target_backend="cpp",
                invariants=[InvariantConstraint(
                    variable_name="energy_growth",
                    assertion="state[1] > initial_state[1]", tolerance=0.0)],
                initial_state=[0.020, 1000.0], dt=0.05, steps=40),
        ],
        code_template=_STAGE1_CODE,
        target_research_stages=2,
        stage_objectives=["exponential atmosphere", "energy accretion"],
        variable_map={"density": 0, "velocity": 1},
        structured_evidence=evidence,
        extracted_constants={n: float(p.si_value) for n, p in evidence.parameters.items()})


@pytest.mark.asyncio
async def test_end_to_end_evidence_grounded_vault_commit(tmp_path):
    events: List[tuple] = []

    async def cb(channel: str, payload: Dict[str, Any]) -> None:
        events.append((channel, payload))

    machine = ResearchStateMachine(tmp_path, progress_cb=cb,
                                   search_client=_FakeSearch(),
                                   stage_synthesizer=_synth_stage2)
    state = await machine.run(_blueprint())
    assert state.status == "VERIFIED", [
        (r["contract_id"], r["status"], r.get("failure_counterexample"))
        for _, p in events if p.get("event") == "result" for r in [p]]

    run_dir = tmp_path / ".invar" / "research_vault" / "evidence_vault_run"
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    # structured_evidence persisted in metadata.json.
    se = meta.get("structured_evidence") or {}
    assert se.get("domains"), "structured_evidence.domains missing from metadata"
    assert se.get("parameters"), "structured_evidence.parameters missing"
    assert se.get("recommended_backend") in ("rust_hin", "cpp", "multi_variant")
    assert se.get("equations") is not None
    # SI constants exposed cleanly.
    assert meta.get("extracted_constants"), "extracted_constants missing from metadata"
    for name, value in list(meta["extracted_constants"].items())[:4]:
        assert isinstance(value, float), (name, value)
    # report.md carries the dedicated section.
    assert "Extracted Literature Equations & SI Constants" in report
    assert "SI value" in report or "rhs_python_hint" in report or "Python hint" in report

    # build_research_vault_context exposes the SI constants to the Builder.
    ctx = build_research_vault_context(tmp_path, referenced_ids=["evidence_vault_run"])
    assert "structured_evidence_si_constants" in ctx
    assert "evidence_vault_run" in ctx

    # Blueprint grounding validator is clean for this grounded blueprint.
    warnings = validate_blueprint_grounding(state.blueprint)
    assert isinstance(warnings, list)


def test_planner_prompts_request_concise_2_to_4_term_queries():
    from invar_agent.research.planner import PLANNER_PROMPT
    import invar_agent.research.planner as planner_mod
    prompts = [PLANNER_PROMPT]
    src = Path(planner_mod.__file__).read_text()
    # Both planner prompt variants demand concise keyword queries, never sentences.
    assert "2-4 terms" in src
    assert "NEVER full sentences" in src or "no full sentences" in src
    assert "vs" in src and "versus" in src  # filler words explicitly banned
