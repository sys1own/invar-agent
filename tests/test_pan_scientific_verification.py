"""Comprehensive unit tests for the Universal Pan-Scientific Hypothesis Verification Engine."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from invar_agent.algorithms.prng import (
    invar_rand_binomial,
    invar_rand_normal,
    invar_rand_poisson,
    invar_rand_u01,
    invar_splitmix64_stateless,
)
from invar_agent.healing.deterministic_ast import (
    _invar_project_k_simplex,
    sanitize_intrinsics,
)
from invar_agent.research.extractor import (
    MACRO_SCIENTIFIC_DOMAINS,
    ReferenceRegistry,
    classify_macro_scientific_domains,
    classify_primary_macro_domain,
    lookup_reference_registry,
    normalize_unit,
)
from invar_agent.research.literature import (
    LiteratureSearchClient,
    _channels_for_prompt,
    async_search_europe_pmc,
    async_search_pubmed,
    classify_research_domain,
)
from invar_agent.research.state_machine import (
    _eval_invariant,
    execution_archetype_for_blueprint,
    v9_transpiler_violations,
)
from invar_agent.runtime.cabi_bridge import (
    ARCHETYPE_DYNAMICAL_STEPPER,
    ARCHETYPE_EQUILIBRIUM_FIXED_POINT,
    ARCHETYPE_MONTE_CARLO_ENSEMBLE,
    ARCHETYPE_PARAMETER_BIFURCATION_SWEEP,
    EXECUTION_ARCHETYPES,
    DynamicCABIBridge,
)
from invar_agent.schemas.research import (
    InvariantConstraint,
    ResearchBlueprint,
    VerificationContract,
)
from invar_agent.transpiler.emitters.cpp_emitter import CppEmitter
from invar_agent.transpiler.frontend import python_source_to_uast
from invar_agent.uast.totalized_math import (
    betainc_reg,
    gammainc_reg,
    hill_eq,
    kl_term,
    logistic_sigmoid,
    logit,
    michaelis_menten,
    norm_cdf,
    norm_ppf,
    shannon_term,
    welford_cov_update,
    welford_m2_update,
    welford_mean_update,
)


def test_eight_macro_scientific_domain_classifiers() -> None:
    assert len(MACRO_SCIENTIFIC_DOMAINS) == 8
    cases = {
        "PHYSICAL_SCIENCES_AND_ENG": "Simulate Lindblad quantum decoherence and Hamiltonian symplectic orbit",
        "CHEMICAL_AND_MATERIALS": "Patankar stoichiometric reaction kinetics and solid-state battery SEI diffusion",
        "LIFE_SCIENCES_AND_BIOMEDICINE": "Pharmacokinetic PK/PD antiviral Hill inhibition with EC50 and clearance rate",
        "EPIDEMIOLOGY_AND_ECOLOGY": "5-compartment SEIR-V epidemic outbreak with basic reproduction number R0",
        "EARTH_CLIMATE_AND_ENVIRONMENT": "4-reservoir global carbon cycle with Revelle buffer factor and radiative forcing",
        "STATISTICS_AND_CAUSAL_INFERENCE": "Monte Carlo 2SLS instrumental variable causal estimator and Cramer-Rao lower bound",
        "ECONOMICS_AND_GAME_THEORY": "Evolutionary replicator dynamics on the 3-simplex with Nash equilibrium regret",
        "MATHEMATICS_CS_AND_INFO_THEORY": "Kullback-Leibler relative entropy divergence and Shannon mutual information",
    }
    for expected_domain, prompt in cases.items():
        detected = classify_macro_scientific_domains(prompt)
        assert expected_domain in detected, f"Expected {expected_domain} in {detected} for {prompt!r}"
        assert classify_primary_macro_domain(prompt) == expected_domain


def test_pan_scientific_unit_normalizer_and_reference_registry() -> None:
    # Clinical, pharmacological, and biochemical units
    val, unit = normalize_unit(1.0, "M")
    assert math.isclose(val, 1e3) and unit == "mol/m^3"
    val, unit = normalize_unit(5.0, "mM")
    assert math.isclose(val, 5.0) and unit == "mol/m^3"
    val, unit = normalize_unit(2.0, "uM")
    assert math.isclose(val, 2e-3) and unit == "mol/m^3"
    val, unit = normalize_unit(50.0, "nM")
    assert math.isclose(val, 50e-6) and unit == "mol/m^3"
    val, unit = normalize_unit(10.0, "pM")
    assert math.isclose(val, 10e-9) and unit == "mol/m^3"
    val, unit = normalize_unit(100.0, "mg/dL")
    assert math.isclose(val, 1.0) and unit == "kg/m^3"
    val, unit = normalize_unit(1.0, "Da")
    assert math.isclose(val, 1.66053906660e-27) and unit == "kg"
    val, unit = normalize_unit(1.0, "kcal/mol")
    assert math.isclose(val, 4184.0) and unit == "J/mol"

    # Biological, epidemiological, and genomic units (plus cm vs cM disambiguation)
    val_cm, unit_cm = normalize_unit(1.0, "cm")
    assert math.isclose(val_cm, 0.01) and unit_cm == "m"
    val_centimorgan, unit_centimorgan = normalize_unit(1.0, "cM")
    assert math.isclose(val_centimorgan, 0.01) and unit_centimorgan == "Morgan"
    val, unit = normalize_unit(1.0, "1/day")
    assert math.isclose(val, 1.0 / 86400.0) and unit == "1/s"
    val, unit = normalize_unit(1.0, "1/year")
    assert math.isclose(val, 1.0 / 31557600.0) and unit == "1/s"

    # Climate, environmental, and earth-system units
    val, unit = normalize_unit(400.0, "ppm")
    assert math.isclose(val, 400e-6)
    val, unit = normalize_unit(1.0, "GtC")
    assert math.isclose(val, 1e12) and unit == "kg"
    val, unit = normalize_unit(1.0, "PgC")
    assert math.isclose(val, 1e12) and unit == "kg"
    val, unit = normalize_unit(1.0, "Sv")
    assert math.isclose(val, 1e6) and unit == "m^3/s"

    # Statistical, economic, and information-theoretic units
    val, unit = normalize_unit(1.0, "bit")
    assert math.isclose(val, 1.0) and unit == "bit"
    val_nat, unit_nat = normalize_unit(1.0, "nat")
    assert math.isclose(val_nat, 1.0 / math.log(2.0)) and unit_nat == "bit"
    val, unit = normalize_unit(25.0, "%")
    assert math.isclose(val, 0.25)
    val, unit = normalize_unit(50.0, "bps")
    assert math.isclose(val, 0.005)

    # All 7 tightly-gated domain packs exist in ReferenceRegistry
    for pack in (
        "epidemiology_population",
        "pharmacology_biochemistry",
        "genetics_evolution",
        "climate_earth_systems",
        "statistics_causal_inference",
        "economics_game_theory",
        "math_info_theory",
    ):
        assert pack in ReferenceRegistry
    epi_hits = lookup_reference_registry("SEIR epidemic outbreak R0")
    assert any(p.name == "basic_reproduction_number" for p in epi_hits)
    clim_hits = lookup_reference_registry("global carbon cycle Revelle buffer factor preindustrial CO2")
    assert any("revelle_buffer_factor" in p.name for p in clim_hits)


def test_collision_free_splitmix64_and_statistical_intrinsics(tmp_path: Path) -> None:
    # Safeguard 1: step == channel must NOT collide across distinct k == c pairs
    v1 = invar_splitmix64_stateless(42, 1, 1)
    v2 = invar_splitmix64_stateless(42, 2, 2)
    v3 = invar_splitmix64_stateless(42, 3, 3)
    assert len({v1, v2, v3}) == 3

    for step in range(50):
        u = invar_rand_u01(42, step, step)
        assert 1e-15 <= u <= 1.0 - 1e-15
        z = invar_rand_normal(42, step, step)
        assert math.isfinite(z)

    # Verify inverse accuracy of norm_ppf(norm_cdf(x))
    for x in (-2.0, -1.0, -0.25, 0.0, 0.5, 1.5, 2.5):
        p = norm_cdf(x)
        x_rec = norm_ppf(p)
        assert abs(x - x_rec) < 1e-3
    assert math.isclose(logit(logistic_sigmoid(1.25)), 1.25, rel_tol=1e-9)
    assert math.isclose(hill_eq(2.0, 2.0, 2.0), 0.5, rel_tol=1e-9)
    assert math.isclose(michaelis_menten(4.0, 10.0, 4.0), 5.0, rel_tol=1e-9)
    assert abs(kl_term(0.4, 0.4)) < 1e-12
    assert shannon_term(0.0) == 0.0
    assert 0.0 <= gammainc_reg(2.0, 2.0) <= 1.0
    assert 0.0 <= betainc_reg(2.0, 2.0, 0.5) <= 1.0

    # Transpile and compile a kernel calling all pan-scientific intrinsics in C++20
    kernel_py = """import math

def simulation_step(buf, dt):
    u = invar_rand_u01(42, buf.steps, 1)
    z = invar_rand_normal(42, buf.steps, 2)
    pois = invar_rand_poisson(42, buf.steps, 4.5)
    bino = invar_rand_binomial(42, buf.steps, 10.0, 0.4)
    k = buf.steps + 1.0
    m_old = buf.data[0]
    m_new = welford_mean_update(m_old, z, k)
    m2_new = welford_m2_update(buf.data[1], m_old, m_new, z)
    cov_new = welford_cov_update(buf.data[2], m_old, z, m_new, u)
    p_norm = norm_cdf(0.0)
    z_q = norm_ppf(0.975)
    sig = logistic_sigmoid(logit(0.75))
    h_occ = hill_eq(2.0, 2.0, 1.5)
    mm_rate = michaelis_menten(3.0, 6.0, 3.0)
    kl_v = kl_term(0.5, 0.25)
    sh_v = shannon_term(0.5)
    g_v = gammainc_reg(2.0, 2.0)
    b_v = betainc_reg(2.0, 2.0, 0.5)
    buf.data[0] = m_new
    buf.data[1] = m2_new
    buf.data[2] = cov_new
    buf.data[3] = p_norm + h_occ
    buf.data[4] = z_q
    buf.data[5] = sig
    buf.data[6] = mm_rate
    buf.data[7] = kl_v + sh_v + g_v + b_v + 0.0 * (pois + bino)
"""
    san = sanitize_intrinsics(kernel_py)
    assert not v9_transpiler_violations(san.code)
    uast = python_source_to_uast(san.code)
    cpp_code = CppEmitter().emit_module(uast)
    bridge = DynamicCABIBridge(tmp_path / "cabi")
    res_cpp = bridge.run(cpp_code, [0.0] * 16, dt=0.01, steps=20, isolated=False)
    res_py = bridge.run_python(san.code, [0.0] * 16, dt=0.01, steps=20)
    assert res_cpp["finite"] and res_py["finite"]
    for i in range(8):
        assert math.isclose(res_cpp["state"][i], res_py["state"][i], rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(res_cpp["state"][3], 1.0, abs_tol=1e-6)
    assert math.isclose(res_cpp["state"][5], 0.75, abs_tol=1e-6)
    assert math.isclose(res_cpp["state"][6], 3.0, abs_tol=1e-6)


def test_random_ast_lowering_and_cpp20_compilation(tmp_path: Path) -> None:
    raw_code = """import math
import random

def simulation_step(buf, dt):
    u = random.random()
    v = random.uniform(-1.0, 1.0)
    g = random.gauss(0.0, 1.0)
    n = random.normalvariate(2.0, 0.5)
    e = random.expovariate(1.5)
    r = random.randint(1, 6)
    buf.data[0] = u
    buf.data[1] = v
    buf.data[2] = g + n + e + r
"""
    san = sanitize_intrinsics(raw_code)
    assert "import random" not in san.code
    assert "buf.data[62] = buf.data[62] + 1.0" in san.code
    assert "invar_rand_u01" in san.code
    assert "invar_rand_normal" in san.code
    assert not v9_transpiler_violations(san.code)

    uast = python_source_to_uast(san.code)
    cpp_code = CppEmitter().emit_module(uast)
    bridge = DynamicCABIBridge(tmp_path / "cabi_rand")
    res = bridge.run(cpp_code, [0.0] * 8, dt=0.01, steps=10, isolated=False)
    assert res["finite"]
    assert 1e-15 <= res["state"][0] <= 1.0 - 1e-15
    assert -1.0 <= res["state"][1] <= 1.0


@pytest.mark.parametrize("k_slots,target_sum", [
    (2, 1.0),
    (3, 1.0),
    (4, 41000.0),
    (5, 1.0),
    (6, 1000.0),
])
def test_arbitrary_k_simplex_projector(k_slots: int, target_sum: float) -> None:
    slot_lines = [f"    buf.data[{i}] = buf.data[{i}] * (1.0 + {0.03 * (i + 1)} * dt)" for i in range(k_slots)]
    raw_code = "import math\n\ndef simulation_step(buf, dt):\n" + "\n".join(slot_lines) + "\n"
    sum_expr = " + ".join(f"state[{i}]" for i in range(k_slots))
    invariants = [
        InvariantConstraint(
            variable_name=f"simplex_conservation_k{k_slots}",
            assertion=f"abs(({sum_expr}) - {target_sum!r}) <= 1e-9",
            tolerance=1e-9,
        )
    ] + [
        InvariantConstraint(
            variable_name=f"nonneg_{i}",
            assertion=f"state[{i}] >= 0.0",
            tolerance=0.0,
        )
        for i in range(k_slots)
    ]
    init = [target_sum / k_slots for _ in range(k_slots)] + [0.0] * 8
    contract = VerificationContract(
        contract_id=f"c1_simplex_k{k_slots}",
        target_backend="cpp",
        initial_state=init,
        dt=0.05,
        steps=25,
        invariants=invariants,
    )
    san = _invar_project_k_simplex(raw_code, contracts=[contract])
    ns: dict = {}
    exec(san.code, ns)  # noqa: S102

    class _DummyBuf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(data)
            self.dt = 0.05
            self.steps = 0

    buf = _DummyBuf(init)
    for _ in range(25):
        ns["simulation_step"](buf, 0.05)
        buf.steps += 1
    for inv in invariants:
        assert _eval_invariant(inv.assertion, buf.data, {})


def test_statistical_domain_conditioners_and_semantic_tags() -> None:
    raw_code = """import math

def simulation_step(buf, dt):
    buf.data[0] = 4.0
    buf.data[1] = 9.0
    buf.data[2] = 10.0
    buf.data[3] = 0.05
    buf.data[4] = -1.5
"""
    invariants = [
        InvariantConstraint(
            variable_name="cauchy_schwarz_cov",
            assertion="abs(state[2]) <= math.sqrt(state[0] * state[1])",
            tolerance=1e-9,
        ),
        InvariantConstraint(
            variable_name="cramer_rao_lb",
            assertion="state[3] >= 1.0 / state[0]",
            tolerance=1e-9,
        ),
        InvariantConstraint(
            variable_name="correlation_bounds",
            assertion="state[4] >= -1.0 and state[4] <= 1.0",
            tolerance=1e-9,
        ),
    ]
    contract = VerificationContract(
        contract_id="c1_stats",
        target_backend="cpp",
        initial_state=[4.0, 9.0, 0.0, 0.5, 0.0],
        dt=0.01,
        steps=5,
        invariants=invariants,
    )
    san = sanitize_intrinsics(raw_code, contracts=[contract])
    ns: dict = {}
    exec(san.code, ns)  # noqa: S102

    class _DummyBuf:
        def __init__(self, data):
            self.data = list(data)
            self.size = len(data)
            self.dt = 0.01
            self.steps = 0

    buf = _DummyBuf(contract.initial_state)
    ns["simulation_step"](buf, 0.01)
    for inv in invariants:
        assert _eval_invariant(inv.assertion, buf.data, {})

    # Native Rust PyHolographicHin semantic_tag check
    from invar_agent._native import PyHolographicHin
    quantities = [
        {
            "symbol": "p_prob",
            "si_dimension": [0, 0, 0, 0, 0, 0, 0],
            "interval": [0.0, 1.0],
            "is_state_variable": True,
            "dimension_unspecified": False,
            "semantic_tag": 1,
        },
        {
            "symbol": "u_util",
            "si_dimension": [0, 0, 0, 0, 0, 0, 0],
            "interval": [-10.0, 10.0],
            "is_state_variable": True,
            "dimension_unspecified": False,
            "semantic_tag": 5,
        },
    ]
    hin = PyHolographicHin(json.dumps(quantities))
    eqs = {
        "bad_sum": {
            "kind": "binary",
            "op": "+",
            "left": {"kind": "symbol", "name": "p_prob"},
            "right": {"kind": "symbol", "name": "u_util"},
        }
    }
    report = json.loads(hin.verify_equations(json.dumps(eqs)))
    assert any("Semantic Tag Mismatch" in d for d in report["diagnostics"])


def test_four_execution_archetypes_and_literature_connectors() -> None:
    assert len(EXECUTION_ARCHETYPES) == 4
    dummy_c = VerificationContract(
        contract_id="c1",
        target_backend="cpp",
        initial_state=[1.0],
        dt=0.01,
        steps=10,
        invariants=[InvariantConstraint(variable_name="x", assertion="state[0] >= 0.0", tolerance=0.0)],
    )
    bp_mc = ResearchBlueprint(
        blueprint_id="bp_mc",
        title="Monte Carlo 2SLS Instrumental Variable Benchmark",
        falsifiable_hypothesis="Verify 2SLS vs confounded OLS estimator and Cramer-Rao lower bound.",
        search_queries=["2SLS instrumental variable"],
        contracts=[dummy_c],
        code_template="def simulation_step(buf, dt): pass",
    )
    assert execution_archetype_for_blueprint(bp_mc) == ARCHETYPE_MONTE_CARLO_ENSEMBLE

    bp_eq = ResearchBlueprint(
        blueprint_id="bp_eq",
        title="Evolutionary Game Theory Replicator Fixed Point",
        falsifiable_hypothesis="Verify interior Nash equilibrium fixed point and KL Lyapunov function.",
        search_queries=["replicator dynamics Nash equilibrium"],
        contracts=[dummy_c],
        code_template="def simulation_step(buf, dt): pass",
    )
    assert execution_archetype_for_blueprint(bp_eq) == ARCHETYPE_EQUILIBRIUM_FIXED_POINT

    bp_sw = ResearchBlueprint(
        blueprint_id="bp_sw",
        title="Pharmacological Dose-Response Hill Curve Sweep",
        falsifiable_hypothesis="Verify parameter sweep across drug dose and Hill curve bifurcation.",
        search_queries=["Hill dose response curve"],
        contracts=[dummy_c],
        code_template="def simulation_step(buf, dt): pass",
    )
    assert execution_archetype_for_blueprint(bp_sw) == ARCHETYPE_PARAMETER_BIFURCATION_SWEEP

    bp_dyn = ResearchBlueprint(
        blueprint_id="bp_dyn",
        title="4-Reservoir Global Carbon Cycle ODE",
        falsifiable_hypothesis="Integrate coupled atmosphere-ocean-biosphere carbon reservoirs.",
        search_queries=["global carbon cycle"],
        contracts=[dummy_c],
        code_template="def simulation_step(buf, dt): pass",
    )
    assert execution_archetype_for_blueprint(bp_dyn) == ARCHETYPE_DYNAMICAL_STEPPER

    # Biomedical / epidemiological domain routing activates Europe PMC & PubMed
    channels = classify_research_domain("SEIR-V epidemic outbreak with antiviral Hill EC50 pharmacology")
    assert "biomed_epmc_pubmed" in channels
    sources = _channels_for_prompt("SEIR-V epidemic outbreak with antiviral Hill EC50 pharmacology")
    assert "europe_pmc" in sources and "pubmed" in sources

    # Mock fetch verification for Europe PMC and NCBI PubMed connectors
    def _mock_fetch(url: str, _timeout: float) -> str:
        if "europepmc" in url:
            return json.dumps({
                "resultList": {
                    "result": [
                        {
                            "title": "Antiviral Hill Pharmacodynamics in SEIR Models",
                            "abstractText": "Measured EC50 = 50 nM and R0 = 2.5 in viral kinetics.",
                            "pmid": "12345678",
                            "pubYear": "2024",
                            "citedByCount": 19,
                        }
                    ]
                }
            })
        if "esearch.fcgi" in url:
            return json.dumps({
                "esearchresult": {
                    "idlist": ["87654321"],
                    "webenv": "MCID_test",
                    "querykey": "1",
                }
            })
        if "esummary.fcgi" in url:
            return json.dumps({
                "result": {
                    "uids": ["87654321"],
                    "87654321": {
                        "title": "Population Pharmacokinetics and Compartmental Epidemiology",
                        "fulljournalname": "Journal of Infectious Diseases",
                        "pubdate": "2023 May",
                        "authors": [{"name": "Smith J"}],
                    },
                }
            })
        return "{}"

    epmc_hits = asyncio.run(async_search_europe_pmc("SEIR antiviral EC50", fetch=_mock_fetch))
    assert len(epmc_hits) == 1
    assert epmc_hits[0].source == "europe_pmc"
    assert epmc_hits[0].badge == "EuropePMC"

    pubmed_hits = asyncio.run(async_search_pubmed("SEIR pharmacokinetics", fetch=_mock_fetch))
    assert len(pubmed_hits) == 1
    assert pubmed_hits[0].source == "pubmed"
    assert pubmed_hits[0].badge == "PubMed"
