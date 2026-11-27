"""Python boundary checks of the actual native proof inventory and its consumers."""

import cmath
import json
import math
from pathlib import Path
import re
import sys

import mpmath as mp
import pytest
from pydantic import ValidationError

from invar_agent import _native
from invar_agent.research import falsification as f
from invar_agent.research import meaningfulness as m
from invar_agent.research import uniqueness as u


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT/"invar_agent/_native/src/csel_egraph.rs").read_text(encoding="utf-8")
RULE_SOURCE = SOURCE.split("fn verification_rules()",1)[1].split("fn absolute_power_exponent",1)[0]
RULES = re.findall(r'rewrite!\("([^"\n]+)"; "([^"\n]+)" => "([^"\n]+)"',RULE_SOURCE)


def _tree(expression):
    tokens = iter(re.findall(r"\(|\)|[^\s()]+",expression))
    def read(token):
        if token != "(": return token
        result = []
        for token in tokens:
            if token == ")": return result
            result.append(read(token))
        raise ValueError("unterminated proof expression")
    return read(next(tokens))


def _expression(tree):
    if isinstance(tree,str): return tree.removeprefix("?")
    op,*children = tree
    args = [_expression(child) for child in children]
    if op in {"+","-","*"}: return f"({args[0]}{op}{args[1]})"
    if op == "neg": return f"(-{args[0]})"
    if op == "safe_sqrt": return f"sqrt(max({args[0]},0))"
    if op == "safe_log": return f"log(max({args[0]},1e-30))"
    if op == "safe_pow": return f"(abs({args[0]})**({args[1]}))"
    if op == "safe_tanh": return f"(2/(1+exp(-2*({args[0]})))-1)"
    if op == "logistic_sigmoid": return f"(1/(1+exp(-({args[0]}))))"
    if op == "kl_term": return f"max(0,({args[0]})*log(({args[0]})/({args[1]}))-({args[0]})+({args[1]}))"
    if op == "shannon_term":
        assert args == ["0"]
        return "0"
    if op == "michaelis_menten": return f"(({args[1]})*({args[0]})/(({args[2]})+({args[0]})))"
    return f"{op.removeprefix('safe_')}({','.join(args)})"


@pytest.mark.parametrize("name,lhs,rhs",RULES,ids=[rule[0] for rule in RULES])
def test_every_native_proof_axiom_at_1024_points_and_interval_boxes(name,lhs,rhs):
    left,right = _expression(_tree(lhs)),_expression(_tree(rhs))
    left_node,right_node = m._parse(left),m._parse(right)
    variables = sorted(m._variables(left_node)|m._variables(right_node))
    specialize = lambda text: text.replace("?r","0.5").replace("?","")
    proof = json.loads(_native.py_check_algebraic_equivalence(specialize(lhs),specialize(rhs)))
    assert proof["equivalent"],name
    left_code,right_code = compile(left,"<proof-lhs>","eval"),compile(right,"<proof-rhs>","eval")
    with mp.workdps(60):
        functions = dict(exp=mp.exp,log=mp.log,sqrt=mp.sqrt,sin=mp.sin,cos=mp.cos,abs=abs,min=min,max=max)
        for index in range(1,1025):
            point = {v:0.2+2.8*m._halton(index,(2,3,5,7,11,13)[j]) for j,v in enumerate(variables)}
            environment = {**functions,**{v:mp.mpf(value) for v,value in point.items()}}
            a,b = (mp.mpf(eval(code,{"__builtins__":{}},environment)) for code in (left_code,right_code))
            assert mp.almosteq(a,b,rel_eps=mp.mpf("1e-45"),abs_eps=mp.mpf("1e-45")),(name,point,a,b)
            box = {v:((value,value) if v == "r" else (value-1e-8,value+1e-8)) for v,value in point.items()}
            ai,bi = m._iv(left_node,box),m._iv(right_node,box)
            assert ai.lo <= a <= ai.hi and bi.lo <= b <= bi.hi,(name,point)
            assert max(ai.lo,bi.lo) <= min(ai.hi,bi.hi),(name,point)


def test_proof_inventory_is_complete_and_healing_only_inverses_stay_excluded():
    assert len(RULES) == RULE_SOURCE.count('rewrite!(') >= 26
    names = {name for name,_,_ in RULES}
    exclusions = SOURCE.split("pub const HEALING_ONLY_RULES",1)[1].split("];",1)[0]
    excluded = re.findall(r'\("([^"\n]+)", "([^"\n]+)"\)',exclusions)
    assert len(excluded) >= 13 and all(reason for _,reason in excluded)
    assert not names & {name for name,_ in excluded}
    assert {"logit-sigmoid","sigmoid-logit","ppf-cdf"} <= {name for name,_ in excluded}
    assert not json.loads(_native.py_check_algebraic_equivalence("(logit (logistic_sigmoid x))","x"))["equivalent"]
    print(f"\nNative proof inventory: {len(RULES)} rules x 1024 valid-domain points and interval boxes; {len(excluded)} healing-only rules excluded")


def _native_interval(op,a,b):
    expression = dict(kind="binary",op=op,left=dict(kind="symbol",name="a"),right=dict(kind="symbol",name="b"))
    return _native.evaluate_interval_f64(json.dumps(expression),dict(a=a,b=b))


@pytest.mark.parametrize("op,a,b",[
    ("+",0.1,0.2),("-",0.3,0.2),("*",math.ulp(0.0),0.5),
    ("/",sys.float_info.min,3.0),("*",sys.float_info.max,2.0),
    ("+",sys.float_info.max,sys.float_info.max),("-",-sys.float_info.max,sys.float_info.max),
    ("+",0.0,-0.0),("-",-0.0,0.0),("*",-0.0,3.0),
    ("+",sys.float_info.max,-sys.float_info.max),
])
def test_interval_f64_outward_rounding_at_ieee_edges(op,a,b):
    lo,hi = _native_interval(op,(a,a),(b,b))
    with mp.workdps(400):
        x,y = mp.mpf(a),mp.mpf(b)
        exact = {"+":lambda:x+y,"-":lambda:x-y,"*":lambda:x*y,"/":lambda:x/y}[op]()
        assert lo <= exact <= hi
    assert not math.isnan(lo) and not math.isnan(hi)
    if a == b == 0:
        assert lo < 0 < hi


def test_interval_infinities_singularities_and_invalid_inputs():
    for op,a,b in (("*",(0,0),(math.inf,math.inf)),("+",(-math.inf,-math.inf),(math.inf,math.inf)),
                   ("/",(1,2),(-0.0,0.0))):
        assert _native_interval(op,a,b) == (-math.inf,math.inf)
    assert _native_interval("+",(math.inf,math.inf),(1,1))[1] == math.inf
    assert _native_interval("-",(-math.inf,-math.inf),(1,1))[0] == -math.inf
    for invalid in ((math.nan,1),(2,1)):
        with pytest.raises(ValueError,match="invalid interval"):
            _native_interval("+",invalid,(0,0))


@pytest.mark.parametrize("expression",["x*x*x-2*x", "exp(x)","log(x)","sqrt(x)","sin(x)","cos(x)","exp(-x)*sin(x)"])
def test_native_symbolic_adjoint_matches_complex_step_and_central_fd(expression):
    tree = m._csel(m._parse(expression))
    adjoint = _native.PyAdjointHin(["x"],json.dumps({"x":tree}))
    code = compile(expression,"<adjoint-oracle>","eval")
    def evaluate(x):
        return eval(code,{"__builtins__":{}},{"x":x,**{name:getattr(cmath,name) for name in ("exp","log","sqrt","sin","cos")}})
    for index in range(1,1025):
        x = 0.2+2*m._halton(index,2)
        report = json.loads(adjoint.analyze(json.dumps({"x":x})))
        derivative = report["jacobian"][0][0]
        assert report["symbolic_jacobian"][0][0]
        assert derivative == pytest.approx(evaluate(x+1e-25j).imag/1e-25,rel=2e-12,abs=2e-12)
        h = 1e-5
        assert derivative == pytest.approx(((evaluate(x+h)-evaluate(x-h))/(2*h)).real,rel=2e-8,abs=2e-8)


def _graph(claim="x > 0",*,domain=None,mode="FORALL",environment=()):
    domain = domain or {"x":[-2,2],"k":[0,1]}
    evidence = dict(slots=[dict(slot_idx=i,symbol=s,si_dim=[0]*7,role="DYNAMIC_STATE") for i,s in enumerate(domain)],
        equations={"law":dict(expression="x == x",si_dim=[0]*7)},environmental_assumptions=list(environment),
        obligations=[dict(id="h",expression=claim,obligation_class="HYPOTHESIS_TARGET",prompt_quote=claim,prompt_span=[0,len(claim)])],quantifier_mode=mode)
    return m.IhcgCompiler().compile_from_prompt(claim,evidence,domain,"ODE_PDE")


def _fidelity():
    return m.FidelityAudit(passed=True,algebraic_equiv_passed=True,si_and_semantic_tag_passed=True,
        convergence_passed=True,physical_checks_passed=True,convergence_reliability_ratio=0.5,epsilon_disc=1e-8,epsilon_total=1e-8)


def test_krawczyk_fd_containment_detects_mislowered_executable():
    encoded = json.dumps(m._csel(m._parse("x*x-2")))
    report = json.loads(_native.py_certify_krawczyk_fixed_point([encoded],[math.sqrt(2)],0.05))
    assert report["certified"]
    seen = []
    def executable(point):
        seen.append(tuple(point))
        return [point[0]*point[0]-2]
    audit = m.cross_check_krawczyk_jacobian(executable,[math.sqrt(2)],report,[[0.0]],executable_rounding_bound=1e-15)
    assert audit.passed and audit.samples_checked == 3 and len(set(seen)) == 6
    assert min(x[0] for x in seen) < report["krawczyk_box"][0]["lo"]
    assert max(x[0] for x in seen) > report["krawczyk_box"][0]["hi"]
    bad = m.cross_check_krawczyk_jacobian(lambda z:[2*(z[0]*z[0]-2)],[math.sqrt(2)],report,[[0.0]],executable_rounding_bound=1e-15)
    assert not bad.passed and bad.failure_reason == "JACOBIAN_ENCLOSURE_MISMATCH"
    with pytest.raises(ValueError):
        m.cross_check_krawczyk_jacobian(executable,[],report,[],executable_rounding_bound=0.0)


def test_unreliable_gci_cannot_falsify_or_halt_healing():
    a = m.roache_gci(1+0.01**4,1+0.02**4,1+0.04**4)
    b = m.roache_gci(1.1+0.01**2,1.1+0.02**2,1.1+0.04**2,nominal_order=2)
    failed = m.dual_scheme_fidelity(a,b,algebraic_equiv_passed=True,si_and_semantic_tag_passed=True,physical_checks_passed=True)
    assert not failed.passed and failed.convergence_reliability_ratio > 1
    graph = _graph()
    for refined,action in ((False,"REFINE_PRECISION_ONCE"),(True,"TERMINATE_INCONCLUSIVE")):
        decision = m.route_post_execution(graph,m.ContractAudit(entailment_locked=True,classification="INFEASIBLE_UNDER_MODEL",z3_logic_tier="QF_NRA"),
            m.PurityAudit(static_purity_passed=True,domain_tag="tcb"),failed,m.IdentifiabilityAudit(coupling_passed=True),
            m.HypothesisEvalResult(signed_margin=-1.0,epsilon_total=failed.epsilon_total,precision_refined=refined),0,1)
        assert decision.action == action and not decision.halt_healing
    for certificate in (f.run_mode1_deductive_falsification(graph,fidelity=failed),
        f.run_mode2a_counterexample_falsification(graph,{"x":-1.0,"k":0.5},f.WitnessEvaluation(signed_margin=-1.0,epsilon_total=failed.epsilon_total),failed)):
        assert not certificate.passed
        assert certificate.epistemic_verdict == m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
        assert certificate.failure_reason == "CONVERGENCE_ESTIMATOR_UNRELIABLE"


def test_unique_cds_does_not_rescue_environmentally_trivial_claim():
    graph = _graph("x > 0",environment=("x >= 0.5",))
    cds = u.CanonicalSignatureEngine().compute_cds(graph,{"equations":{"law":"x == x"}},m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED)
    novel = u.check_intra_vault_uniqueness(cds,graph,m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED,[],epsilon_disc=1e-8,epsilon_total=1e-8)
    assert novel.passed and novel.vault_class == "NOVEL_FIRST_IN_VAULT"
    nontrivial = m.verify_claim_nontriviality(graph,["x > 0"])
    assert not nontrivial.passed and nontrivial.failure_reason == "TRIVIAL_DEDUCIBLE_WITHOUT_MODEL"
    assert nontrivial.contract_audit.queries["Q3"].status == "UNSAT"
    certificate = m.build_meaningfulness_certificate(run_id="unique_but_trivial",verdict=m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED,
        ihcg=graph,contract_audit=m.ContractAudit(entailment_locked=True,classification="TAUTOLOGICAL",z3_logic_tier="QF_LRA"),
        nontriviality_audit=nontrivial,purity_audit=m.PurityAudit(static_purity_passed=True,domain_tag="tcb"),fidelity_audit=_fidelity(),
        identifiability_audit=m.IdentifiabilityAudit(coupling_passed=True),uast_digest="d"*64,
        uniqueness_audit=m.UniquenessAudit(passed=True,cds_model_hash=cds.model_hash,cds_claim_hash=cds.claim_hash,
            cds_full_hash=cds.full_hash,vault_class=novel.vault_class,reference_nontrivial=True))
    assert certificate.epistemic_verdict == m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
    assert m.verify_certificate_receipt(certificate)
    with pytest.raises(ValidationError,match="eight"):
        m.MeaningfulnessCertificate.model_validate({**certificate.model_dump(),"epistemic_verdict":m.EpistemicVerdict.VERIFIED_DISCOVERY})


def test_five_dimensional_mode2b_budget_exhaustion_coverage_telemetry():
    graph = _graph("x+x1+x2+x3+x4 >= 0.1",domain={name:[-1,1] for name in ("x","x1","x2","x3","x4")},mode="EXISTS")
    certificate = f.run_mode2b_exhaustive_refutation(graph,f.MarginSpec(),_fidelity())
    assert certificate.effective_max_boxes == 8192 and certificate.budget_exhausted
    assert 0 <= certificate.certified_refuted_volume_fraction < 1
    assert certificate.certified_subbox_count > 0 and 0 < certificate.boxes_evaluated <= 8192
    assert certificate.max_unresolved_upper_bound is not None
    assert math.isfinite(certificate.max_unresolved_upper_bound)
    assert not certificate.passed and certificate.epistemic_verdict == m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED
