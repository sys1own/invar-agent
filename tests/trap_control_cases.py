"""Offline scientific inputs; no injected verification evidence or runners."""
import ast
import json

from invar_agent.research.epistemic import parse_run_certificate
from invar_agent.research.state_machine import ResearchStateMachine
from invar_agent.schemas.research import ResearchBlueprint
from invar_agent.research import meaningfulness as m


def blueprint(identifier, equations, bounds, initial, claims, *, controls=(), mechanism=(),
              mode="EXISTS", archetype="ODE_PDE", code=None, assumptions=(), steps=4, dt=.125,units=None,model_notes=""):
    names = list(bounds)
    if isinstance(claims,str): claims = [claims]
    prompt = (units or "All quantities are dimensionless.")+"\n"+model_notes+"\n"+"\n".join(
        [*(f"Model: {k} = {v}" for k,v in equations.items()),*(f"Claim: {c}" for c in claims)])
    obligations = []
    for i,claim in enumerate(claims):
        start = prompt.index("Claim: "+claim)+7
        obligations.append(dict(id=f"h{i}",expression=claim,obligation_class="HYPOTHESIS_TARGET",
            prompt_quote=claim,prompt_span=[start,start+len(claim)]))
    class Bind(ast.NodeTransformer):
        def visit_Name(self,node):
            return ast.parse(f"state[{names.index(node.id)}]",mode="eval").body if node.id in names else node
    invariants = [dict(variable_name=f"sealed_observable_{i}",assertion=ast.unparse(ast.fix_missing_locations(Bind().visit(ast.parse(c,mode="eval").body)))) for i,c in enumerate(claims)]
    if code is None:
        class KernelBind(ast.NodeTransformer):
            def visit_Name(self,node):
                return ast.parse(f"buf.data[{names.index(node.id)}]",mode="eval").body if node.id in names else node
        code = "def simulation_step(buf, dt):\n"+"".join(
            f"    buf.data[{names.index(k)}] = {ast.unparse(ast.fix_missing_locations(KernelBind().visit(ast.parse(v,mode='eval').body)))}\n"
            for k,v in equations.items() if k in names)
    slots = [dict(symbol=name,slot_idx=i,role="CONTROL_PARAM" if name in controls else "DERIVED_OBSERVABLE",si_dim=[0]*7) for i,name in enumerate(names)]
    return ResearchBlueprint(blueprint_id=identifier,title=identifier.replace("_"," "),research_prompt=prompt,
        falsifiable_hypothesis=" and ".join(claims),search_queries=[],code_template=code,
        variable_map=dict(zip(names,range(len(names)))),slot_roles={n:"read_only_parameter" if n in controls else "diagnostic" for n in names},
        ihcg_evidence=dict(archetype=archetype,domain=bounds,evidence=dict(slots=slots,
            equations={k:dict(expression=v,si_dim=[0]*7) for k,v in equations.items()},
            obligations=obligations,environmental_assumptions=list(assumptions),quantifier_mode=mode,mechanism_parameters=list(mechanism))),
        contracts=[dict(contract_id="scientific_model",target_backend="cpp",initial_state=initial,steps=steps,dt=dt,invariants=invariants)],
        max_feedback_rounds=1)


async def run_case(tmp_path,bp,expected):
    events = []
    async def record(channel,payload): events.append(payload)
    machine = ResearchStateMachine(tmp_path,workspace_root=tmp_path,progress_cb=record)
    state = await machine.run(bp)
    path = tmp_path/".invar"/"research_vault"/bp.blueprint_id/"meaningfulness_certificate.json"
    certificate = parse_run_certificate(path.read_text(encoding="utf-8"))
    assert len(certificate.blake3_receipt) == 64
    if isinstance(certificate,m.MeaningfulnessCertificate): assert m.verify_certificate_receipt(certificate)
    assert certificate == parse_run_certificate(state.meaningfulness_certificate)
    assert state.epistemic_verdict == expected,certificate.model_dump_json(indent=2)
    reason = getattr(certificate,"failure_reason",None)
    if hasattr(certificate,"contract_audit"):
        reason = certificate.falsification_certificate.mode if certificate.falsification_certificate else certificate.nontriviality_audit.failure_reason or certificate.contract_audit.classification
    print(f"\n{bp.blueprint_id} | {expected} | {state.epistemic_verdict} | {reason or '-'}")
    return state,certificate,events


def rk4_code(states,rhs,slots,outputs=None):
    class Bind(ast.NodeTransformer):
        def __init__(self,stage): self.stage = stage
        def visit_Name(self,node):
            if node.id not in states or self.stage == 1: return node
            factor = "dt" if self.stage == 4 else "dt/2"
            return ast.parse(f"({node.id}+({factor})*k{self.stage-1}_{node.id})",mode="eval").body
    lines = ["def simulation_step(buf,dt):",*[f"    {s} = buf.data[{i}]" for s,i in slots.items()]]
    for stage in range(1,5):
        for state in states:
            expression = ast.unparse(ast.fix_missing_locations(Bind(stage).visit(ast.parse(rhs[state],mode="eval").body)))
            lines.append(f"    k{stage}_{state} = {expression}")
    for state in states:
        lines.extend([f"    {state} = {state}+dt*(k1_{state}+2*k2_{state}+2*k3_{state}+k4_{state})/6",f"    buf.data[{slots[state]}] = {state}"])
    for name,expression in (outputs or {}).items():
        lines.extend([f"    {name} = {expression}",f"    buf.data[{slots[name]}] = {name}"])
    return "\n".join(lines)+"\n"


def sir_model(identifier="sir_model",*,false_universal=False,frozen=False):
    rhs = dict(S="-gamma*R0*S*I",I="gamma*R0*S*I-gamma*I",R="gamma*I")
    bounds = dict(R0=[1.5 if false_universal else 0,4],gamma=[1,1],S=[0,1],I=[0,1],R=[0,1],Ipeak=[0,1],Rfinal=[0,1])
    slots = dict(zip(bounds,range(len(bounds))))
    code = rk4_code(tuple(rhs),dict.fromkeys(rhs,"0.0") if frozen else rhs,slots,dict(Ipeak="I",Rfinal="R"))
    bp = blueprint(identifier,{**{f"d{s}":e for s,e in rhs.items()},"Ipeak":"I","Rfinal":"R"},bounds,
        [4,1,.99,.01,0,.01,0],"Ipeak <= 0.15" if false_universal else "Ipeak > 0.02",controls=("R0","gamma"),mechanism=("R0",),
        mode="FORALL" if false_universal else "EXISTS",code=code,steps=800,dt=.02,
        units="Time is measured in seconds; gamma has SI s^-1. Other quantities are dimensionless.",
        model_notes="Ipeak is max I(t) over the sealed 16-second trajectory, including quadratic peak localization. Its kernel write Ipeak=I is a per-step sampler, not its trajectory reduction.")
    evidence = bp.ihcg_evidence["evidence"]
    for slot in evidence["slots"]:
        if slot["symbol"] in rhs: slot["role"] = "DYNAMIC_STATE"
        if slot["symbol"] == "gamma": slot["si_dim"] = [0,0,-1,0,0,0,0]
        if slot["symbol"] == "Ipeak": slot["semantic_tag"] = m.TRAJECTORY_MAXIMUM
    for state in rhs:
        evidence["equations"]["d"+state].update(kind="ODE_RHS",state_symbol=state,si_dim=[0,0,-1,0,0,0,0])
    evidence["equations"]["Ipeak"].update(kind="TRAJECTORY_MAX",state_symbol="I")
    for state in rhs: bp.slot_roles[state] = "evolving_state"
    return bp


def oregonator_model(identifier,*,renamed=False,cubic=False):
    names = ("u1","u2","u3","epsilon","stoich","relax","clock") if renamed else ("x","y","z","s","q","w","rate")
    x,y,z,s,q,w,rate = names
    rhs = {x:f"{rate}*{s}*({y}+{x}*(1-{q}*{x}-{y}))"+(f"-0.01*{rate}*{x}**3" if cubic else ""),
        y:f"{rate}*({z}-{y}*(1+{x}))/{s}",z:f"{rate}*{w}*({x}-{z})"}
    if renamed: rhs = {z:f"({x}-{z})*{w}*{rate}",y:f"({z}-{y}-{y}*{x})*{rate}/{s}",x:f"({y}+{x}-{q}*{x}*{x}-{x}*{y})*{s}*{rate}"}
    bounds = dict(zip(names,([0,5],[0,5],[0,5],[1,1],[.002,.002],[0,1],[1,1])))
    slots = dict(zip(names,range(len(names))))
    equations = {"d"+state:expression for state,expression in rhs.items()}
    bp = blueprint(identifier,equations,bounds,[1.2,.6,.8,1,.002,.1,1],f"{z} > 0.01",controls=(s,q,w,rate),mechanism=(w,),
        code=rk4_code((x,y,z),rhs,slots),steps=16,dt=.005,
        units=f"Time is measured in seconds; {rate} has SI s^-1. Other quantities are dimensionless.")
    for slot in bp.ihcg_evidence["evidence"]["slots"]:
        if slot["symbol"] in (x,y,z):
            slot["role"] = "DYNAMIC_STATE"
            bp.slot_roles[slot["symbol"]] = "evolving_state"
        if slot["symbol"] == rate: slot["si_dim"] = [0,0,-1,0,0,0,0]
    for state in rhs: bp.ihcg_evidence["evidence"]["equations"]["d"+state].update(kind="ODE_RHS",state_symbol=state,si_dim=[0,0,-1,0,0,0,0])
    return bp


def allen_eggers_model(identifier,*,kilometers=False):
    h,H,v0,a,v,v2 = ("alt","height_scale","entry","loading","speed","energy") if kilometers else ("h","H","v0","a","v","v2")
    scale = 1000 if kilometers else 1
    bounds = dict(zip((h,H,v0,a,v,v2),([10000/scale,20000/scale],[7000/scale,7000/scale],[8000/scale,8000/scale],[0,2],[0,9000/scale],[0,81000000/scale**2])))
    cutoff = "cutoff" if kilometers else "v_threshold"
    bounds[cutoff] = [2000/scale,2000/scale]
    equations = {v:f"{v0}*exp(-{a}*exp(-{h}/{H}))",v2:f"{v}*{v}"}
    code = f"import math\ndef simulation_step(buf,dt):\n    buf.data[4] = buf.data[2]*math.exp(-buf.data[3]*math.exp(-buf.data[0]/buf.data[1]))\n    buf.data[5] = buf.data[4]*buf.data[4]\n"
    bp = blueprint(identifier,equations,bounds,[15000/scale,7000/scale,8000/scale,.5,0,0,2000/scale],f"{v} > {cutoff}",controls=(h,H,v0,a,cutoff),mechanism=(a,),code=code,
        units=f"{h},{H} are in {'km' if kilometers else 'm'}; {v0},{v},{cutoff} in {'km/s' if kilometers else 'm/s'}; {v2} in squared velocity units; {a} dimensionless.")
    for slot in bp.ihcg_evidence["evidence"]["slots"]:
        symbol = slot["symbol"]
        power = 2 if symbol == v2 else 1
        slot["si_dim"] = [0,power,-power if symbol in (v0,v,v2,cutoff) else 0,0,0,0,0] if symbol != a else [0]*7
        if kilometers and symbol != a: slot["unit_scale"] = [scale**power,1]
    bp.ihcg_evidence["evidence"]["equations"][v]["si_dim"] = [0,1,-1,0,0,0,0]
    bp.ihcg_evidence["evidence"]["equations"][v2]["si_dim"] = [0,2,-2,0,0,0,0]
    return bp
