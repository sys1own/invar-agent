"""Justified falsification, never inferred from failed search or solver UNKNOWN.

Black-box Lipschitz adapters must supply whole-cell gradient enclosures, not a
sampled-corner maximum asserted to be a bound. All executable/oracle calls remain
inside the runner TCB and must enforce their own worker deadlines.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

import z3

from invar_agent import _native
from . import meaningfulness as m
from .meaningfulness import BoundaryCertificate, EpistemicVerdict, FalsificationCertificate, FidelityAudit, IhcgGraph


BoundaryDiscoveryCertificate = BoundaryCertificate


class WitnessEvaluation(m.HypothesisEvalResult):
    margin_expression: str | None = None


@dataclass(frozen=True)
class MarginSpec:
    expression: str | None = None
    evaluator: Callable[[Mapping[str, float]], tuple[float, float]] | None = None
    gradient_enclosure: Callable[[m.Bounds], Sequence[tuple[float, float]]] | None = None
    ihcg_digest: str | None = None


class _Unresolved(Exception):
    pass


def _gate(ihcg: IhcgGraph, fidelity: FidelityAudit) -> str | None:
    ihcg.assert_sealed()
    fidelity = FidelityAudit.model_validate_json(fidelity.model_dump_json())
    if not fidelity.passed: return fidelity.failure_reason or "FIDELITY_UNCERTIFIED"
    if ihcg.archetype in {"ODE_PDE", "PARAM_SWEEP"} and fidelity.convergence_reliability_ratio is None:
        return "CONVERGENCE_RELIABILITY_UNVERIFIED"
    if (fidelity.convergence_reliability_ratio or 0) > 1: return "CONVERGENCE_ESTIMATOR_UNRELIABLE"
    if ihcg.archetype == "MONTE_CARLO" and fidelity.statistical_audit is None: return "STATISTICAL_RELIABILITY_UNVERIFIED"
    if fidelity.statistical_audit is not None and not fidelity.statistical_audit.passed: return "STATISTICAL_ESTIMATOR_UNRELIABLE"
    return None


def _certificate(mode, ihcg, fidelity, reason=None, **evidence):
    return FalsificationCertificate(mode=mode, ihcg_digest=ihcg.h_ihcg, epsilon_total=fidelity.epsilon_total,
        fidelity_passed=fidelity.passed, reliability_passed=_gate(ihcg, fidelity) is None,
        epistemic_verdict=EpistemicVerdict.INCONCLUSIVE_UNVERIFIED if reason else EpistemicVerdict.VERIFIED_FALSIFICATION,
        failure_reason=reason, **evidence)


def _is_formula(node):
    return isinstance(node, (ast.Compare, ast.BoolOp)) or isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) or isinstance(node, ast.Constant) and isinstance(node.value, bool) or isinstance(node, ast.Call) and node.func.id == "implies"


def _model_constraints(ihcg):
    if ihcg.archetype == "MONTE_CARLO" and any(e.kind == "MC_SAMPLE" for e in ihcg.equations.values()):
        from .stochastic_equilibrium import mc_mean_constraints
        return mc_mean_constraints(ihcg)
    names = {s.symbol for s in ihcg.slots}
    result = {}
    for name, equation in ihcg.equations.items():
        if equation.kind in {"ODE_RHS","TRAJECTORY_MAX","TRAJECTORY_MIN"}: continue
        if equation.kind in {"FIXED_POINT_RESIDUAL", "FIXED_POINT_MAP"}:
            target = "0" if equation.kind == "FIXED_POINT_RESIDUAL" else equation.state_symbol
            result[name+"_root"] = f"({target}) == ({equation.expression})"
            if name == equation.state_symbol: continue
        node = m._parse(equation.expression)
        result[name] = equation.expression if _is_formula(node) else f"({name if name in names else '0'}) == ({equation.expression})"
    return result


class _ExactTranslator(m._Translator):
    def term(self, node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and m._fraction(node.right) is None:
            denominator = m._iv(node.right, self.box)
            if denominator.lo <= 0 <= denominator.hi: raise m._DomainError("symbolic denominator is not domain-proven nonzero")
            return self.term(node.left)/self.term(node.right)
        return super().term(node)


def _external_check(smt2: str, requested: bool) -> str:
    if not requested: return "NOT_REQUESTED"
    if importlib.util.find_spec("cvc5") is None or not shutil.which("cvc5") or not shutil.which("carcara"):
        return "UNAVAILABLE"
    try:
        with tempfile.TemporaryDirectory(prefix="invar-alethe-") as directory:
            source, proof = Path(directory)/"core.smt2", Path(directory)/"core.alethe"
            source.write_text(smt2+"\n(get-proof)\n", encoding="utf-8")
            result = subprocess.run(["cvc5", "--lang=smt2", "--produce-proofs", "--proof-format-mode=alethe", str(source)], capture_output=True, text=True, timeout=2)
            if result.returncode or not result.stdout.startswith("unsat\n"): return "FAILED"
            proof.write_text(result.stdout.split("\n", 1)[1], encoding="utf-8")
            checked = subprocess.run(["carcara", "check", str(proof), "--premise-file", str(source)], capture_output=True, timeout=2)
            return "VERIFIED" if checked.returncode == 0 else "FAILED"
    except (OSError, subprocess.TimeoutExpired):
        return "FAILED"


def run_mode1_deductive_falsification(ihcg: IhcgGraph, model_eqs: Mapping[str, str] | None = None,
                                     run_carcara: bool = False, *, fidelity: FidelityAudit) -> FalsificationCertificate:
    mode = "MODE1_DEDUCTIVE_MUS"
    if reason := _gate(ihcg, fidelity): return _certificate(mode, ihcg, fidelity, reason)
    locked = _model_constraints(ihcg)
    if not locked: return _certificate(mode, ihcg, fidelity, "EMPTY_MODEL")
    if model_eqs is not None and ({k: ast.dump(m._parse(v)) for k, v in model_eqs.items()} != {k: ast.dump(m._parse(v)) for k, v in locked.items()}):
        raise ValueError("deductive model must match the sealed IHCG")
    deadline, calls, cache = time.monotonic()+1.5, 0, {}
    from ..runtime.isolation import run_on_z3_thread
    return run_on_z3_thread(_run_mode1_locked, ihcg, locked, mode, fidelity, deadline, run_carcara)


def _run_mode1_locked(ihcg, locked, mode, fidelity, deadline, run_carcara):
    calls, cache = 0, {}
    tr = _ExactTranslator(ihcg.omega, False)
    formulas, sources = {}, {}
    try:
        for name, expression in sorted(locked.items()): sources[f"model:{name}"] = expression
        for i, expression in enumerate(ihcg.environmental_assumptions): sources[f"env:{i}"] = expression
        for name, (lo, hi) in ihcg.omega.items():
            sources[f"domain:{name}:lo"] = f"{name} >= {lo!r}"
            sources[f"domain:{name}:hi"] = f"{name} <= {hi!r}"
        sources["h_user"] = m._and([o.expression for o in ihcg.obligations if o.obligation_class == "HYPOTHESIS_TARGET"])
        formulas = {name: tr.formula(m._parse(text)) for name, text in sources.items()}
    except (ValueError, z3.Z3Exception):
        return _certificate(mode, ihcg, fidelity, "SYMBOLIC_ENCODING_UNSUPPORTED")
    def check(names, tracked=False):
        nonlocal calls
        key = frozenset(names)
        if not tracked and key in cache: return cache[key], None
        if calls >= 64 or time.monotonic() >= deadline: raise _Unresolved("BUDGET_EXHAUSTED")
        calls += 1; solver = z3.Solver()
        solver.set(timeout=max(1, min(250, int((deadline-time.monotonic())*1000))), rlimit=5_000_000)
        solver.set("smt.core.minimize", True)
        labels = {}
        for i, name in enumerate(sorted(names)):
            if tracked:
                label = z3.Bool(f"$invar_track_{i}"); labels[str(label)] = name; solver.assert_and_track(formulas[name], label)
            else: solver.add(formulas[name])
        result = solver.check()
        if result == z3.unknown: raise _Unresolved("SOLVER_UNKNOWN")
        cache[key] = result
        return result, [labels[str(label)] for label in solver.unsat_core()] if tracked and result == z3.unsat else None
    consistency = "UNKNOWN"
    try:
        base = [name for name in formulas if name != "h_user"]
        result, _ = check(base); consistency = "SAT" if result == z3.sat else "UNSAT"
        if result == z3.unsat: return _certificate(mode, ihcg, fidelity, "MODEL_ENVIRONMENT_INCONSISTENT", model_consistency_status=consistency, solver_calls=calls)
        result, core = check(list(formulas), True)
        if result != z3.unsat: return _certificate(mode, ihcg, fidelity, "HYPOTHESIS_NOT_DEDUCTIVELY_REFUTED", model_consistency_status=consistency, solver_calls=calls)
        def quickxplain(background, candidates):
            if not candidates or check(background)[0] == z3.unsat: return []
            if len(candidates) == 1: return candidates
            mid = len(candidates)//2; left, right = candidates[:mid], candidates[mid:]
            first = quickxplain(background+right, left)
            return first+quickxplain(background+first, right)
        core = sorted(quickxplain([], sorted(core)))
        minimal = check(core)[0] == z3.unsat and all(check(core[:i]+core[i+1:])[0] == z3.sat for i in range(len(core)))
        if not minimal or "h_user" not in core or not any(k.startswith("model:") for k in core):
            return _certificate(mode, ihcg, fidelity, "UNJUSTIFIED_OR_NONMINIMAL_CORE", model_consistency_status=consistency, solver_calls=calls)
        replay = z3.Solver(); replay.add(*[formulas[k] for k in core]); smt2 = replay.to_smt2()
        external = _external_check(smt2, run_carcara)
        digest = _native.py_compute_certificate_receipt(m._json({"ihcg": ihcg.h_ihcg, "constraints": {k: sources[k] for k in core}, "smt2": smt2}))
        return _certificate(mode, ihcg, fidelity, minimal_unsat_core=tuple(core), proof_checker="cvc5_alethe_carcara" if external == "VERIFIED" else "z3_quickxplain",
                            proof_digest=digest, proof_smt2=smt2, proof_constraints={k: sources[k] for k in core}, core_minimal=True,
                            model_consistency_status=consistency, solver_calls=calls, external_checker_status=external)
    except _Unresolved as error:
        return _certificate(mode, ihcg, fidelity, str(error), model_consistency_status=consistency, solver_calls=calls, budget_exhausted=str(error) == "BUDGET_EXHAUSTED")


def _claim_margin(ihcg, *, substitute_model=True):
    def margin(node):
        if isinstance(node, ast.BoolOp):
            op = "min" if isinstance(node.op, ast.And) else "max"
            values = [margin(v) for v in node.values]
            out = values[0]
            for value in values[1:]: out = f"{op}({out},{value})"
            return out
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not): return f"-({margin(node.operand)})"
        if isinstance(node, ast.Call) and node.func.id == "implies": return f"max(-({margin(node.args[0])}),{margin(node.args[1])})"
        if not isinstance(node, ast.Compare): raise ValueError("claim needs a signed real-valued comparison")
        parts = []
        for a, op, b in zip([node.left, *node.comparators], node.ops, node.comparators):
            a, b = ast.unparse(a), ast.unparse(b)
            if isinstance(op, (ast.Lt, ast.LtE)): parts.append(f"(({b})-({a}))")
            elif isinstance(op, (ast.Gt, ast.GtE)): parts.append(f"(({a})-({b}))")
            elif isinstance(op, ast.Eq): parts.append(f"-abs(({a})-({b}))")
            else: parts.append(f"abs(({a})-({b}))")
        out = parts[0]
        for part in parts[1:]: out = f"min({out},{part})"
        return out
    claims = [o.expression for o in ihcg.obligations if o.obligation_class == "HYPOTHESIS_TARGET"]
    node = m._parse(margin(m._parse(m._and(claims))))
    if not substitute_model: return node
    substitutions = {k: m._parse(e.expression) for k, e in ihcg.equations.items()
                     if e.kind == "ALGEBRAIC" and k in ihcg.omega and not _is_formula(m._parse(e.expression)) and e.expression.strip() != k}
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, current):
            return m._parse(ast.unparse(substitutions[current.id])) if current.id in substitutions else current
    for _ in range(len(substitutions)+1):
        if not m._variables(node) & substitutions.keys(): return node
        node = Substitute().visit(node)
    raise ValueError("cyclic model substitution has no closed interval margin")


def _bound_expression(ihcg, expression):
    derived = _claim_margin(ihcg)
    if expression is None or ast.dump(m._parse(expression)) == ast.dump(derived): return derived
    result = json.loads(_native.py_check_algebraic_equivalence(m._json(m._csel(derived)), m._json(m._csel(m._parse(expression))), time_limit_ms=50))
    if not result["equivalent"]: raise ValueError("margin expression is not the sealed hypothesis margin")
    return m._parse(expression)


def _point(ihcg, point):
    if set(point) != set(ihcg.omega): raise ValueError("a concrete witness must cover the entire IHCG frame")
    result = {s: (float(x), float(x)) for s, x in point.items()}
    if any(not math.isfinite(x[0]) or not ihcg.omega[s][0] <= x[0] <= ihcg.omega[s][1] for s, x in result.items()): raise ValueError("witness is outside Omega")
    if m._truth(m._parse(m._and(ihcg.environmental_assumptions)), result) is not True:
        max_eq_res = 0.0
        for expr in ihcg.environmental_assumptions:
            node = m._parse(expr)
            if m._truth(node, result) is True:
                continue
            if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                lhs_iv = m._iv(node.left, result)
                rhs_iv = m._iv(node.comparators[0], result)
                res = max(abs(lhs_iv.lo - rhs_iv.lo), abs(lhs_iv.hi - rhs_iv.hi))
                scale = max(1.0, abs(lhs_iv.lo), abs(lhs_iv.hi), abs(rhs_iv.lo), abs(rhs_iv.hi))
                if res <= 1e-9 * scale:
                    max_eq_res = max(max_eq_res, res)
                    continue
            raise ValueError("witness does not interval-verify environmental assumptions")
        if max_eq_res > 0.0:
            pad = max(1e-11, 10.0 * max_eq_res)
            dyn = {s.symbol for s in ihcg.slots if s.role == "DYNAMIC_STATE"}
            for s in dyn:
                lo, hi = ihcg.omega[s]
                v = result[s][0]
                result[s] = (max(lo, v - pad), min(hi, v + pad))
    return result


def _model_feasibility(ihcg, box, deadline=None):
    constraints = [*_model_constraints(ihcg).values(), *ihcg.environmental_assumptions]
    return m.solve_query(m._and(constraints), box, witness=True, _deadline=deadline).status


def _algebraic_witness_box(ihcg, point):
    box = _point(ihcg,point)
    outputs = {s.symbol for s in ihcg.slots if s.role in {"DERIVED_OBSERVABLE","CONSERVATION_AUX"}}
    pending = {name:e for name,e in ihcg.equations.items() if name in outputs and e.kind == "ALGEBRAIC" and not _is_formula(m._parse(e.expression))}
    while pending:
        ready = [name for name,e in pending.items() if not m._variables(m._parse(e.expression)) & pending.keys()]
        if not ready: raise _Unresolved("CYCLIC_ALGEBRAIC_WITNESS_UNSUPPORTED")
        for name in ready:
            value = m._iv(m._parse(pending.pop(name).expression),box)
            lo,hi = ihcg.omega[name]
            a,b = max(lo,value.lo),min(hi,value.hi)
            if a > b: raise _Unresolved("WITNESS_OUTSIDE_OMEGA")
            box[name] = (a,b)
    return box


def run_mode2a_counterexample_falsification(ihcg: IhcgGraph, theta_star: Mapping[str, float],
                                          eval_res: WitnessEvaluation, fidelity: FidelityAudit) -> FalsificationCertificate:
    if ihcg.quantifier_mode != "FORALL": raise ValueError("Mode 2A requires a universal claim")
    mode = "MODE2A_COUNTEREXAMPLE"
    if reason := _gate(ihcg, fidelity): return _certificate(mode, ihcg, fidelity, reason)
    if eval_res.epsilon_total != fidelity.epsilon_total: raise ValueError("witness may not reduce the certified error budget")
    try: box = _algebraic_witness_box(ihcg, theta_star)
    except _Unresolved: return _certificate(mode, ihcg, fidelity, "WITNESS_MODEL_NOT_CERTIFIED")
    node = _bound_expression(ihcg, eval_res.margin_expression)
    if _model_feasibility(ihcg, box) != "SAT":
        return _certificate(mode, ihcg, fidelity, "WITNESS_MODEL_NOT_CERTIFIED")
    try: bound = m._iv(node, box)
    except (ValueError, OverflowError): return _certificate(mode, ihcg, fidelity, "WITNESS_INTERVAL_UNVERIFIED")
    if not bound.lo-fidelity.epsilon_total <= eval_res.signed_margin <= bound.hi+fidelity.epsilon_total:
        return _certificate(mode, ihcg, fidelity, "EXECUTABLE_MARGIN_MISMATCH")
    signed = max(bound.hi, eval_res.signed_margin)
    reason = None if signed+fidelity.epsilon_total < 0 else "MARGIN_WITHIN_UNCERTAINTY_OR_NOT_NEGATIVE"
    return _certificate(mode, ihcg, fidelity, reason, counterexample_theta=dict(theta_star), signed_margin_at_witness=signed,
                        witness_margin_interval=(bound.lo, bound.hi))


def dimension_scaled_box_budget(n: int) -> int:
    if not 1 <= n <= 64: raise ValueError("refutation dimension must be between 1 and 64")
    return min(32768, 2048*2**max(0, n-3))


def _gradient_enclosure(node, box, names):
    """Whole-cell interval AD; min/max/abs use conservative generalized slopes."""
    zero = lambda: [m._IV(0) for _ in names]
    if isinstance(node, ast.Constant): return zero()
    if isinstance(node, ast.Name): return [m._IV(int(s == node.id)) for s in names]
    if isinstance(node, ast.UnaryOp):
        gradient = _gradient_enclosure(node.operand, box, names)
        return [-g for g in gradient] if isinstance(node.op, ast.USub) else gradient
    if isinstance(node, ast.BinOp):
        a, b = m._iv(node.left, box), m._iv(node.right, box)
        ga, gb = _gradient_enclosure(node.left, box, names), _gradient_enclosure(node.right, box, names)
        if isinstance(node.op, ast.Add): return [x+y for x, y in zip(ga, gb)]
        if isinstance(node.op, ast.Sub): return [x-y for x, y in zip(ga, gb)]
        if isinstance(node.op, ast.Mult): return [x*b+a*y for x, y in zip(ga, gb)]
        if isinstance(node.op, ast.Div): return [(x*b-a*y)/(b*b) for x, y in zip(ga, gb)]
        exponent = m._fraction(node.right)
        if exponent is not None and exponent.denominator == 1:
            if exponent == 0: return zero()
            if exponent == 1: return ga
            factor = m._iv(m._parse(f"({exponent})*({ast.unparse(node.left)})**({exponent-1})"), box)
            return [factor*g for g in ga]
        return _gradient_enclosure(m._parse(f"exp(({ast.unparse(node.right)})*log({ast.unparse(node.left)}))"), box, names)
    if isinstance(node, ast.Call):
        name = node.func.id
        if name in {"min", "max"}:
            gradients = [_gradient_enclosure(a, box, names) for a in node.args]
            return [m._IV(min(g[i].lo for g in gradients), max(g[i].hi for g in gradients)) for i in range(len(names))]
        if name == "abs":
            argument = m._iv(node.args[0], box); gradient = _gradient_enclosure(node.args[0], box, names)
            slope = m._IV(1) if argument.lo > 0 else m._IV(-1) if argument.hi < 0 else m._IV(-1, 1)
            return [slope*g for g in gradient]
        if name not in {"exp", "log", "sqrt", "sin", "cos"}:
            expanded = m._expand(node)
            if ast.dump(expanded) == ast.dump(node): raise m._DomainError("no verified gradient enclosure")
            return _gradient_enclosure(expanded, box, names)
        text = ast.unparse(node.args[0])
        factor = m._iv(m._parse({"exp": f"exp({text})", "log": f"1/({text})", "sqrt": f"0.5/sqrt({text})", "sin": f"cos({text})", "cos": f"-sin({text})"}[name]), box)
        return [factor*g for g in _gradient_enclosure(node.args[0], box, names)]
    raise m._DomainError("unsupported gradient expression")


def _lipschitz_grid(ihcg, spec, fidelity, limit, deadline):
    mode, box = "MODE2B_EXHAUSTIVE_ENCLOSURE", dict(ihcg.omega)
    if spec.evaluator is None or spec.gradient_enclosure is None or spec.ihcg_digest != ihcg.h_ihcg:
        return _certificate(mode, ihcg, fidelity, "VALIDATED_LIPSCHITZ_ORACLE_REQUIRED", effective_max_boxes=limit, unresolved_bound_unbounded=True)
    queue = [(box, Fraction(1))]; evaluated = certified = 0; volume = Fraction(0); unresolved = []; certified_upper = -math.inf
    try: expression = _bound_expression(ihcg, spec.expression)
    except ValueError: expression = None
    while queue and evaluated < limit and time.monotonic() < deadline:
        current, weight = queue.pop(); evaluated += 1
        center = {s: lo/2+hi/2 for s, (lo, hi) in current.items()}
        values = spec.evaluator(center); interval = m._IV(*values)
        gradient = spec.gradient_enclosure(current)
        if len(gradient) != len(current): raise ValueError("whole-cell gradient dimension mismatch")
        if expression is not None:
            try:
                exact = m._iv(expression, {s: (v, v) for s, v in center.items()})
                if interval.hi+fidelity.epsilon_total < exact.lo or interval.lo-fidelity.epsilon_total > exact.hi:
                    return _certificate(mode, ihcg, fidelity, "EXECUTABLE_MARGIN_MISMATCH", boxes_evaluated=evaluated, effective_max_boxes=limit)
                expected = _gradient_enclosure(expression, current, list(current))
                if any(not lo <= actual.lo <= actual.hi <= hi for (lo, hi), actual in zip(gradient, expected)):
                    return _certificate(mode, ihcg, fidelity, "GRADIENT_ENCLOSURE_MISMATCH", boxes_evaluated=evaluated, effective_max_boxes=limit)
            except m._DomainError:
                return _certificate(mode, ihcg, fidelity, "ORACLE_BINDING_UNVERIFIED", boxes_evaluated=evaluated, effective_max_boxes=limit)
        squared, radius_squared = m._IV(0), m._IV(0)
        for (lo, hi), (lower, upper) in zip(current.values(), gradient):
            if not all(map(math.isfinite, (lower, upper))) or lower > upper: raise ValueError("invalid gradient enclosure")
            norm = m._IV(max(abs(lower), abs(upper))); squared = squared+norm*norm
            radius = m._IV(hi)-m._IV(lo); radius = radius*m._IV(0.5); radius_squared = radius_squared+radius*radius
        lip = m._IV(1.5)*m._IV(max(0, squared.lo), squared.hi).op("sqrt")
        radius = m._IV(max(0, radius_squared.lo), radius_squared.hi).op("sqrt")
        upper = (interval+lip*radius).hi
        if not math.isfinite(upper): unresolved.append(None); continue
        if upper+fidelity.epsilon_total < 0:
            certified += 1; volume += weight; certified_upper = max(certified_upper, upper); continue
        split = max(current, key=lambda s: (current[s][1]-current[s][0], s))
        lo, hi = current[split]; mid = lo/2+hi/2
        if not lo < mid < hi: unresolved.append(upper); continue
        left, right = current.copy(), current.copy(); left[split], right[split] = (lo, mid), (mid, hi)
        fraction = (Fraction(mid)-Fraction(lo))/(Fraction(hi)-Fraction(lo))
        queue.extend(((right, weight*(1-fraction)), (left, weight*fraction)))
    exhausted = bool(queue)
    closed = not queue and not unresolved
    fraction = float(volume)
    if Fraction(fraction) > volume: fraction = math.nextafter(fraction, -math.inf)
    return _certificate(mode, ihcg, fidelity, None if closed else "BUDGET_EXHAUSTED" if exhausted else "UNRESOLVED_GRID_ENCLOSURE",
        budget_exhausted=exhausted, certified_refuted_volume_fraction=1.0 if closed else min(1.0, fraction), boxes_evaluated=evaluated,
        certified_subbox_count=certified, certified_sup_upper_bound=certified_upper if closed else None,
        max_unresolved_upper_bound=max((x for x in unresolved if x is not None), default=None), unresolved_bound_unbounded=bool(queue) or None in unresolved,
        effective_max_boxes=limit, enclosure_method="LIPSCHITZ_GRID")


def run_mode2b_exhaustive_refutation(ihcg: IhcgGraph, margin_spec: MarginSpec, fidelity: FidelityAudit,
                                    max_boxes: int | None = None) -> FalsificationCertificate:
    if ihcg.quantifier_mode != "EXISTS": raise ValueError("Mode 2B requires an existential claim")
    mode = "MODE2B_EXHAUSTIVE_ENCLOSURE"
    if reason := _gate(ihcg, fidelity): return _certificate(mode, ihcg, fidelity, reason)
    limit = dimension_scaled_box_budget(len(ihcg.omega))
    if max_boxes is not None:
        if not 1 <= max_boxes <= 32768: raise ValueError("invalid refutation box budget")
        limit = min(limit, max_boxes)
    deadline = time.monotonic()+1.5
    if _model_feasibility(ihcg, ihcg.omega, deadline) != "SAT":
        return _certificate(mode, ihcg, fidelity, "MODEL_ENVIRONMENT_NOT_CERTIFIED_FEASIBLE", effective_max_boxes=limit)
    if margin_spec.evaluator is not None:
        return _lipschitz_grid(ihcg, margin_spec, fidelity, limit, deadline)
    node = _bound_expression(ihcg, margin_spec.expression)
    names = sorted(m._variables(node))
    class Rename(ast.NodeTransformer):
        def visit_Name(self, current):
            return ast.Name(id=f"x{names.index(current.id)}", ctx=ast.Load()) if current.id in ihcg.omega else current
        def visit_BinOp(self,current):
            self.generic_visit(current)
            exponent = m._fraction(current.right)
            if isinstance(current.op,ast.Pow) and exponent is not None and exponent.denominator == 1 and 0 <= exponent <= 16:
                result = ast.Constant(1)
                for _ in range(exponent.numerator): result = ast.BinOp(left=result,op=ast.Mult(),right=copy.deepcopy(current.left))
                return result
            return current
    renamed = Rename().visit(node)
    indices = sorted(range(len(names)),key=lambda i:f"x{i}")
    try:
        result = json.loads(_native.py_branch_and_bound_refute(m._json(m._csel(renamed)), [ihcg.omega[names[i]][0] for i in indices],
                            [ihcg.omega[names[i]][1] for i in indices], fidelity.epsilon_total, max_boxes=limit))
    except ValueError:
        return _certificate(mode, ihcg, fidelity, "NATIVE_INTERVAL_EXTENSION_UNSUPPORTED", effective_max_boxes=limit, unresolved_bound_unbounded=True)
    closed = result["refuted"] and not result["budget_exhausted"]
    # Every certified native upper bound is an f64 strictly below -epsilon.
    # Its next smaller representable neighbor is therefore a valid (loose)
    # global upper bound without inventing the unreported maximum leaf bound.
    upper = math.nextafter(-fidelity.epsilon_total, -math.inf) if closed else None
    return _certificate(mode, ihcg, fidelity, None if closed else "BUDGET_EXHAUSTED" if result["budget_exhausted"] else "UNRESOLVED_NATIVE_ENCLOSURE",
        certified_sup_upper_bound=upper, certified_refuted_volume_fraction=result["certified_refuted_volume_fraction"], budget_exhausted=result["budget_exhausted"],
        boxes_evaluated=result["boxes_evaluated"], certified_subbox_count=result["certified_subbox_count"], max_unresolved_upper_bound=result["max_unresolved_upper_bound"],
        unresolved_bound_unbounded=result["unresolved_bound_unbounded"], effective_max_boxes=result["effective_max_boxes"], enclosure_method="NATIVE_STRICT_F64_THRESHOLD")


def certify_phase_boundary(ihcg: IhcgGraph, sandbox_runner: Callable[[Mapping[str, float]], WitnessEvaluation], delta_boundary: float, *,
                           fidelity: FidelityAudit, parameter: str | None = None, fixed_values: Mapping[str, float] | None = None,
                           budget_seconds: float = 30.0) -> BoundaryDiscoveryCertificate:
    ihcg.assert_sealed()
    if ihcg.quantifier_mode != "BOUNDARY" or not math.isfinite(delta_boundary) or delta_boundary <= 0: raise ValueError("boundary mode and positive tolerance required")
    varying = [s.symbol for s in ihcg.slots if s.role == "CONTROL_PARAM" and ihcg.omega[s.symbol][0] < ihcg.omega[s.symbol][1]]
    if parameter is None:
        if len(varying) != 1: raise ValueError("select one boundary parameter")
        parameter = varying[0]
    if parameter not in varying: raise ValueError("boundary parameter must be a nondegenerate control parameter")
    fixed = dict(fixed_values or {})
    for name, (lo, hi) in ihcg.omega.items():
        if name != parameter and name not in fixed:
            if lo != hi: raise ValueError("all other coordinates need pinned values")
            fixed[name] = lo
    left, right = {**fixed, parameter: ihcg.omega[parameter][0]}, {**fixed, parameter: ihcg.omega[parameter][1]}
    reason, evaluations, deadline = _gate(ihcg, fidelity), 0, time.monotonic()+budget_seconds
    node = _claim_margin(ihcg)
    enclosures = {}
    def evaluate(point):
        nonlocal evaluations
        if evaluations >= 64 or time.monotonic() >= deadline: raise _Unresolved("BUDGET_EXHAUSTED")
        evaluations += 1; box = _algebraic_witness_box(ihcg, point); result = sandbox_runner(dict(point))
        for name,(lo,hi) in box.items(): point[name] = (lo+hi)/2
        enclosures[point[parameter]] = box
        if _model_feasibility(ihcg, box, deadline) != "SAT": raise _Unresolved("WITNESS_MODEL_NOT_CERTIFIED")
        if not isinstance(result, WitnessEvaluation) or result.epsilon_total != fidelity.epsilon_total: raise ValueError("boundary runner must preserve fidelity uncertainty")
        expression = _bound_expression(ihcg, result.margin_expression)
        value = m._iv(expression, box)
        if not value.lo-fidelity.epsilon_total <= result.signed_margin <= value.hi+fidelity.epsilon_total: raise _Unresolved("EXECUTABLE_MARGIN_MISMATCH")
        return value.lo-fidelity.epsilon_total, value.hi+fidelity.epsilon_total
    lower, upper = (0.0, 0.0), (0.0, 0.0)
    try:
        if reason: raise _Unresolved(reason)
        m._iv(node, {s: (left[s], right[s]) if s == parameter else (fixed[s], fixed[s]) for s in ihcg.omega})
        lower, upper = evaluate(left), evaluate(right)
        def sign(v): return 1 if v[0] > 0 else -1 if v[1] < 0 else 0
        if not sign(lower)*sign(upper) == -1: raise _Unresolved("OPPOSING_CERTIFIED_WITNESSES_REQUIRED")
        while right[parameter]-left[parameter] > delta_boundary:
            width = right[parameter]-left[parameter]; changed = False
            for fraction in (0.5, 0.25, 0.75):
                point = {**fixed, parameter: left[parameter]+fraction*(right[parameter]-left[parameter])}
                if not left[parameter] < point[parameter] < right[parameter]: continue
                value = evaluate(point)
                if sign(value) == sign(lower): left, lower, changed = point, value, True; break
                if sign(value) == sign(upper): right, upper, changed = point, value, True; break
            if not changed or right[parameter]-left[parameter] >= width: raise _Unresolved("MARGIN_WITHIN_UNCERTAINTY")
        reason = None
    except (_Unresolved, m._DomainError) as error:
        reason = str(error)
    valid, invalid = (left, right) if lower[0] > 0 else (right, left)
    positive, negative = (lower, upper) if lower[0] > 0 else (upper, lower)
    return BoundaryDiscoveryCertificate(valid_witness_theta=valid, falsified_witness_theta=invalid, valid_margin_lower=positive[0], falsified_margin_upper=negative[1],
        bracket_width=right[parameter]-left[parameter], target_tolerance=delta_boundary, epsilon_total=fidelity.epsilon_total,
        ihcg_digest=ihcg.h_ihcg, evaluations=evaluations, budget_exhausted=reason == "BUDGET_EXHAUSTED", failure_reason=reason,
        boundary_parameter=parameter,valid_witness_enclosure=enclosures.get(valid[parameter]),falsified_witness_enclosure=enclosures.get(invalid[parameter]),
        epistemic_verdict=EpistemicVerdict.INCONCLUSIVE_UNVERIFIED if reason else EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY)
