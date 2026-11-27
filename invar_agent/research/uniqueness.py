"""Native CDS commitments and conservative novelty in canonical Pi coordinates.

Projected boxes are NOT assumed to be exact domains: correlated Pi images use
source-box containment or an attained marginal extension as sufficient tests.
Unresolved reference annotations and domain comparisons fail closed. Repository
names, SHBT labels and zero dissipation never bypass a novelty obligation.
"""

from __future__ import annotations

import ast
import copy
import itertools
import json
import math
import time
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from invar_agent import _native
from . import meaningfulness as m
from .extractor import ReferenceRegistry as LEGACY_REFERENCE_REGISTRY, parse_equation


VaultClass = Literal["NOVEL_FIRST_IN_VAULT", "NOVEL_INDEPENDENT", "EXACT_ALGEBRAIC_DUPLICATE", "VALID_DOMAIN_EXTENSION", "VALID_BOUND_SHARPENING", "VALID_COUPLED_EXTENSION", "VALID_REFUTATION_OF_PRIOR"]


class CanonicalizationError(ValueError):
    pass


class PiDomain(m.StrictModel):
    bounds: dict[str, tuple[str, str]]
    source_bounds: tuple[tuple[str, str], ...]
    exact_box: bool
    full_dimensional: bool
    coordinate_system: str
    environmental_constraints: tuple[str, ...] = ()


class CdsTriplet(m.StrictModel):
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    full_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_model: str
    canonical_claim: str
    domain_and_verdict: str
    canonical_claim_expressions: tuple[str, ...]
    pi_domain: PiDomain
    equation_count: int = Field(ge=1)
    stage_count: int = Field(ge=1)

    @model_validator(mode="after")
    def commitments_match_payload(self):
        payload = json.loads(self.domain_and_verdict)
        if (payload["domain"] != self.pi_domain.model_dump(mode="json") or payload["equation_count"] != self.equation_count
                or payload["stage_count"] != self.stage_count or self.canonical_claim != m._json({"predicates": self.canonical_claim_expressions})):
            raise ValueError("CDS metadata is not the committed payload")
        hashes = json.loads(_native.py_compute_cds_digests(self.canonical_model, self.canonical_claim, self.domain_and_verdict))
        if any(hashes[name] != getattr(self, name) for name in hashes): raise ValueError("CDS digest mismatch")
        return self


class VaultEntry(m.StrictModel):
    run_id: str
    cds: CdsTriplet
    verdict: m.EpistemicVerdict
    epsilon_total: float = Field(ge=0)
    epsilon_disc: float = Field(ge=0)

    @model_validator(mode="after")
    def verdict_matches_commitment(self):
        if self.verdict.value != json.loads(self.cds.domain_and_verdict)["verdict"] or self.epsilon_disc > self.epsilon_total:
            raise ValueError("Vault verdict or uncertainty differs from its commitment")
        return self


class VaultUniquenessAudit(m.StrictModel):
    passed: bool
    vault_class: VaultClass | None = None
    related_run_id: str | None = None
    lineage_edge: Literal["extends_domain_of", "sharpens_bounds_of", "refutes", "coupled_extension_of"] | None = None
    failure_reason: str | None = None


class NonTrivialityAudit(m.StrictModel):
    passed: bool
    reason: str | None = None
    matched_reference: str | None = None
    reference_coverage_complete: bool
    unresolved_references: tuple[str, ...] = ()
    shbt_open_boundary_closure: bool = False


def _rref_basis(tags, order):
    matrix = [[Fraction(tags[s]["si_dim"][i]) for s in order] for i in range(7)]
    pivots, row = [], 0
    for column in range(len(order)):
        pivot = next((i for i in range(row, 7) if matrix[i][column]), None)
        if pivot is None: continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        scale = matrix[row][column]; matrix[row] = [x/scale for x in matrix[row]]
        for i in range(7):
            if i != row:
                scale = matrix[i][column]; matrix[i] = [a-scale*b for a, b in zip(matrix[i], matrix[row])]
        pivots.append(column); row += 1
        if row == 7: break
    free, basis = [i for i in range(len(order)) if i not in pivots], []
    for column in free:
        vector = [Fraction(0)]*len(order); vector[column] = Fraction(1)
        for i, pivot in enumerate(pivots): vector[pivot] = -matrix[i][column]
        scale = math.lcm(*(v.denominator for v in vector))
        integers = [int(v*scale) for v in vector]; divisor = math.gcd(*integers)
        basis.append([v//divisor for v in integers])
    return pivots, free, basis


def _fraction_bounds_power(pair, power):
    a, b = pair
    if power < 0:
        if a <= 0 <= b: raise CanonicalizationError("Pi denominator is not domain-proven nonzero")
        return _fraction_bounds_power((1/b, 1/a), -power)
    if power == 0: return Fraction(1), Fraction(1)
    if power > 4096: raise CanonicalizationError("Pi exponent budget exceeded")
    values = [a**power, b**power]
    if power % 2 == 0 and a <= 0 <= b: values.append(Fraction(0))
    return min(values), max(values)


def _pi_domain(ihcg, tags, order, pivots, free, basis, chart, environment):
    source = []
    for s in order:
        scale = Fraction(*tags[s]["unit_scale"])
        source.append(tuple(Fraction(str(v))*scale for v in ihcg.omega[s]))
    bounds = {}
    for i, vector in enumerate(basis):
        lo = hi = Fraction(1)
        for pair, power in zip(source, vector):
            a, b = _fraction_bounds_power(pair, power)
            products = (lo*a, lo*b, hi*a, hi*b); lo, hi = min(products), max(products)
        bounds[f"pi{i}"] = (str(lo), str(hi))
    return PiDomain(bounds=bounds, source_bounds=tuple((str(a), str(b)) for a, b in source),
                    exact_box=not environment and all(source[p][0] == source[p][1] for p in pivots),
                    full_dimensional=not environment and bool(free) and all(source[p][0] < source[p][1] for p in free),
                    coordinate_system=chart, environmental_constraints=environment)


def _residuals(ihcg, uast):
    equations = uast.get("equations", {k: e.expression for k, e in ihcg.equations.items()})
    if set(equations) != set(ihcg.equations): raise CanonicalizationError("canonical model must preserve every sealed equation")
    residuals = []
    for name, expression in sorted(equations.items()):
        if ast.dump(m._parse(expression)) != ast.dump(m._parse(ihcg.equations[name].expression)):
            raise CanonicalizationError("canonical model differs from the sealed IHCG")
        node = m._parse(expression)
        if ihcg.equations[name].kind in {"FIXED_POINT_RESIDUAL", "FIXED_POINT_MAP"}:
            state = ihcg.equations[name].state_symbol
            residuals.append(expression if ihcg.equations[name].kind == "FIXED_POINT_RESIDUAL" else f"({expression})-({state})")
            continue
        if ihcg.equations[name].kind == "MC_SAMPLE" and name not in {s.symbol for s in ihcg.slots}:
            residuals.append(f"({ihcg.equations[name].state_symbol})-({expression})")
            continue
        if ihcg.equations[name].kind == "ODE_RHS":
            residuals.append(expression)
            continue
        comparisons = [n for n in ast.walk(node) if isinstance(n, ast.Compare)]
        if comparisons:
            for comparison in comparisons:
                residuals.extend(f"({ast.unparse(a)})-({ast.unparse(b)})" for a, b in zip([comparison.left, *comparison.comparators], comparison.comparators))
        elif uast.get("equation_form") == "residual": residuals.append(expression)
        elif name in ihcg.omega: residuals.append(f"({name})-({expression})")
        else: raise CanonicalizationError("declare equation_form='residual' or use a declared assignment output")
    if not residuals: raise CanonicalizationError("empty governing system")
    return residuals


def _tagged(expressions, tags):
    return [{"expr": m._json(m._csel(m._parse(expr))), "symbols": {s: tags[s] for s in sorted(m._variables(m._parse(expr)))}} for expr in expressions]


def _native_canonical(equations):
    try: result = json.loads(_native.py_compute_alpha_canonical_system(m._json(equations), max_aut_branches=64))
    except ValueError as error: raise CanonicalizationError(str(error)) from error
    if not result["saturated"]: raise CanonicalizationError("native canonicalization did not saturate")
    return result["canonical_system"]


def _canonical_arithmetic(node):
    return _native.py_canonicalize_csel_expr(m._json(m._csel(node)))


def _polynomial(node):
    if isinstance(node, ast.Constant): return {(): Fraction(str(node.value))}
    if isinstance(node, ast.Name): return {((node.id, 1),): Fraction(1)}
    if isinstance(node, ast.Call):
        arguments = [_polynomial_text(_polynomial(arg)) for arg in node.args]
        atom = f"{node.func.id}({','.join(arguments)})"
        return {((atom,1),): Fraction(1)}
    if isinstance(node, ast.UnaryOp):
        return {k: (-v if isinstance(node.op, ast.USub) else v) for k, v in _polynomial(node.operand).items()}
    if not isinstance(node, ast.BinOp): raise CanonicalizationError("nonpolynomial oriented claim normalization")
    a, b = _polynomial(node.left), _polynomial(node.right)
    if isinstance(node.op, (ast.Add, ast.Sub)):
        out = a.copy()
        for k, v in b.items(): out[k] = out.get(k, 0)+(-v if isinstance(node.op, ast.Sub) else v)
        return {k: v for k, v in out.items() if v}
    def multiply(a, b):
        out = {}
        for x, c in a.items():
            for y, d in b.items():
                powers = dict(x)
                for s, p in y: powers[s] = powers.get(s, 0)+p
                key = tuple(sorted((s, p) for s, p in powers.items() if p))
                out[key] = out.get(key, 0)+c*d
        if len(out) > 4096: raise CanonicalizationError("oriented polynomial budget exceeded")
        return {k: v for k, v in out.items() if v}
    if isinstance(node.op, ast.Mult): return multiply(a, b)
    if isinstance(node.op, ast.Div) and len(b) == 1:
        (powers, coefficient), = b.items()
        if not coefficient: raise CanonicalizationError("zero divisor")
        return multiply(a, {tuple((s, -p) for s, p in powers): 1/coefficient})
    exponent = m._fraction(node.right)
    if isinstance(node.op, ast.Pow) and exponent is not None and exponent.denominator == 1 and 0 <= exponent <= 16:
        out = {(): Fraction(1)}
        for _ in range(exponent.numerator): out = multiply(out, a)
        return out
    raise CanonicalizationError("nonpolynomial oriented claim normalization")


def _polynomial_text(polynomial):
    return "+".join(f"(({v})*({ '*'.join(f'({s})**({p})' for s, p in powers) or '1'}))" for powers,v in sorted(polynomial.items())) or "0"


def _normalized_residual(node, oriented=True):
    try:
        polynomial = _polynomial(node)
        if not polynomial: return "0"
        first = polynomial[min(polynomial)]; divisor = abs(first) if oriented else first
        return "+".join(f"(({v/divisor})*({ '*'.join(f'({s})**({p})' for s, p in powers) or '1'}))" for powers, v in sorted(polynomial.items()))
    except CanonicalizationError:
        if oriented: return ast.unparse(ast.fix_missing_locations(node))
        return _canonical_arithmetic(node)


class CanonicalSignatureEngine:
    def compute_cds(self, ihcg: m.IhcgGraph, uast: Mapping[str, Any], verdict: m.EpistemicVerdict) -> CdsTriplet:
        ihcg.assert_sealed(); deadline = time.monotonic()+20
        residuals = _residuals(ihcg, uast)
        tags = {}
        metadata = uast.get("symbol_metadata", {})
        strictly_positive = set()
        for expr_text in [*ihcg.environmental_assumptions, *(o.expression for o in ihcg.obligations if o.obligation_class == "CONSERVATION_GUARD")]:
            for node in ast.walk(m._parse(expr_text)):
                if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
                    lhs_n, op_n, rhs_n = node.left, node.ops[0], node.comparators[0]
                    if isinstance(lhs_n, ast.Name) and isinstance(op_n, ast.Gt) and isinstance(rhs_n, ast.Constant) and type(rhs_n.value) in (int, float) and rhs_n.value >= 0:
                        strictly_positive.add(lhs_n.id)
                    elif isinstance(rhs_n, ast.Name) and isinstance(op_n, ast.Lt) and isinstance(lhs_n, ast.Constant) and type(lhs_n.value) in (int, float) and lhs_n.value >= 0:
                        strictly_positive.add(rhs_n.id)
        for slot in ihcg.slots:
            extra = metadata.get(slot.symbol, {})
            if set(extra)-{"unit_scale"}: raise CanonicalizationError("only exact coordinate unit scales may supplement sealed slot metadata")
            scale = extra.get("unit_scale", [1, 1])
            if len(scale) != 2 or any(type(x) is not int for x in scale) or scale[0] <= 0 or scale[1] <= 0: raise CanonicalizationError("positive rational unit scale required")
            tags[slot.symbol] = dict(role=slot.role, si_dim=list(slot.si_dim), semantic_tag=slot.semantic_tag,
                                     unit_scale=list(scale), positive=ihcg.omega[slot.symbol][0] > 0 or (ihcg.omega[slot.symbol][0] >= 0 and slot.symbol in strictly_positive),
                                     domain=list(ihcg.omega[slot.symbol]))
        model = _native_canonical(_tagged(residuals, tags))
        claims = [o.expression for o in ihcg.obligations if o.obligation_class == "HYPOTHESIS_TARGET"]
        governing = [e.expression for e in ihcg.equations.values() if isinstance(m._parse(e.expression), (ast.BoolOp, ast.UnaryOp, ast.Call)) and any(isinstance(n, ast.Compare) for n in ast.walk(m._parse(e.expression)))
                     or isinstance(m._parse(e.expression), ast.Compare) and any(not isinstance(op, ast.Eq) for op in m._parse(e.expression).ops)]
        comparison_residuals = []
        for claim in [*claims, *ihcg.environmental_assumptions]:
            for node in ast.walk(m._parse(claim)):
                if isinstance(node, ast.Compare):
                    for a, b in zip([node.left, *node.comparators], node.comparators):
                        cand = f"({ast.unparse(a)})-({ast.unparse(b)})"
                        try:
                            _native_canonical(_tagged([cand], tags))
                            comparison_residuals.append(cand)
                        except CanonicalizationError:
                            # Dimensional constant comparison (e.g. time_days <= 120 or gamma == 0.1):
                            # bare numeric literal carries implicit SI units; omit from untagged Pi coloring.
                            continue
        joint = _tagged([*residuals, *comparison_residuals], tags)
        marker = next((i for i in range(256) if i not in {s.semantic_tag for s in ihcg.slots}), None)
        if marker is None: raise CanonicalizationError("no individualization marker available")
        colors = {}
        for symbol in sorted(tags):
            if time.monotonic() >= deadline: raise CanonicalizationError("canonical coordinate budget exhausted")
            rooted = copy.deepcopy(joint)
            for equation in rooted:
                if symbol in equation["symbols"]: equation["symbols"][symbol]["semantic_tag"] = marker
            colors[symbol] = (not tags[symbol]["positive"], _native_canonical(rooted), m._json({k: v for k, v in tags[symbol].items() if k not in {"unit_scale", "domain"}}))
        referenced_symbols = set()
        for expr_text in [*residuals, *comparison_residuals, *claims, *governing, *ihcg.environmental_assumptions]:
            referenced_symbols.update(m._variables(m._parse(expr_text)))
        for eq in ihcg.equations.values():
            referenced_symbols.update(m._variables(m._parse(eq.expression)))
            if eq.state_symbol:
                referenced_symbols.add(eq.state_symbol)
        cells = {}
        for symbol, color in colors.items(): cells.setdefault(color, []).append(symbol)
        cell_perms = []
        count = 1
        for color in sorted(cells):
            cell = cells[color]
            active_cell = [s for s in cell if s in referenced_symbols or any(tags[s]["si_dim"])]
            inert_cell = tuple(s for s in cell if s not in active_cell)
            count *= math.factorial(len(active_cell))
            cell_perms.append((active_cell, inert_cell))
        if count > 64: raise CanonicalizationError("canonical domain symmetry budget exhausted")
        best = None
        for parts in itertools.product(*((p + inert for p in itertools.permutations(active)) for active, inert in cell_perms)):
            if time.monotonic() >= deadline: raise CanonicalizationError("canonical coordinate budget exhausted")
            order = [s for part in parts for s in part]
            pivots, free, basis = _rref_basis(tags, order)
            if any(not tags[order[p]]["positive"] for p in pivots): raise CanonicalizationError("positive repeating quantities required")
            replacements = {}
            for j, symbol in enumerate(order):
                if j in free:
                    index = free.index(j); degree = basis[index][j]
                    if degree != 1 and not tags[symbol]["positive"]: raise CanonicalizationError("positive radical coordinate required")
                    value = f"pi{index}" if degree == 1 else f"pi{index}**(1/{degree})"
                else: value = "1"
                p, q = tags[symbol]["unit_scale"]; replacements[symbol] = m._parse(f"({value})*({q}/{p})")
            class Substitute(ast.NodeTransformer):
                def visit_Name(self, n): return copy.deepcopy(replacements[n.id]) if n.id in replacements else n
            normalized_model = tuple(sorted(_normalized_residual(Substitute().visit(m._parse(expr)), False) for expr in residuals))
            evolution = tuple(sorted([(ast.unparse(Substitute().visit(m._parse(e.state_symbol))),
                ("" if e.kind == "ODE_RHS" else e.kind+":")+_normalized_residual(Substitute().visit(m._parse(e.expression)), True))
                for e in ihcg.equations.values() if e.kind != "ALGEBRAIC"]+
                [(ast.unparse(Substitute().visit(m._parse(s.symbol))),"stochastic:"+s.stochastic_role) for s in ihcg.slots if s.stochastic_role]))
            def predicate(node):
                if isinstance(node, ast.Constant) and isinstance(node.value, bool): return str(node.value)
                if isinstance(node, ast.BoolOp):
                    return "("+(" and " if isinstance(node.op, ast.And) else " or ").join(sorted(predicate(x) for x in node.values))+")"
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not): return f"not ({predicate(node.operand)})"
                if isinstance(node, ast.Call) and node.func.id == "implies": return f"implies({predicate(node.args[0])},{predicate(node.args[1])})"
                if not isinstance(node, ast.Compare): raise CanonicalizationError("boolean comparison claim required")
                parts = []
                for a, op, b in zip([node.left, *node.comparators], node.ops, node.comparators):
                    a, b = Substitute().visit(copy.deepcopy(a)), Substitute().visit(copy.deepcopy(b))
                    if isinstance(op, (ast.Lt, ast.LtE)): a, b = b, a
                    operator = ">" if isinstance(op, (ast.Lt, ast.Gt)) else ">=" if isinstance(op, (ast.LtE, ast.GtE)) else "==" if isinstance(op, ast.Eq) else "!="
                    residual = _normalized_residual(ast.BinOp(left=a, op=ast.Sub(), right=b), oriented=operator not in {"==", "!="})
                    parts.append(f"({residual}) {operator} 0")
                return m._and(parts)
            normalized_claims = tuple(sorted(predicate(m._parse(expr)) for expr in claims))
            normalized_governing = tuple(sorted(predicate(m._parse(expr)) for expr in governing))
            environment = tuple(sorted(predicate(m._parse(expr)) for expr in ihcg.environmental_assumptions if expr != "True"))
            chart = m._json({"model": normalized_model, "basis": basis, "tags": [{k: v for k, v in tags[s].items() if k not in {"unit_scale", "domain"}} for s in order]})
            domain = _pi_domain(ihcg, tags, order, pivots, free, basis, chart, environment)
            key = (normalized_model, evolution, normalized_governing, normalized_claims, domain.model_dump_json())
            if best is None or key < best[0]: best = key, normalized_claims, domain, normalized_governing, evolution
        _, normalized_claims, domain, normalized_governing, evolution = best
        if normalized_governing: model = m._json({"native_system": model, "governing_predicates": normalized_governing})
        if evolution: model = m._json({"native_system":model,"evolution":evolution})
        claim = m._json({"predicates": normalized_claims})
        full = m._json({"domain": domain.model_dump(mode="json"), "verdict": verdict.value, "quantifier": ihcg.quantifier_mode,
                        "equation_count": len(residuals), "stage_count": 1+len(ihcg.stage_chain_digests)})
        hashes = json.loads(_native.py_compute_cds_digests(model, claim, full))
        return CdsTriplet(**hashes, canonical_model=model, canonical_claim=claim, domain_and_verdict=full, canonical_claim_expressions=normalized_claims,
                          pi_domain=domain, equation_count=len(residuals), stage_count=1+len(ihcg.stage_chain_digests))


def _intervals(domain): return {s: (Fraction(a), Fraction(b)) for s, (a, b) in domain.bounds.items()}


def _contains(outer, inner):
    return len(outer) == len(inner) and all(a <= c <= d <= b for (a, b), (c, d) in zip(outer, inner))


def _relations(new, old):
    if new.coordinate_system != old.coordinate_system or new.environmental_constraints != old.environmental_constraints: return None
    n, o = _intervals(new), _intervals(old)
    raw_n, raw_o = [[tuple(map(Fraction, pair)) for pair in domain.source_bounds] for domain in (new, old)]
    subset = _contains(raw_o, raw_n) or old.exact_box and _contains(list(o.values()), list(n.values()))
    overlap = all(max(a, c) <= min(b, d) for (a, b), (c, d) in zip(raw_n, raw_o)) or new.exact_box and old.exact_box and all(max(n[s][0], o[s][0]) <= min(n[s][1], o[s][1]) for s in n)
    if new.environmental_constraints: overlap = subset
    extension = new.full_dimensional and any(n[s][0] < o[s][0] or n[s][1] > o[s][1] for s in n)
    return subset, overlap, extension


def _strict_strengthening(new, old, epsilon_new, epsilon_old):
    if not new.pi_domain.exact_box or len(new.canonical_claim_expressions) != 1: return False
    box = {s: (float(a), float(b)) for s, (a, b) in _intervals(new.pi_domain).items()}
    # Widen conversions before solver queries; never prove over an inward box.
    box = {s: (math.nextafter(a, -math.inf), math.nextafter(b, math.inf)) for s, (a, b) in box.items()}
    if not box: return False
    hnew, hold = m._and(new.canonical_claim_expressions), m._and(old.canonical_claim_expressions)
    implication = m.solve_query(f"({hnew}) and not ({hold})", box)
    if implication.status != "UNSAT": return False
    margins = []
    for signature in (new, old):
        if len(signature.canonical_claim_expressions) != 1: return False
        node = m._parse(signature.canonical_claim_expressions[0])
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Gt, ast.GtE)): return False
        margins.append(f"({ast.unparse(node.left)})-({ast.unparse(node.comparators[0])})")
    strict = m.solve_query(f"({hold}) and not ({hnew}) and ({margins[0]}) < {-5*epsilon_new!r} and ({margins[1]}) > {5*epsilon_old!r}", box, witness=True)
    if strict.status != "SAT" or strict.witness is None: return False
    return all(Fraction(str(strict.witness[s])) >= a and Fraction(str(strict.witness[s])) <= b for s, (a, b) in _intervals(new.pi_domain).items())


def si_error_budgets(ihcg,epsilon_disc,epsilon_total):
    if all(s.unit_scale == (1,1) for s in ihcg.slots): return epsilon_disc,epsilon_total
    scales = {s.symbol:Fraction(*s.unit_scale) for s in ihcg.slots}
    factors = set()
    for obligation in ihcg.obligations:
        if obligation.obligation_class != "HYPOTHESIS_TARGET": continue
        node = m._parse(obligation.expression)
        if not isinstance(node,ast.Compare) or len(node.ops) != 1:
            raise CanonicalizationError("scaled margin requires an arithmetic comparison")
        polynomial = _polynomial(ast.BinOp(left=node.left,op=ast.Sub(),right=node.comparators[0]))
        for powers,coefficient in polynomial.items():
            if not coefficient: continue
            if any(name not in scales for name,_ in powers): raise CanonicalizationError("scaled nonlinear margin is unbound")
            factors.add(math.prod(scales[name]**power for name,power in powers))
    if len(factors) != 1: raise CanonicalizationError("inconsistent coordinate scales in margin")
    scale = next(iter(factors))
    def convert(error):
        exact = Fraction.from_float(error)*scale
        value = float(exact)
        if math.isfinite(value) and Fraction.from_float(value) < exact: value = math.nextafter(value,math.inf)
        return value
    converted = tuple(convert(error) for error in (epsilon_disc,epsilon_total))
    if not all(math.isfinite(error) for error in converted): raise CanonicalizationError("nonfinite SI uncertainty")
    return converted


def check_intra_vault_uniqueness(candidate_cds: CdsTriplet, ihcg: m.IhcgGraph, verdict: m.EpistemicVerdict,
                                vault_entries: Sequence[VaultEntry], *, epsilon_disc: float, epsilon_total: float,
                                active_audit: m.IdentifiabilityAudit | None = None, significance_audit: m.SignificanceAudit | None = None,
                                compared_run_id: str | None = None) -> VaultUniquenessAudit:
    """Compare trusted verified entries; errors are in canonical margin units."""
    ihcg.assert_sealed()
    candidate_cds = CdsTriplet.model_validate_json(candidate_cds.model_dump_json())
    if epsilon_disc < 0 or epsilon_total < epsilon_disc or not all(map(math.isfinite, (epsilon_disc, epsilon_total))): raise ValueError("invalid novelty error budget")
    entries = sorted((VaultEntry.model_validate_json(entry.model_dump_json()) for entry in vault_entries if entry.verdict != m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED), key=lambda e: e.run_id)
    if not entries: return VaultUniquenessAudit(passed=True, vault_class="NOVEL_FIRST_IN_VAULT")
    admissions, unresolved = [], False
    for entry in entries:
        old = entry.cds
        if old.model_hash != candidate_cds.model_hash: continue
        same_claim = old.claim_hash == candidate_cds.claim_hash
        # ARCH-008 (BUG-027): Distinct / opposite-polarity claim on the same model
        # (e.g. a VERIFIED_FALSIFICATION following a VERIFIED_DISCOVERY with a
        # different claim polarity or environmental slice) is a distinct epistemic
        # finding, never UNIQUENESS_FAILED:UNCLASSIFIED.
        if not same_claim and verdict != entry.verdict and verdict == m.EpistemicVerdict.VERIFIED_FALSIFICATION:
            admissions.append(VaultUniquenessAudit(passed=True, vault_class="NOVEL_INDEPENDENT", related_run_id=f"@research:{entry.run_id}")); continue
        relation = _relations(candidate_cds.pi_domain, old.pi_domain)
        if relation is None: unresolved = True; continue
        subset, overlap, extension = relation
        if same_claim and overlap and verdict == m.EpistemicVerdict.VERIFIED_FALSIFICATION and entry.verdict in {m.EpistemicVerdict.VERIFIED_DISCOVERY, m.EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY}:
            admissions.append(VaultUniquenessAudit(passed=True, vault_class="VALID_REFUTATION_OF_PRIOR", related_run_id=f"@research:{entry.run_id}", lineage_edge="refutes")); continue
        precision = entry.epsilon_disc > 0 and epsilon_disc <= entry.epsilon_disc/10
        boundary = verdict == m.EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY and entry.verdict == m.EpistemicVerdict.VERIFIED_DISCOVERY
        universal_upgrade = json.loads(candidate_cds.domain_and_verdict)["quantifier"] == "FORALL" and json.loads(old.domain_and_verdict)["quantifier"] == "EXISTS"
        strengthening = subset and not same_claim and _strict_strengthening(candidate_cds, old, epsilon_total, entry.epsilon_total)
        if subset and (same_claim and (precision or boundary or universal_upgrade) or strengthening):
            admissions.append(VaultUniquenessAudit(passed=True, vault_class="VALID_BOUND_SHARPENING", related_run_id=f"@research:{entry.run_id}", lineage_edge="sharpens_bounds_of")); continue
        if same_claim and subset and verdict == entry.verdict:
            return VaultUniquenessAudit(passed=False, vault_class="EXACT_ALGEBRAIC_DUPLICATE", related_run_id=f"@research:{entry.run_id}", failure_reason="EXACT_ALGEBRAIC_DUPLICATE")
        if extension:
            admissions.append(VaultUniquenessAudit(passed=True, vault_class="VALID_DOMAIN_EXTENSION", related_run_id=f"@research:{entry.run_id}", lineage_edge="extends_domain_of")); continue
        unresolved = True
    if admissions:
        rank = {"VALID_REFUTATION_OF_PRIOR": 0, "VALID_BOUND_SHARPENING": 1, "VALID_DOMAIN_EXTENSION": 2, "NOVEL_INDEPENDENT": 3}
        return min(admissions, key=lambda a: (rank[a.vault_class], a.related_run_id))
    if all(entry.cds.model_hash != candidate_cds.model_hash for entry in entries):
        if compared_run_id is not None and compared_run_id in {e.run_id for e in entries}:
            if active_audit and active_audit.coupling_passed and significance_audit and significance_audit.passed:
                return VaultUniquenessAudit(passed=True, vault_class="VALID_COUPLED_EXTENSION", related_run_id=f"@research:{compared_run_id}", lineage_edge="coupled_extension_of")
            return VaultUniquenessAudit(passed=False, failure_reason="ACTIVE_COUPLING_AND_PRIOR_BASELINE_SIGNIFICANCE_REQUIRED")
        # ARCH-003 (BUG-008): A model whose canonical family does not exist in the Vault
        # is NOVEL_INDEPENDENT if its claim is also distinct from existing vault entries;
        # if it reuses an existing entry's exact claim_hash over a swapped model without
        # active coupling + baseline significance, reject as uncoupled model substitution.
        if any(entry.cds.canonical_claim == candidate_cds.canonical_claim for entry in entries):
            return VaultUniquenessAudit(passed=False, failure_reason="ACTIVE_COUPLING_AND_PRIOR_BASELINE_SIGNIFICANCE_REQUIRED")
        return VaultUniquenessAudit(passed=True, vault_class="NOVEL_INDEPENDENT")
    return VaultUniquenessAudit(passed=False, failure_reason="PI_DOMAIN_OR_STRICT_STRENGTHENING_UNPROVEN" if unresolved else "NOVELTY_UNPROVEN")


class ReferenceRegistry:
    """Immutable native fingerprints, with explicit legacy annotation gaps.

    Formal packs provide ``canonical_equations`` entries containing native
    TaggedCselEq objects. Legacy LaTeX packs are inventoried but cannot acquire
    invented SI/role metadata merely to make a reference comparison pass.
    """
    def __init__(self, packs: Mapping[str, Any] | None = None):
        hashes, unresolved = {}, []
        from .reference_annotations import typed_reference_packs
        for pack, data in sorted((typed_reference_packs(LEGACY_REFERENCE_REGISTRY) if packs is None else packs).items()):
            for i, equation in enumerate(data.get("canonical_equations", ())):
                name = f"{pack}:{i}"
                try:
                    model = _native_canonical([equation]); digest = json.loads(_native.py_compute_cds_digests(model, "", ""))["model_hash"]
                    hashes[digest] = name
                except ValueError: unresolved.append(name)
            for i, equation in enumerate(data.get("equations", ())):
                if i in data.get("annotated_equation_indices", ()): continue
                # Parsing exposes coverage rather than silently discarding a
                # failed formula or assigning all its symbols dimension zero.
                parse_equation(equation[0], source=pack)
                unresolved.append(f"{pack}:latex:{i}:MISSING_TYPED_SYMBOL_METADATA")
        self.model_hashes = frozenset(hashes)
        self.sources = MappingProxyType(hashes)
        self.unresolved_references = tuple(unresolved)

    @property
    def coverage_complete(self): return not self.unresolved_references


def check_reference_nontriviality(candidate_cds: CdsTriplet, ihcg: m.IhcgGraph, verdict: m.EpistemicVerdict,
                                 ref_registry: ReferenceRegistry, lit_eqs: Sequence[Mapping[str, Any]] = (), *,
                                 significance_audit: m.SignificanceAudit | None = None,
                                 active_audit: m.IdentifiabilityAudit | None = None) -> NonTrivialityAudit:
    ihcg.assert_sealed()
    candidate_cds = CdsTriplet.model_validate_json(candidate_cds.model_dump_json())
    literature = ReferenceRegistry({"literature": {"canonical_equations": list(lit_eqs)}})
    unresolved = (*ref_registry.unresolved_references, *literature.unresolved_references)
    complete = not unresolved
    matched = ref_registry.sources.get(candidate_cds.model_hash) or literature.sources.get(candidate_cds.model_hash)
    exempt = (candidate_cds.equation_count >= 2 or candidate_cds.stage_count >= 2
              or verdict in {m.EpistemicVerdict.VERIFIED_FALSIFICATION, m.EpistemicVerdict.VERIFIED_BOUNDARY_DISCOVERY}
              or significance_audit is not None and significance_audit.passed and significance_audit.method in {"PARETO_EPS_DOMINANCE", "MC_EMPIRICAL_BERNSTEIN", "MC_CLOPPER_PEARSON"})
    reason = None if exempt else "TRIVIAL_TEXTBOOK_PLAYBACK" if matched else "REFERENCE_COVERAGE_INCOMPLETE" if not complete else None
    invariant_slots = [s for s in ihcg.slots if s.semantic_tag == m.ZERO_DISSIPATION_INVARIANT]
    boundary_flux = any(s.semantic_tag == m.BOUNDARY_FLUX for s in ihcg.slots)
    closure = (boundary_flux and len(invariant_slots) >= 2 and significance_audit is not None and significance_audit.passed
        and active_audit is not None and active_audit.coupling_passed
        and {s.symbol for s in invariant_slots} <= set(active_audit.excluded_certified_invariants))
    return NonTrivialityAudit(passed=reason is None, reason=reason, matched_reference=matched, reference_coverage_complete=complete,
        unresolved_references=unresolved,shbt_open_boundary_closure=closure)


def combine_uniqueness_audits(cds: CdsTriplet, vault: VaultUniquenessAudit, reference: NonTrivialityAudit) -> m.UniquenessAudit:
    if vault.vault_class is None: raise ValueError(vault.failure_reason or "unresolved Vault classification")
    return m.UniquenessAudit(passed=vault.passed and reference.passed, cds_model_hash=cds.model_hash, cds_claim_hash=cds.claim_hash,
        cds_full_hash=cds.full_hash, vault_class=vault.vault_class, related_run_id=vault.related_run_id,
        reference_nontrivial=reference.passed, shbt_open_boundary_closure=reference.shbt_open_boundary_closure)


def load_verified_vault_entries(vault) -> tuple[VaultEntry, ...]:
    entries = []
    for metadata in vault.list_runs():
        path = vault.run_dir(metadata["run_id"])/"meaningfulness_certificate.json"
        if not path.is_file(): continue
        from .epistemic import parse_run_certificate
        certificate = parse_run_certificate(path.read_text(encoding="utf-8"))
        if certificate.run_id != metadata["run_id"]: raise ValueError("Vault run identity mismatch")
        if certificate.epistemic_verdict == m.EpistemicVerdict.INCONCLUSIVE_UNVERIFIED: continue
        cds = CdsTriplet.model_validate_json(m._json(metadata["canonical_signature"]))
        if (cds.model_hash, cds.claim_hash, cds.full_hash) != (certificate.uniqueness_audit.cds_model_hash, certificate.uniqueness_audit.cds_claim_hash, certificate.uniqueness_audit.cds_full_hash):
            raise ValueError("Vault signature is not the committed certificate signature")
        disc,total = certificate.fidelity_audit.epsilon_disc,certificate.fidelity_audit.epsilon_total
        blueprint_path = vault.run_dir(metadata["run_id"])/"blueprint.json"
        if blueprint_path.is_file():
            from ..schemas.research import ResearchBlueprint
            from .epistemic import compile_blueprint_ihcg
            # AUDIT-PATCH-006: vault blueprints are UTF-8; cp1252 decoding poisoned every later
            # run on Windows once any verified entry contained non-ASCII text.
            blueprint = ResearchBlueprint.model_validate_json(blueprint_path.read_text(encoding="utf-8"))
            pack = blueprint.ihcg_evidence or {}
            if any(s.get("unit_scale",[1,1]) != [1,1] for e in [pack,*pack.get("extensions",())] for s in e.get("evidence",{}).get("slots",())):
                graph = compile_blueprint_ihcg(blueprint,stage=cds.stage_count)
                disc,total = si_error_budgets(graph,disc,total)
        entries.append(VaultEntry(run_id=certificate.run_id, cds=cds, verdict=certificate.epistemic_verdict,
                                 epsilon_total=total, epsilon_disc=disc))
    return tuple(entries)
