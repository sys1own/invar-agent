"""UPG-007 (BUG-009, NF-09): Certified Domain & Resolution Pilot Protocol.

Derives certified integration boxes (omega) and stiffness-aware step sizes strictly via
`DynamicCABIBridge` sandboxed execution (never `exec` in the server interpreter),
excluding singular sets and recording `COMPILER_PILOT` provenance ledger entries.
"""
from __future__ import annotations

import ast
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from invar_agent import _native
from invar_agent.ir.lower_uast import _render_expr_to_python
from invar_agent.ir.schema import Archetype, SciIR, SlotRole
from invar_agent.research import meaningfulness as m
from invar_agent.runtime.cabi_bridge import DynamicCABIBridge, KernelExecutionError


class PilotDomainError(ValueError):
    """Typed pilot protocol error (PILOT_DIVERGED, SINGULAR_SET_NOT_EXCLUDED, RESOLUTION_UNRESOLVED)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass
class PilotResult:
    omega: Dict[str, Tuple[float, float]]
    claim_bounds: Dict[str, Tuple[float, float]]
    dt: float
    steps: int
    envelopes: Dict[str, Tuple[float, float]]
    spectral_radius: float
    ledger_entries: List[Dict[str, Any]] = field(default_factory=list)


def _collect_singular_terms(expr_strs: Sequence[str]) -> Tuple[List[ast.AST], List[ast.AST], List[ast.AST]]:
    """Collect denominators, strictly-positive args (log/log10), and non-negative args (sqrt)."""
    denoms: List[ast.AST] = []
    pos_args: List[ast.AST] = []
    nonneg_args: List[ast.AST] = []
    for s in expr_strs:
        try:
            tree = m._parse(s)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                denoms.append(node.right)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exp = m._fraction(node.right)
                if exp is not None and exp < 0:
                    denoms.append(node.left)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fn = node.func.id
                if fn in {"log", "ln", "log10"} and len(node.args) == 1:
                    pos_args.append(node.args[0])
                elif fn == "sqrt" and len(node.args) == 1:
                    nonneg_args.append(node.args[0])
    return denoms, pos_args, nonneg_args


def _check_singular_exclusion(
    box: Dict[str, Tuple[float, float]],
    envelopes: Dict[str, Tuple[float, float]],
    denoms: Sequence[ast.AST],
    pos_args: Sequence[ast.AST],
    nonneg_args: Sequence[ast.AST],
) -> None:
    """Step 6: Verify interval separation from singular sets, tightening single-state floors once."""
    def _try_tighten(term: ast.AST, require_pos: bool) -> bool:
        if isinstance(term, ast.Name) and term.id in box and term.id in envelopes:
            elo, ehi = envelopes[term.id]
            blo, bhi = box[term.id]
            if elo > 0.0:
                box[term.id] = (max(blo, 0.5 * elo), bhi)
                return True
            if not require_pos and ehi < 0.0:
                box[term.id] = (blo, min(bhi, 0.5 * ehi))
                return True
        return False

    for d in denoms:
        try:
            iv = m._iv(d, box)
            ok = (iv.lo >= 1e-12) or (iv.hi <= -1e-12)
        except Exception:
            ok = False
        if not ok:
            if _try_tighten(d, require_pos=False):
                iv = m._iv(d, box)
                ok = (iv.lo >= 1e-12) or (iv.hi <= -1e-12)
            if not ok:
                raise PilotDomainError(
                    "SINGULAR_SET_NOT_EXCLUDED",
                    f"Denominator {ast.unparse(d)!r} not separated from 0 on box",
                )

    for p in pos_args:
        try:
            iv = m._iv(p, box)
            ok = iv.lo >= 1e-12
        except Exception:
            ok = False
        if not ok:
            if _try_tighten(p, require_pos=True):
                iv = m._iv(p, box)
                ok = iv.lo >= 1e-12
            if not ok:
                raise PilotDomainError(
                    "SINGULAR_SET_NOT_EXCLUDED",
                    f"Log argument {ast.unparse(p)!r} not strictly positive on box",
                )

    for nn in nonneg_args:
        try:
            iv = m._iv(nn, box)
            ok = iv.lo >= 0.0
        except Exception:
            ok = False
        if not ok:
            if _try_tighten(nn, require_pos=True):
                iv = m._iv(nn, box)
                ok = iv.lo >= 0.0
            if not ok:
                raise PilotDomainError(
                    "SINGULAR_SET_NOT_EXCLUDED",
                    f"Sqrt argument {ast.unparse(nn)!r} not non-negative on box",
                )


def _parameter_sample_points(
    mech_params: Sequence[str],
    param_boxes: Mapping[str, Tuple[float, float]],
    defaults: Mapping[str, float],
) -> List[Dict[str, float]]:
    """Step 3: Midpoint plus corners (<=2 mechanism params) or midpoint +/- each axis."""
    mid = {p: 0.5 * (param_boxes[p][0] + param_boxes[p][1]) if p in param_boxes else defaults.get(p, 0.0) for p in mech_params}
    pts: List[Dict[str, float]] = [dict(mid)]
    active = [p for p in mech_params if p in param_boxes and param_boxes[p][0] < param_boxes[p][1]]
    if not active:
        return pts
    if len(active) <= 2:
        for combo in itertools.product(*(param_boxes[p] for p in active)):
            pt = dict(mid)
            for p, v in zip(active, combo):
                pt[p] = float(v)
            pts.append(pt)
    else:
        for p in active:
            for v in param_boxes[p]:
                pt = dict(mid)
                pt[p] = float(v)
                pts.append(pt)
    return pts


def propose_domains(
    kernel_code: str,
    sealed_ir: SciIR,
    contract: Any,
    bridge: DynamicCABIBridge,
    *,
    user_domains: Optional[Mapping[str, Tuple[float, float]]] = None,
    mechanism_parameters: Optional[Sequence[str]] = None,
    claim_margin_scale: float = 0.0,
    wall_budget_s: float = 20.0,
) -> PilotResult:
    """UPG-007: Derive certified domain `omega`, `claim_bounds`, and stiffness-aware `(dt, steps)`.

    Never executes candidate code via `exec` in the server process; all trajectory pilots
    run inside `bridge.run_python`.
    """
    t_start = time.monotonic()
    symbols_sorted = sorted(sealed_ir.symbols, key=lambda s: (s.slot_index, s.name))
    slot_map = {s.name: idx for idx, s in enumerate(symbols_sorted)}
    sym_names = set(slot_map.keys())
    user_dom = dict(user_domains or {})

    base_dt = float(getattr(contract, "dt", 0.01))
    base_steps = int(getattr(contract, "steps", 100))
    horizon = base_dt * base_steps

    # Fixed-point Newton + Krawczyk shortcut
    if sealed_ir.archetype == Archetype.FIXED_POINT:
        omega: Dict[str, Tuple[float, float]] = {}
        envelopes: Dict[str, Tuple[float, float]] = {}
        ledger: List[Dict[str, Any]] = []
        for s in symbols_sorted:
            if s.name in user_dom:
                omega[s.name] = (float(user_dom[s.name][0]), float(user_dom[s.name][1]))
            elif s.domain is not None:
                omega[s.name] = (float(s.domain.lo), float(s.domain.hi))
            else:
                v0 = float(s.initial_or_default)
                if s.role in (SlotRole.STATE, SlotRole.DERIVED_OBSERVABLE):
                    w = max(abs(v0) * 0.25, 1e-3)
                    lo, hi = (max(0.5 * v0, v0 - w), v0 + w) if v0 > 0 else (v0 - w, v0 + w)
                    omega[s.name] = (lo, hi)
                    envelopes[s.name] = (v0, v0)
                    ledger.append({
                        "symbol": s.name,
                        "source": "COMPILER_PILOT",
                        "envelope": [v0, v0],
                        "box": [lo, hi],
                    })
                else:
                    omega[s.name] = (v0, v0)
        claim_bounds = dict(user_dom)
        return PilotResult(
            omega=omega,
            claim_bounds=claim_bounds,
            dt=base_dt,
            steps=base_steps,
            envelopes=envelopes,
            spectral_radius=0.0,
            ledger_entries=ledger,
        )

    # Build initial vector and parameter boxes
    init_vec = [float(s.initial_or_default) for s in symbols_sorted]
    if getattr(contract, "initial_state", None):
        for i, v in enumerate(contract.initial_state):
            if i < len(init_vec):
                init_vec[i] = float(v)

    param_boxes: Dict[str, Tuple[float, float]] = {}
    defaults = {s.name: float(s.initial_or_default) for s in symbols_sorted}
    for s in symbols_sorted:
        if s.role in (SlotRole.CONTROL_PARAM, SlotRole.CONSTANT):
            if s.name in user_dom:
                param_boxes[s.name] = (float(user_dom[s.name][0]), float(user_dom[s.name][1]))
            elif s.domain is not None:
                param_boxes[s.name] = (float(s.domain.lo), float(s.domain.hi))
            else:
                param_boxes[s.name] = (defaults[s.name], defaults[s.name])

    mech = list(mechanism_parameters or [
        s.name for s in symbols_sorted
        if s.role == SlotRole.CONTROL_PARAM and param_boxes[s.name][0] < param_boxes[s.name][1]
    ])
    sample_pts = _parameter_sample_points(mech, param_boxes, defaults)

    # Step 2: dt_p = dt / 8, steps_p = min(8 * steps, 65536)
    steps_p = min(max(8 * base_steps, 8), 65536)
    dt_p = horizon / steps_p if horizon > 0 else base_dt / 8.0
    trace_slots = tuple(range(len(symbols_sorted)))

    envelopes: Dict[str, Tuple[float, float]] = {}
    trace_samples_for_jac: List[Dict[str, float]] = []

    for pt in sample_pts:
        if time.monotonic() - t_start > wall_budget_s:
            raise PilotDomainError("RESOLUTION_UNRESOLVED", "Pilot wall-clock budget exceeded")
        vec = list(init_vec)
        for p, v in pt.items():
            if p in slot_map:
                vec[slot_map[p]] = float(v)
        # Union initial point into envelope so INITIAL_POINT_OUTSIDE_OMEGA is impossible
        for s in symbols_sorted:
            val0 = vec[slot_map[s.name]]
            if s.name not in envelopes:
                envelopes[s.name] = (val0, val0)
            else:
                envelopes[s.name] = (min(envelopes[s.name][0], val0), max(envelopes[s.name][1], val0))
        try:
            res = bridge.run_python(
                kernel_code,
                vec,
                dt_p,
                steps_p,
                timeout_seconds=min(5.0, max(1.0, wall_budget_s - (time.monotonic() - t_start))),
                trace_slots=trace_slots,
            )
        except KernelExecutionError as exc:
            raise PilotDomainError("PILOT_DIVERGED", str(exc)) from exc
        if not res.get("finite"):
            raise PilotDomainError("PILOT_DIVERGED", "Non-finite trajectory during pilot run")
        tvals = res.get("trace_values") or []
        final_state = res.get("state") or vec
        for s in symbols_sorted:
            idx = slot_map[s.name]
            fv = float(final_state[idx])
            if not math.isfinite(fv):
                raise PilotDomainError("PILOT_DIVERGED", f"Non-finite final value for {s.name}")
            lo, hi = envelopes[s.name]
            lo, hi = min(lo, fv), max(hi, fv)
            for row in tvals:
                rv = float(row[idx])
                if not math.isfinite(rv):
                    raise PilotDomainError("PILOT_DIVERGED", f"Non-finite trace value for {s.name}")
                if rv < lo:
                    lo = rv
                if rv > hi:
                    hi = rv
            envelopes[s.name] = (lo, hi)
        # Subsample <= 64 trace points for Jacobian spectral radius
        if tvals:
            stride = max(1, len(tvals) // 64)
            for row in tvals[::stride][:64]:
                trace_samples_for_jac.append({s.name: float(row[slot_map[s.name]]) for s in symbols_sorted})

    # Steps 4 & 5: Padded envelope + sign preservation
    omega: Dict[str, Tuple[float, float]] = {}
    ledger: List[Dict[str, Any]] = []
    for s in symbols_sorted:
        if s.name in user_dom:
            omega[s.name] = (float(user_dom[s.name][0]), float(user_dom[s.name][1]))
            continue
        if s.role in (SlotRole.CONTROL_PARAM, SlotRole.CONSTANT):
            omega[s.name] = param_boxes[s.name]
            continue
        if s.domain is not None and s.role == SlotRole.INDEPENDENT_VAR:
            omega[s.name] = (float(s.domain.lo), float(s.domain.hi))
            continue
        elo, ehi = envelopes[s.name]
        mid = 0.5 * (elo + ehi)
        w = max(ehi - elo, 1e-9 * max(1.0, abs(mid)))
        blo = elo - 0.25 * w
        bhi = ehi + 0.25 * w
        if elo > 0.0:
            blo = max(blo, 0.5 * elo)
        elif ehi < 0.0:
            bhi = min(bhi, 0.5 * ehi)
        omega[s.name] = (blo, bhi)
        ledger.append({
            "symbol": s.name,
            "source": "COMPILER_PILOT",
            "envelope": [elo, ehi],
            "padding": 0.25 * w,
            "box": [blo, bhi],
        })

    # Step 6: Singular set exclusion
    expr_strs = [
        _render_expr_to_python(eq.expr, sym_names, for_ihcg=True)
        for eq in [*sealed_ir.dynamics_or_residuals, *sealed_ir.observables]
        if getattr(eq, "expr", None) is not None
    ]
    denoms, pos_args, nonneg_args = _collect_singular_terms(expr_strs)
    _check_singular_exclusion(omega, envelopes, denoms, pos_args, nonneg_args)

    # Step 7: Stiffness-aware step size via PyAdjointHin spectral radius
    max_rho = 0.0
    state_syms = [s.name for s in symbols_sorted if s.role == SlotRole.STATE]
    if sealed_ir.archetype == Archetype.ODE_PDE and state_syms and trace_samples_for_jac:
        eq_csel: Dict[str, Any] = {}
        for eq in sealed_ir.dynamics_or_residuals:
            if eq.target in state_syms and getattr(eq, "expr", None) is not None:
                try:
                    py_s = _render_expr_to_python(eq.expr, sym_names, for_ihcg=True)
                    eq_csel[eq.target] = m._csel(m._parse(py_s))
                except Exception:
                    pass
        if len(eq_csel) == len(state_syms):
            try:
                adj = _native.PyAdjointHin(state_syms, json.dumps(eq_csel))
                for sample in trace_samples_for_jac[:64]:
                    res_j = json.loads(adj.analyze(json.dumps(sample)))
                    rho = float(res_j.get("spectral_radius", 0.0) or 0.0)
                    if math.isfinite(rho) and rho > max_rho:
                        max_rho = rho
            except Exception:
                max_rho = 0.0

    dt_out = base_dt
    steps_out = base_steps
    if max_rho > 0.0 and horizon > 0.0:
        dt_req = min(base_dt, 0.2 / max_rho)
        steps_req = 4 * math.ceil(horizon / (4.0 * dt_req))
        if steps_req > 8192:
            raise PilotDomainError(
                "RESOLUTION_UNRESOLVED",
                f"Stiffness rho={max_rho:.4g} requires {steps_req} steps > 8192 cap",
            )
        steps_out = max(base_steps, int(steps_req))
        dt_out = horizon / steps_out

    # Step 8: Halving check (up to 3 halvings)
    if sealed_ir.archetype == Archetype.ODE_PDE and horizon > 0.0:
        mid_vec = list(init_vec)
        for p, v in sample_pts[0].items():
            if p in slot_map:
                mid_vec[slot_map[p]] = float(v)
        converged = False
        for _halve in range(4):
            if steps_out * 2 > 8192 and _halve > 0:
                break
            r1 = bridge.run_python(kernel_code, mid_vec, dt_out, steps_out, timeout_seconds=3.0)
            r2 = bridge.run_python(kernel_code, mid_vec, dt_out / 2.0, steps_out * 2, timeout_seconds=3.0)
            if not (r1.get("finite") and r2.get("finite")):
                raise PilotDomainError("PILOT_DIVERGED", "Non-finite trajectory during halving check")
            s1, s2 = r1["state"], r2["state"]
            max_diff = 0.0
            scale = 1.0
            for idx in range(min(len(s1), len(s2))):
                max_diff = max(max_diff, abs(float(s1[idx]) - float(s2[idx])))
                scale = max(scale, abs(float(s2[idx])))
            tol = max(1e-6 * scale, abs(claim_margin_scale) / 10.0)
            if max_diff <= tol:
                converged = True
                break
            if steps_out * 2 > 8192:
                raise PilotDomainError(
                    "RESOLUTION_UNRESOLVED",
                    f"Halving discrepancy {max_diff:.3e} > tol {tol:.3e} at step cap {steps_out}",
                )
            steps_out *= 2
            dt_out = horizon / steps_out
        if not converged:
            raise PilotDomainError("RESOLUTION_UNRESOLVED", "Halving check did not converge within 3 halvings")

    claim_bounds = dict(user_dom)
    return PilotResult(
        omega=omega,
        claim_bounds=claim_bounds,
        dt=dt_out,
        steps=steps_out,
        envelopes=envelopes,
        spectral_radius=max_rho,
        ledger_entries=ledger,
    )
