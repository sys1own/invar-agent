"""UPG-004: Structured PDE/lattice StateFamily expansion (Dirichlet, Neumann, Periodic)."""
from __future__ import annotations

import itertools
import math
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

MAX_VERIFIER_VARIABLES = 32


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundarySpec(_Strict):
    axis: int = Field(ge=0, le=2)
    side: Literal["LOW", "HIGH"]
    kind: Literal["DIRICHLET", "NEUMANN", "PERIODIC"]
    value_text: Optional[str] = None


class StateFamily(_Strict):
    name: str = Field(min_length=1, max_length=32)
    shape: List[int] = Field(min_length=1, max_length=3)
    length_symbols: List[str] = Field(min_length=1, max_length=3)
    coordinate_symbols: List[str] = Field(min_length=1, max_length=3)
    initial_text: str
    rhs_text: str
    boundaries: List[BoundarySpec] = Field(default_factory=list)


def _node_name(family_name: str, idx: Tuple[int, ...]) -> str:
    return f"{family_name}__{'_'.join(str(i) for i in idx)}"


def expand_state_family(
    family: StateFamily,
    *,
    length_values: Mapping[str, float],
    existing_symbol_count: int = 0,
    initial_env: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Deterministically expand a `StateFamily` in row-major order into scalar SciIR
    symbol specs and RHS equations.

    - Dirichlet end nodes become `CONSTANT` slots (avoids zero RHS).
    - Neumann end nodes use ghost values `u[-1] = u[1] - 2*h*g` / `u[N] = u[N-2] + 2*h*g`.
    - Periodic axes wrap modulo `N` with `h = L / N`; non-periodic use `h = L / (N - 1)`.
    - Enforces `len(slots) == prod(shape)` (`COMPILER_INVARIANT`) and total variables <= 32
      (`STATE_FAMILY_EXCEEDS_VERIFIER_BUDGET`).
    """
    ndim = len(family.shape)
    if len(family.length_symbols) != ndim or len(family.coordinate_symbols) != ndim:
        raise ValueError("COMPILER_INVARIANT: StateFamily shape/length/coordinate arity mismatch")
    total_nodes = math.prod(family.shape)
    if total_nodes <= 0:
        raise ValueError("COMPILER_INVARIANT: StateFamily shape must be positive")
    if existing_symbol_count + total_nodes > MAX_VERIFIER_VARIABLES:
        raise ValueError(
            f"STATE_FAMILY_EXCEEDS_VERIFIER_BUDGET: {existing_symbol_count + total_nodes} > {MAX_VERIFIER_VARIABLES}"
        )

    # Index boundaries by (axis, side)
    bc_map: Dict[Tuple[int, str], BoundarySpec] = {}
    for bc in family.boundaries:
        bc_map[(bc.axis, bc.side)] = bc

    hs: List[float] = []
    h_exprs: List[str] = []
    for axis in range(ndim):
        N = family.shape[axis]
        L_sym = family.length_symbols[axis]
        L_val = float(length_values[L_sym])
        is_periodic = (
            (axis, "LOW") in bc_map and bc_map[(axis, "LOW")].kind == "PERIODIC"
        ) or (
            (axis, "HIGH") in bc_map and bc_map[(axis, "HIGH")].kind == "PERIODIC"
        )
        if is_periodic:
            h = L_val / float(N)
            h_exprs.append(f"({L_sym} / {N})")
        else:
            if N < 2:
                raise ValueError("COMPILER_INVARIANT: non-periodic axis requires N >= 2")
            h = L_val / float(N - 1)
            h_exprs.append(f"({L_sym} / {N - 1})")
        hs.append(h)

    env_base = dict(initial_env or {})
    env_base.update({k: float(v) for k, v in length_values.items()})

    def _eval_numeric(expr_s: str, extra: Mapping[str, float]) -> float:
        ns = {"__builtins__": {}, "math": math, "sin": math.sin, "cos": math.cos, "exp": math.exp, "pi": math.pi}
        ns.update(env_base)
        ns.update(extra)
        return float(eval(expr_s, ns, {}))

    slots: List[Dict[str, Any]] = []
    equations: List[Dict[str, Any]] = []
    seen_targets: set[str] = set()

    for idx in itertools.product(*(range(n) for n in family.shape)):
        sname = _node_name(family.name, idx)
        coords = {family.coordinate_symbols[a]: idx[a] * hs[a] for a in range(ndim)}

        # Check if node lies on a Dirichlet boundary
        dirichlet_val: Optional[float] = None
        for axis in range(ndim):
            if idx[axis] == 0 and (axis, "LOW") in bc_map and bc_map[(axis, "LOW")].kind == "DIRICHLET":
                dirichlet_val = _eval_numeric(bc_map[(axis, "LOW")].value_text or "0.0", coords)
                break
            if idx[axis] == family.shape[axis] - 1 and (axis, "HIGH") in bc_map and bc_map[(axis, "HIGH")].kind == "DIRICHLET":
                dirichlet_val = _eval_numeric(bc_map[(axis, "HIGH")].value_text or "0.0", coords)
                break

        if dirichlet_val is not None:
            slots.append({
                "name": sname,
                "role": "CONSTANT",
                "initial_or_default": dirichlet_val,
                "coords": coords,
            })
            continue

        init_val = _eval_numeric(family.initial_text, coords)
        slots.append({
            "name": sname,
            "role": "STATE",
            "initial_or_default": init_val,
            "coords": coords,
        })

        # Build neighbor expressions per axis for lap / ddx / ddy / ddz
        def _neighbor(axis: int, delta: int) -> str:
            N = family.shape[axis]
            i = idx[axis]
            j = i + delta
            if 0 <= j < N:
                nidx = list(idx)
                nidx[axis] = j
                return _node_name(family.name, tuple(nidx))
            side = "LOW" if j < 0 else "HIGH"
            bc = bc_map.get((axis, side))
            if bc is not None and bc.kind == "PERIODIC":
                nidx = list(idx)
                nidx[axis] = j % N
                return _node_name(family.name, tuple(nidx))
            if bc is not None and bc.kind == "NEUMANN":
                g = bc.value_text or "0.0"
                # Ghost: u[-1] = u[1] - 2*h*g (LOW), u[N] = u[N-2] + 2*h*g (HIGH)
                mirror_i = 1 if side == "LOW" else N - 2
                nidx = list(idx)
                nidx[axis] = mirror_i
                mname = _node_name(family.name, tuple(nidx))
                sign = "-" if side == "LOW" else "+"
                return f"({mname} {sign} 2.0 * {h_exprs[axis]} * ({g}))"
            raise ValueError(f"COMPILER_INVARIANT: missing boundary spec for axis={axis} side={side}")

        lap_terms: List[str] = []
        deriv_terms: Dict[int, str] = {}
        for axis in range(ndim):
            um = _neighbor(axis, -1)
            up = _neighbor(axis, +1)
            h_e = h_exprs[axis]
            lap_terms.append(f"(({um} - 2.0 * {sname} + {up}) / ({h_e} ** 2))")
            deriv_terms[axis] = f"(({up} - {um}) / (2.0 * {h_e}))"

        lap_expr = "(" + " + ".join(lap_terms) + ")"
        rhs = family.rhs_text
        rhs = re.sub(rf"\blap\(\s*{re.escape(family.name)}\s*\)", lap_expr, rhs)
        for axis, opname in enumerate(("ddx", "ddy", "ddz")[:ndim]):
            rhs = re.sub(rf"\b{opname}\(\s*{re.escape(family.name)}\s*\)", deriv_terms[axis], rhs)
        rhs = re.sub(rf"\b{re.escape(family.name)}\b", sname, rhs)

        if sname in seen_targets:
            raise ValueError(f"COMPILER_INVARIANT: duplicate equation target {sname!r}")
        seen_targets.add(sname)
        equations.append({"target": sname, "expr_text": rhs})

    if len(slots) != total_nodes:
        raise ValueError(f"COMPILER_INVARIANT: expanded {len(slots)} slots != prod(shape) {total_nodes}")
    return {"slots": slots, "equations": equations, "h": hs}
