"""SI reference annotations; roles describe each law used as a stand-alone model.

Order is M,L,T,I,Theta,N,J. Inputs are held controls, outputs are pure derived
observables, and conserved population frequencies are dynamic states. Unspecified
semantic tags stay real (0); no entropy/probability tag is inferred from LaTeX.
Only positive physical branches are used for dimensional Pi denominators.
"""
from __future__ import annotations

import copy

from . import meaningfulness as m

ONE = (0,0,0,0,0,0,0)
MASS = (1,0,0,0,0,0,0)
LENGTH = (0,1,0,0,0,0,0)
TIME = (0,0,1,0,0,0,0)
RATE = (0,0,-1,0,0,0,0)
VELOCITY = (0,1,-1,0,0,0,0)
FLUX = (1,0,-3,0,0,0,0)
TEMPERATURE = (0,0,0,0,1,0,0)

# Derivative identities lacking explicit nondimensionalization, generic
# covariance/payoff units, fractional SI constants and implicit log bases or
# dimensional coefficients remain unannotated in their original packs.
ANNOTATIONS = {
    ("fundamental_constants", 0): ("E-m*c**2", "E", {"E": (1,2,-2,0,0,0,0), "m": MASS, "c": VELOCITY}),
    ("fundamental_constants", 1): ("F-G*m1*m2/r**2", "F", {"F": (1,1,-2,0,0,0,0), "G": (-1,3,-2,0,0,0,0), "m1": MASS, "m2": MASS, "r": LENGTH}),
    ("fundamental_constants", 2): ("q_rad-epsilon*sigma_SB*T**4", "q_rad", {"q_rad": FLUX, "epsilon": ONE, "sigma_SB": (1,0,-3,0,-4,0,0), "T": TEMPERATURE}),
    ("aerospace_entry", 1): ("rho-rho0*exp(-h/H)", "rho", {"rho": (1,-3,0,0,0,0,0), "rho0": (1,-3,0,0,0,0,0), "h": LENGTH, "H": LENGTH}),
    ("relativistic_plasma", 0): ("gamma-1/sqrt(1-v**2/c**2)", "gamma", {"gamma": ONE, "v": VELOCITY, "c": VELOCITY}),
    ("relativistic_plasma", 1): ("r_L-gamma*m*v/(q*B)", "r_L", {"r_L": LENGTH, "gamma": ONE, "m": MASS, "v": VELOCITY, "q": (0,0,1,1,0,0,0), "B": (1,0,-2,-1,0,0,0)}),
    ("relativistic_plasma", 2): ("F_drag-rho*v**2*C_D*A/2", "F_drag", {"F_drag": (1,1,-2,0,0,0,0), "rho": (1,-3,0,0,0,0,0), "v": VELOCITY, "C_D": ONE, "A": (0,2,0,0,0,0,0)}),
    ("nonlinear_dynamics", 6): ("C-u*delta_t/delta_x", "C", {"C": ONE, "u": VELOCITY, "delta_t": TIME, "delta_x": LENGTH}),
    ("epidemiology_population", 0): ("R0-beta/gamma", "R0", {"R0": ONE, "beta": RATE, "gamma": RATE}),
    ("epidemiology_population", 1): ("p_c-1+1/R0", "p_c", {"p_c": ONE, "R0": ONE}),
    ("epidemiology_population", 5): ("R_eff-beta_eff*S/(gamma*N)", "R_eff", {"R_eff": ONE, "beta_eff": RATE, "gamma": RATE, "S": ONE, "N": ONE}),
    ("pharmacology_biochemistry", 2): ("CL-k_el*V_d", "CL", {"CL": (0,3,-1,0,0,0,0), "k_el": RATE, "V_d": (0,3,0,0,0,0,0)}),
    ("genetics_evolution", 0): ("p+q-1", None, {"p": ONE, "q": ONE}),
    ("genetics_evolution", 1): ("f_AA-p**2", "f_AA", {"f_AA": ONE, "p": ONE}),
    ("genetics_evolution", 2): ("f_Aa-2*p*q", "f_Aa", {"f_Aa": ONE, "p": ONE, "q": ONE}),
    ("climate_earth_systems", 2): ("M_total-M_atm-M_bio-M_surf-M_deep", "M_total", {s: MASS for s in ("M_total", "M_atm", "M_bio", "M_surf", "M_deep")}),
}


def typed_reference_packs(packs):
    result = copy.deepcopy(packs)
    for (pack, index), (expression, output, dimensions) in ANNOTATIONS.items():
        if pack not in result or index >= len(result[pack].get("equations", ())): continue
        states = set(dimensions) if output is None else ({"p", "q"} & dimensions.keys() if pack == "genetics_evolution" else set())
        tags = {s: {"si_dim": list(dim), "semantic_tag": 0, "unit_scale": [1,1],
                    "role": "DYNAMIC_STATE" if s in states else "DERIVED_OBSERVABLE" if s == output else "CONTROL_PARAM",
                    "positive": s not in states} for s, dim in dimensions.items()}
        equation = {"expr": m._json(m._csel(m._parse(expression))), "symbols": tags}
        result[pack].setdefault("canonical_equations", []).append(equation)
        result[pack].setdefault("annotated_equation_indices", []).append(index)
    return result
