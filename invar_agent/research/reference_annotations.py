"""SI reference annotations; roles describe each law used as a stand-alone model.

Order is M,L,T,I,Theta,N,J. Inputs are held controls, outputs are pure derived
observables, and conserved population frequencies are dynamic states. Unspecified
semantic tags stay real (0); no entropy/probability tag is inferred from LaTeX.
Only positive physical branches are used for dimensional Pi denominators.
"""
from __future__ import annotations

import copy

from . import meaningfulness as m

ONE = (0, 0, 0, 0, 0, 0, 0)
MASS = (1, 0, 0, 0, 0, 0, 0)
LENGTH = (0, 1, 0, 0, 0, 0, 0)
TIME = (0, 0, 1, 0, 0, 0, 0)
RATE = (0, 0, -1, 0, 0, 0, 0)
VELOCITY = (0, 1, -1, 0, 0, 0, 0)
ACCEL = (0, 1, -2, 0, 0, 0, 0)
AREA = (0, 2, 0, 0, 0, 0, 0)
VOLUME = (0, 3, 0, 0, 0, 0, 0)
DENSITY = (1, -3, 0, 0, 0, 0, 0)
FORCE = (1, 1, -2, 0, 0, 0, 0)
ENERGY = (1, 2, -2, 0, 0, 0, 0)
POWER = (1, 2, -3, 0, 0, 0, 0)
FLUX = (1, 0, -3, 0, 0, 0, 0)
TEMPERATURE = (0, 0, 0, 0, 1, 0, 0)
CONC = (0, -3, 0, 0, 0, 1, 0)
CONC_RATE = (0, -3, -1, 0, 0, 1, 0)

# Complete 56-equation SI-typed reference registry (all 40 Round 4 §5.1 entries annotated in Sci order M,L,T,I,Theta,N,J).
# Format: (pack, index): (expression, output_or_None, {symbol: si_dim})
ANNOTATIONS = {
    # --- fundamental_constants (3) ---
    ("fundamental_constants", 0): ("E-m*c**2", "E", {"E": ENERGY, "m": MASS, "c": VELOCITY}),
    ("fundamental_constants", 1): ("F-G*m1*m2/r**2", "F", {"F": FORCE, "G": (-1, 3, -2, 0, 0, 0, 0), "m1": MASS, "m2": MASS, "r": LENGTH}),
    ("fundamental_constants", 2): ("q_rad-epsilon*sigma_SB*T**4", "q_rad", {"q_rad": FLUX, "epsilon": ONE, "sigma_SB": (1, 0, -3, 0, -4, 0, 0), "T": TEMPERATURE}),

    # --- aerospace_entry (5) ---
    ("aerospace_entry", 0): ("q_s**2*R_n-k2*rho*v**6", "q_s", {"q_s": FLUX, "R_n": LENGTH, "k2": (1, -2, 0, 0, 0, 0, 0), "rho": DENSITY, "v": VELOCITY}),
    ("aerospace_entry", 1): ("rho-rho0*exp(-h/H)", "rho", {"rho": DENSITY, "rho0": DENSITY, "h": LENGTH, "H": LENGTH}),
    ("aerospace_entry", 2): ("-0.5*rho*v**2*C_D*A/m-g*sin(gam)", "v", {"rho": DENSITY, "v": VELOCITY, "C_D": ONE, "A": AREA, "m": MASS, "g": ACCEL, "gam": ONE}),
    ("aerospace_entry", 3): ("v*sin(gam)", "h", {"v": VELOCITY, "gam": ONE}),
    ("aerospace_entry", 4): ("a_peak-v_e**2*sin(gam_e)/(2*exp(1)*H)", "a_peak", {"a_peak": ACCEL, "v_e": VELOCITY, "gam_e": ONE, "H": LENGTH}),

    # --- chemical_kinetics (5) ---
    ("chemical_kinetics", 0): ("k_r-A*exp(-E_a/(R*T))", "k_r", {"k_r": RATE, "A": RATE, "E_a": (1, 2, -2, 0, 0, -1, 0), "R": (1, 2, -2, 0, -1, -1, 0), "T": TEMPERATURE}),
    ("chemical_kinetics", 1): ("-k1*y1+k2*y2*y3", "y1", {"k1": RATE, "k2": RATE, "y1": ONE, "y2": ONE, "y3": ONE}),
    ("chemical_kinetics", 2): ("k1*y1-k2*y2*y3-k3*y2**2", "y2", {"k1": RATE, "k2": RATE, "k3": RATE, "y1": ONE, "y2": ONE, "y3": ONE}),
    ("chemical_kinetics", 3): ("k3*y2**2", "y3", {"k3": RATE, "y2": ONE}),
    ("chemical_kinetics", 4): ("v-V_max*S/(K_m+S)", "v", {"v": CONC_RATE, "V_max": CONC_RATE, "S": CONC, "K_m": CONC}),

    # --- climate_earth_systems (3) ---
    ("climate_earth_systems", 0): ("dF-k_F*log(M_atm/M_atm0)", "dF", {"dF": FLUX, "k_F": FLUX, "M_atm": MASS, "M_atm0": MASS}),
    ("climate_earth_systems", 1): ("(dF-lam*T_s)/C_eff", "T_s", {"dF": FLUX, "lam": (1, 0, -3, 0, -1, 0, 0), "T_s": TEMPERATURE, "C_eff": (1, 0, -2, 0, -1, 0, 0)}),
    ("climate_earth_systems", 2): ("M_total-M_atm-M_bio-M_surf-M_deep", "M_total", {s: MASS for s in ("M_total", "M_atm", "M_bio", "M_surf", "M_deep")}),

    # --- discrete_finance (4) ---
    ("discrete_finance", 0): ("F_c-F_a-F_b", "F_c", {"F_c": ONE, "F_a": ONE, "F_b": ONE}),
    ("discrete_finance", 1): ("C_next-2*(2*n+1)*C_n/(n+2)", "C_next", {"C_next": ONE, "n": ONE, "C_n": ONE}),
    ("discrete_finance", 2): ("T_n-N*log(N)", "T_n", {"T_n": ONE, "N": ONE}),
    ("discrete_finance", 3): ("d1-((log(S/K)+(r+sig**2/2))/sig)", "d1", {"d1": ONE, "S": ONE, "K": ONE, "r": ONE, "sig": ONE}),

    # --- economics_game_theory (3) ---
    ("economics_game_theory", 0): ("p_i*(f_i-f_bar)+mu*(p_star-p_i)", "p_i", {"p_i": ONE, "f_i": RATE, "f_bar": RATE, "mu": RATE, "p_star": ONE}),
    ("economics_game_theory", 1): ("D_kl-p_star*log(p_star/p_1)", "D_kl", {"D_kl": ONE, "p_star": ONE, "p_1": ONE}),
    ("economics_game_theory", 2): ("Psi-max(0,f_i-f_bar)", "Psi", {"Psi": ONE, "f_i": ONE, "f_bar": ONE}),

    # --- epidemiology_population (6) ---
    ("epidemiology_population", 0): ("R0-beta/gamma", "R0", {"R0": ONE, "beta": RATE, "gamma": RATE}),
    ("epidemiology_population", 1): ("p_c-1+1/R0", "p_c", {"p_c": ONE, "R0": ONE}),
    ("epidemiology_population", 2): ("-beta*S*I/N", "S", {"beta": RATE, "S": ONE, "I": ONE, "N": ONE}),
    ("epidemiology_population", 3): ("beta*S*I/N-sig*E", "E", {"beta": RATE, "S": ONE, "I": ONE, "N": ONE, "sig": RATE, "E": ONE}),
    ("epidemiology_population", 4): ("sig*E-gam*I", "I", {"sig": RATE, "E": ONE, "gam": RATE, "I": ONE}),
    ("epidemiology_population", 5): ("R_eff-beta_eff*S/(gamma*N)", "R_eff", {"R_eff": ONE, "beta_eff": RATE, "gamma": RATE, "S": ONE, "N": ONE}),

    # --- genetics_evolution (4) ---
    ("genetics_evolution", 0): ("p+q-1", None, {"p": ONE, "q": ONE}),
    ("genetics_evolution", 1): ("f_AA-p**2", "f_AA", {"f_AA": ONE, "p": ONE}),
    ("genetics_evolution", 2): ("f_Aa-2*p*q", "f_Aa", {"f_Aa": ONE, "p": ONE, "q": ONE}),
    ("genetics_evolution", 3): ("s*p*(1-p)/w", "p", {"s": ONE, "p": ONE, "w": ONE}),

    # --- materials_thermophysics (2) ---
    ("materials_thermophysics", 0): ("q_cond+kappa*G", "q_cond", {"q_cond": FLUX, "kappa": (1, 1, -3, 0, -1, 0, 0), "G": (0, -1, 0, 0, 1, 0, 0)}),
    ("materials_thermophysics", 1): ("q_in/(rho*c_p*L)", "T", {"q_in": FLUX, "rho": DENSITY, "c_p": (0, 2, -2, 0, -1, 0, 0), "L": LENGTH}),

    # --- math_info_theory (2) ---
    ("math_info_theory", 0): ("C*log(2)-B*log(1+S/N)", "C", {"C": RATE, "B": RATE, "S": POWER, "N": POWER}),
    ("math_info_theory", 1): ("H_b+p*log(p)+(1-p)*log(1-p)", "H_b", {"H_b": ONE, "p": ONE}),

    # --- nonlinear_dynamics (7) ---
    ("nonlinear_dynamics", 0): ("sig*(y-x)", "x", {"sig": ONE, "x": ONE, "y": ONE}),
    ("nonlinear_dynamics", 1): ("x*(rho-z)-y", "y", {"x": ONE, "rho": ONE, "z": ONE, "y": ONE}),
    ("nonlinear_dynamics", 2): ("x*y-beta*z", "z", {"x": ONE, "y": ONE, "beta": ONE, "z": ONE}),
    ("nonlinear_dynamics", 3): ("gam*cos(om*t)-dlt*v-alf*x-bet*x**3", "v", {"gam": ONE, "om": RATE, "t": TIME, "dlt": ONE, "v": ONE, "alf": ONE, "bet": ONE, "x": ONE}),
    ("nonlinear_dynamics", 4): ("mu*(1-x**2)*v-x", "v", {"mu": ONE, "x": ONE, "v": ONE}),
    ("nonlinear_dynamics", 5): ("D_2-2*A", "D_2", {"D_2": ONE, "A": ONE}),
    ("nonlinear_dynamics", 6): ("C-u*delta_t/delta_x", "C", {"C": ONE, "u": VELOCITY, "delta_t": TIME, "delta_x": LENGTH}),

    # --- pharmacology_biochemistry (4) ---
    ("pharmacology_biochemistry", 0): ("theta-hill_eq(C_p,EC50,n)", "theta", {"theta": ONE, "C_p": ONE, "EC50": ONE, "n": ONE}),
    ("pharmacology_biochemistry", 1): ("V_d-Dose/C_p", "V_d", {"V_d": VOLUME, "Dose": MASS, "C_p": DENSITY}),
    ("pharmacology_biochemistry", 2): ("CL-k_el*V_d", "CL", {"CL": (0, 3, -1, 0, 0, 0, 0), "k_el": RATE, "V_d": VOLUME}),
    ("pharmacology_biochemistry", 3): ("v-V_max*S/(K_m*(1+I/K_i)+S)", "v", {"v": CONC_RATE, "V_max": CONC_RATE, "S": CONC, "K_m": CONC, "I": CONC, "K_i": CONC}),

    # --- relativistic_plasma (3) ---
    ("relativistic_plasma", 0): ("gam**2*(c**2-v**2)-c**2", "gam", {"gam": ONE, "v": VELOCITY, "c": VELOCITY}),
    ("relativistic_plasma", 1): ("r_L-gamma*m*v/(q*B)", "r_L", {"r_L": LENGTH, "gamma": ONE, "m": MASS, "v": VELOCITY, "q": (0, 0, 1, 1, 0, 0, 0), "B": (1, 0, -2, -1, 0, 0, 0)}),
    ("relativistic_plasma", 2): ("F_drag-rho*v**2*C_D*A/2", "F_drag", {"F_drag": FORCE, "rho": DENSITY, "v": VELOCITY, "C_D": ONE, "A": AREA}),

    # --- statistics_causal_inference (4) ---
    ("statistics_causal_inference", 0): ("beta_ols-cov_xy/var_x", "beta_ols", {"beta_ols": ONE, "cov_xy": ONE, "var_x": ONE}),
    ("statistics_causal_inference", 1): ("beta_iv-cov_zy/cov_zx", "beta_iv", {"beta_iv": ONE, "cov_zy": ONE, "cov_zx": ONE}),
    ("statistics_causal_inference", 2): ("v_crb-1/I_F", "v_crb", {"v_crb": ONE, "I_F": ONE}),
    ("statistics_causal_inference", 3): ("r*sqrt(var_x*var_z)-cov_xz", "r", {"r": ONE, "var_x": ONE, "var_z": ONE, "cov_xz": ONE}),
}


def typed_reference_packs(packs):
    result = copy.deepcopy(packs)
    for (pack, index), (expression, output, dimensions) in ANNOTATIONS.items():
        if pack not in result or index >= len(result[pack].get("equations", ())):
            continue
        states = (
            set(dimensions)
            if output is None
            else ({"p", "q"} & dimensions.keys() if pack == "genetics_evolution" else set())
        )
        tags = {
            s: {
                "si_dim": list(dim),
                "semantic_tag": 0,
                "unit_scale": [1, 1],
                "role": "DYNAMIC_STATE" if s in states else "DERIVED_OBSERVABLE" if s == output else "CONTROL_PARAM",
                "positive": s not in states,
            }
            for s, dim in dimensions.items()
        }
        equation = {"expr": m._json(m._csel(m._parse(expression))), "symbols": tags}
        result[pack].setdefault("canonical_equations", []).append(equation)
        result[pack].setdefault("annotated_equation_indices", []).append(index)
    return result
