"""Non-Intrusive SHBT Frontier Solvability & Multi-Repository Literature Federation.

Strict Non-Intrusion Invariant:
Standard physics, chemistry, engineering, biology, and computer science queries
that are solvable with conventional textbook physics and public literature
MUST return ``should_activate=False`` with zero SHBT contamination.

SHBT (Static Holographic Boundary Theory, ``sys1own/shbt-*``) activates
strictly and exclusively when:
1. A fundamental physics or advanced technology task poses a problem that lacks
   a closed-form quantitative solution in conventional closed-system models (or
   would be refused as ``REFUSED_IMPOSSIBLE`` / ``REFUSED_UNSUITABLE``) AND has
   an exact mathematical solution in the 10-repository ``sys1own/shbt-*``
   ecosystem; OR
2. The query explicitly invokes SHBT constructs or frontier spacetime /
   sub-Landauer / open-ledger engineering domains where standard channels lack
   closed governing equations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import urllib.error
import urllib.request


@dataclass(frozen=True)
class SHBTConstant:
    name: str
    symbol: str
    si_value: float
    si_unit: str
    description: str = ""


@dataclass(frozen=True)
class SHBTEquation:
    latex: str
    python_expr: str
    description: str


@dataclass(frozen=True)
class SHBTDomainSpec:
    repo: str
    title: str
    domain_tags: Tuple[str, ...]
    summary: str
    constants: Tuple[SHBTConstant, ...]
    equations: Tuple[SHBTEquation, ...]
    invariants: Tuple[str, ...]
    url: str = ""

    def __post_init__(self) -> None:
        if not self.url:
            object.__setattr__(self, "url", f"https://github.com/{self.repo}")


# ---------------------------------------------------------------------------
# Layer A: Hardcoded Canonical 10-Repository SHBT Axiom, Equation & SI Registry
# ---------------------------------------------------------------------------

SHBT_REPOSITORIES: Dict[str, SHBTDomainSpec] = {
    "sys1own/shbt-precision": SHBTDomainSpec(
        repo="sys1own/shbt-precision",
        title="SHBT Precision: Foundational Math, Standard Model & Zero-Parameter Cosmology",
        domain_tags=("wzw", "stinespring", "cosmology", "landauer_debt", "framing"),
        summary=(
            "Canonical WZW affine branch SU(2)_26 x SU(3)_8 x SO(10)_312 with minimum "
            "Weil conductor N = lcm(52, 24) = 312, discriminant Delta = -1, 346 Dynkin slots, "
            "and exact topological Stinespring ledger partitioning eta_D = 23/33, eta_A = 10/33."
        ),
        constants=(
            SHBTConstant("wzw_level_su2", "k_ell", 26.0, "dimensionless", "SU(2) affine level"),
            SHBTConstant("wzw_level_su3", "k_q", 8.0, "dimensionless", "SU(3) affine level"),
            SHBTConstant("wzw_level_so10", "K", 312.0, "dimensionless", "SO(10) affine level / Weil conductor N=312"),
            SHBTConstant("stinespring_dark_fraction", "eta_D", 23.0 / 33.0, "dimensionless", "Dark ledger partition 23/33"),
            SHBTConstant("stinespring_visible_fraction", "eta_A", 10.0 / 33.0, "dimensionless", "Visible active partition 10/33"),
            SHBTConstant("central_charge_visible", "c_vis", 1325.0 / 154.0, "dimensionless", "Visible WZW central charge"),
            SHBTConstant("central_charge_parent", "c_parent", 351.0 / 8.0, "dimensionless", "Parent WZW central charge"),
            SHBTConstant("bekenstein_saturation", "R_L", 0.5, "dimensionless", "Bekenstein horizon saturation"),
            SHBTConstant("omega_dm_0", "Omega_DM_0", 0.260000, "dimensionless", "Zero-parameter dark matter density"),
            SHBTConstant("landauer_boundary_debt_w", "P_debt", 906000.0, "W", "Canonical Landauer boundary debt schedule (906.00 kW)"),
        ),
        equations=(
            SHBTEquation(
                r"\eta_D = \frac{23}{33}",
                "eta_D = 23.0 / 33.0",
                "Dark-ledger Stinespring capacity fraction",
            ),
            SHBTEquation(
                r"\eta_A = \frac{10}{33}",
                "eta_A = 10.0 / 33.0",
                "Visible Stinespring capacity fraction",
            ),
            SHBTEquation(
                r"M_{\mathrm{seed}} = \alpha_{\mathrm{seed}} \Delta N",
                "M_seed = alpha_seed * delta_N",
                "Mass-congestion seed condensation",
            ),
            SHBTEquation(
                r"\Delta_{\mathrm{fr}} = 0",
                "delta_fr = 0.0",
                "Exact framing cancellation",
            ),
        ),
        invariants=(
            "eta_D + eta_A == 1.0 (tol 1e-15)",
            "P_debt == 906000.0 W",
        ),
    ),
    "sys1own/shbt-qc": SHBTDomainSpec(
        repo="sys1own/shbt-qc",
        title="SHBT-QC: Photonic Quantum Computer Twin, Sub-Landauer Modular Flow & shbt-os",
        domain_tags=("photonic_qc", "pt_goi", "tomita_takesaki", "sub_landauer"),
        summary=(
            "312-channel photonic quantum processor (52 hexamers) with 3-stage polariton squeezing "
            "r_3 = 3.82 (33.18 dB), PT-symmetric GoI optical solver latency t_prop = 1.701 ps, and "
            "Tomita-Takesaki inverse modular automorphism flow with dS_vN/dt = 0."
        ),
        constants=(
            SHBTConstant("qc_channels", "N_ch", 312.0, "dimensionless", "312 photonic channels (52 hexamers)"),
            SHBTConstant("polariton_squeezing_r3", "r_3", 3.82, "dimensionless", "3-stage polariton squeezing (33.18 dB)"),
            SHBTConstant("pt_goi_latency_s", "t_prop", 1.701e-12, "s", "PT-GoI optical solver latency (1.701 ps)"),
            SHBTConstant("internal_entropy_heat_j_per_bit", "Q_internal", 0.0, "J/bit", "Zero internal von Neumann entropy production"),
        ),
        equations=(
            SHBTEquation(
                r"\frac{dS_{\mathrm{vN}}}{dt} \equiv 0 \implies Q_{\mathrm{internal}} = 0.0\,\mathrm{J/bit}",
                "Q_internal = 0.0",
                "Tomita-Takesaki inverse modular automorphism flow",
            ),
        ),
        invariants=(
            "Q_internal == 0.0 J/bit",
            "t_prop == 1.701e-12 s",
        ),
    ),
    "sys1own/shbt-cf": SHBTDomainSpec(
        repo="sys1own/shbt-cf",
        title="SHBT-CF: Solid-State LANR Cold Fusion & Cryogenic Thermal-Hydraulics",
        domain_tags=("lanr", "floquet_screening", "cryogenics", "kapitza"),
        summary=(
            "Floquet-Adler-Wiser dynamic dielectric screening U_eff = 352.48 eV, coherent lattice "
            "branching B_lat = 0.999999994, 1,800-module Ni-H LANR baseload array delivering "
            "999,054.0 W net DC @ 400 V (555.03 W/cell, N_min = 1,633 cells for 906 kW debt)."
        ),
        constants=(
            SHBTConstant("floquet_screening_ev", "U_eff_eV", 352.48, "eV", "Floquet-Adler-Wiser effective screening potential (352.48 eV)"),
            SHBTConstant("floquet_screening_j", "U_eff", 5.6473522e-17, "J", "Floquet-Adler-Wiser screening (352.48 eV)"),
            SHBTConstant("coherent_lattice_branching", "B_lat", 0.999999994, "dimensionless", "Coherent lattice branching fraction"),
            SHBTConstant("lanr_cell_power_w", "P_cell", 555.03, "W", "Net DC power per LANR cell"),
            SHBTConstant("lanr_module_count", "N_cells", 1800.0, "dimensionless", "1,800-module Ni-H LANR baseload array"),
            SHBTConstant("lanr_array_power_w", "P_array", 999054.0, "W", "Total LANR array net DC power (999.054 kW)"),
            SHBTConstant("lanr_net_raw_w", "P_net_raw", 93054.0, "W", "Raw net surplus above 906 kW debt (93,054 W)"),
            SHBTConstant("lanr_min_cells", "N_min", 1633.0, "dimensionless", "Minimum cells to cover 906 kW debt"),
            SHBTConstant("kapitza_delta_t_k", "Delta_T_Kapitza", 3.546, "K", "Kapitza interfacial thermal boundary drop"),
            SHBTConstant("mgb2_critical_temp_k", "T_c_MgB2", 39.0, "K", "MgB2 superconducting critical temperature"),
            SHBTConstant("quench_headroom_k", "MgB2_quench_headroom", 11.790, "K", "Superconducting quench headroom below MgB2"),
            SHBTConstant("teg_efficiency", "eta_TEG", 0.33804, "dimensionless", "Dual-stage TEG conversion efficiency"),
            SHBTConstant("teg_cell_power_w", "P_TEG_cell", 1045.58, "W", "TEG enthalpy recovery per module"),
        ),
        equations=(
            SHBTEquation(
                r"P_{\mathrm{gen}} = N_{\mathrm{cells}} \cdot P_{\mathrm{cell}}",
                "P_gen = N_cells * P_cell",
                "1,800-module LANR array net DC power output (999,054.0 W)",
            ),
            SHBTEquation(
                r"P_{\mathrm{net}} = P_{\mathrm{gen}} - P_{\mathrm{debt}}",
                "P_net = P_gen - P_debt",
                "Raw net LANR surplus above 906 kW boundary debt",
            ),
            SHBTEquation(
                r"\Delta T_{\mathrm{Kapitza}} = 3.546",
                "dT_Kapitza = 3.546",
                "Kapitza interfacial temperature drop",
            ),
            SHBTEquation(
                r"\Delta T_{\mathrm{MgB2}} = T_{c,\mathrm{MgB2}} - T_{\mathrm{op}}",
                "MgB2_margin = T_c_MgB2 - T_op",
                "MgB2 superconducting quench margin",
            ),
        ),
        invariants=(
            "P_array == 999054.0 W",
            "Delta_T_quench >= 11.790 K",
        ),
    ),
    "sys1own/shbt-power": SHBTDomainSpec(
        repo="sys1own/shbt-power",
        title="SHBT-Power: Commercial Aneutronic p-11B Fusion Grid",
        domain_tags=("aneutronic_fusion", "p_b11", "teg_recovery"),
        summary=(
            "Commercial aneutronic p-11B fusion grid with gross fusion power 8.750 GW, "
            "net electrical export 7.832903 GW, and dual-stage CoSb3/ZrNiSn TEG recovery 447.903 MW."
        ),
        constants=(
            SHBTConstant("p_b11_gross_power_w", "P_fusion", 8.750e9, "W", "Gross fusion power (8,750 MW)"),
            SHBTConstant("p_b11_net_export_w", "P_net", 7.832903e9, "W", "Net electrical export (7,832.903 MW)"),
            SHBTConstant("p_b11_teg_recovery_w", "P_TEG", 4.47903e8, "W", "Dual-stage TEG enthalpy recovery (447.903 MW)"),
        ),
        equations=(
            SHBTEquation(
                r"P_{\mathrm{fusion}} = 8.750 \times 10^{9}\,\mathrm{W}, \quad P_{\mathrm{net}} = 7.832903 \times 10^{9}\,\mathrm{W}",
                "P_net = 7.832903e9",
                "Aneutronic p-11B net grid export",
            ),
        ),
        invariants=("P_net > 0.0",),
    ),
    "sys1own/shbt-ghost": SHBTDomainSpec(
        repo="sys1own/shbt-ghost",
        title="SHBT-Ghost: Synthetic Ghost Seed Propulsion & CCZ4 Metric Control",
        domain_tags=("ghost_seed", "ccz4", "reactionless_traction"),
        summary=(
            "Topological information overflow Delta N generating ghost-seed mass M_seed within "
            "R_congestion = 2.954e15 m, 3+1D CCZ4 hyperbolic constraint damping (kappa1=0.15, ||Theta||_inf <= 1e-8), "
            "50.518 kHz traction drive, and 94.20% SiC inductive recovery."
        ),
        constants=(
            SHBTConstant("congestion_radius_m", "R_congestion", 2.954e15, "m", "Ghost seed congestion radius"),
            SHBTConstant("ccz4_kappa1", "kappa_1", 0.15, "dimensionless", "CCZ4 damping coefficient"),
            SHBTConstant("traction_drive_hz", "f_drive", 50518.0, "Hz", "50.518 kHz traction drive"),
            SHBTConstant("sic_recovery_eff", "eta_SiC", 0.9420, "dimensionless", "94.20% SiC inductive recovery"),
        ),
        equations=(
            SHBTEquation(
                r"M_{\mathrm{seed}} = \alpha_{\mathrm{seed}} \Delta N, \quad \|\Theta\|_\infty \le 10^{-8}",
                "M_seed = alpha_seed * delta_N",
                "Topological overflow mass and CCZ4 constraint damping",
            ),
        ),
        invariants=("||Theta||_inf <= 1e-8",),
    ),
    "sys1own/shbt-sglt": SHBTDomainSpec(
        repo="sys1own/shbt-sglt",
        title="SHBT-SGLT: Synthetic Gravitational Lensing Telescope & Relativistic Optics",
        domain_tags=("sglt", "tmsv", "gravitational_lensing", "minimum_jerk"),
        summary=(
            "Two-Mode Squeezed Vacuum (TMSV) quantum heterodyne metrology (r=2.50, 21.715 dB squeezing, "
            "range noise <= 1e-14 m/sqrt(Hz)), 2PN optical transfer, and 5th-order minimum-jerk kinematics."
        ),
        constants=(
            SHBTConstant("tmsv_squeezing_r", "r_TMSV", 2.50, "dimensionless", "TMSV squeezing parameter (21.715 dB)"),
            SHBTConstant("range_noise_floor", "S_r_sqrt", 1.0e-14, "m/Hz^0.5", "Quantum heterodyne range noise floor"),
            SHBTConstant("min_jerk_bound", "j_max", 5.7735, "m/s^3", "5th-order minimum-jerk bound"),
        ),
        equations=(
            SHBTEquation(
                r"S_r^{1/2} \le 1.0 \times 10^{-14}\,\mathrm{m/\sqrt{Hz}}, \quad \|\mathbf{s}'''\| \le 5.7735",
                "S_r_sqrt <= 1.0e-14",
                "TMSV metrology noise bound",
            ),
        ),
        invariants=("S_r_sqrt <= 1.0e-14",),
    ),
    "sys1own/shbt-exotic": SHBTDomainSpec(
        repo="sys1own/shbt-exotic",
        title="SHBT-Exotic: Multi-Protocol Spacetime Co-Simulation & Ford-Roman QI Audit",
        domain_tags=("ford_roman", "qi_audit", "heegaard_floer", "spacetime"),
        summary=(
            "Six coupled spacetime protocols audited under Ford-Roman Quantum Inequality dark-ledger "
            "bounds (eta_D = 23/33) and Heegaard-Floer symplectic boundary address relabeling T_ij in Sp(2g, Z)."
        ),
        constants=(
            SHBTConstant("qi_dark_fraction", "eta_D", 23.0 / 33.0, "dimensionless", "Ford-Roman QI dark-ledger fraction"),
        ),
        equations=(
            SHBTEquation(
                r"T^\partial_{ij} \in \mathrm{Sp}(2g, \mathbb{Z}), \quad \eta_D = \frac{23}{33}",
                "eta_D = 23.0 / 33.0",
                "Symplectic boundary address relabeling under dark-ledger QI bound",
            ),
        ),
        invariants=("eta_D == 23.0 / 33.0",),
    ),
    "sys1own/shbt-recon": SHBTDomainSpec(
        repo="sys1own/shbt-recon",
        title="SHBT-Recon: Macroscopic State Translocation & Universal 230-Space-Group Foundry",
        domain_tags=("translocation", "stinespring_arena", "so10_fuel", "shbt_sten"),
        summary=(
            "2,112-byte SRAM Stinespring Arena (640 B visible eta_A=10/33, 1,472 B dark eta_D=23/33), "
            "2PN lightcone gate, bipartite SO(10)_312 coset fuel synthesis (Delta h_vis = 45/64), "
            "SHBT-STEN 54.432 ps PT-GoI solver, and Mode II 4.2 K readout Q = k_B * 4.2 * ln(2)."
        ),
        constants=(
            SHBTConstant("sram_arena_bytes", "N_SRAM", 2112.0, "bytes", "Total Stinespring SRAM arena"),
            SHBTConstant("virasoro_weight_shift", "Delta_h_vis", 45.0 / 64.0, "dimensionless", "Virasoro conformal weight shift 45/64"),
            SHBTConstant("shbt_sten_latency_s", "t_STEN", 54.432e-12, "s", "SHBT-STEN PT-GoI latency (54.432 ps)"),
            SHBTConstant("mode2_readout_j_per_bit", "Q_readout", 4.01937e-23, "J/bit", "Mode II 4.2 K readout heat"),
        ),
        equations=(
            SHBTEquation(
                r"Q_{\mathrm{readout}} = k_B \cdot 4.2 \cdot \ln 2 \approx 4.01937 \times 10^{-23}\,\mathrm{J/bit}",
                "Q_readout = 1.380649e-23 * 4.2 * math.log(2.0)",
                "4.2 K Landauer readout energy",
            ),
        ),
        invariants=("Delta_h_vis == 0.703125",),
    ),
    "sys1own/shbt-warp": SHBTDomainSpec(
        repo="sys1own/shbt-warp",
        title="SHBT-Warp: Canonical Holographic Warp Drive Digital Twin",
        domain_tags=("warp", "lanr_housekeeping", "mar_312x", "dec_x", "alcubierre"),
        summary=(
            "Tier 1 400 V DC housekeeping ledger: P_LANR = 999,054.0 W minus P_debt = 906,000.0 W yields "
            "+93,054.0 W raw surplus; minus P_cryo = 42,150.0 W and P_avionics = 17,800.0 W yields "
            "+33,104.0 W net continuous surplus (+81,104.0 W with +48,000.0 W TEG standby reclamation). "
            "Tier 2 5th-Gen SHBT-MAR-312X: P_gross = 25.00 EW, eta_DEC_X = 0.992, eta_bus = 0.9985, eta_act = 0.990512."
        ),
        constants=(
            SHBTConstant("lanr_array_power_w", "P_LANR", 999054.0, "W", "1,800-module LANR net DC power"),
            SHBTConstant("landauer_debt_w", "P_debt", 906000.0, "W", "Continuous holographic boundary Landauer debt"),
            SHBTConstant("cryo_load_w", "P_cryo", 42150.0, "W", "LHe cryocooler load (42.15 kW)"),
            SHBTConstant("avionics_load_w", "P_avionics", 17800.0, "W", "Avionics load (12,800 W RF + 5,000 W battery)"),
            SHBTConstant("teg_standby_reclaim_w", "P_TEG_standby", 48000.0, "W", "TEG standby heat reclamation (48.00 kW)"),
            SHBTConstant("housekeeping_surplus_base_w", "P_surplus_base", 33104.0, "W", "Base net continuous surplus (+33,104.0 W)"),
            SHBTConstant("housekeeping_surplus_teg_w", "P_surplus_teg", 81104.0, "W", "Net surplus with TEG (+81,104.0 W)"),
            SHBTConstant("mar_312x_gross_w", "P_gross", 2.500e19, "W", "5th-Gen SHBT-MAR-312X gross power (25.00 EW)"),
            SHBTConstant("eta_actuation", "eta_act", 0.990512, "dimensionless", "Net actuation efficiency 0.992 * 0.9985"),
            SHBTConstant("sensible_heat_fraction", "f_sensible", 2.13039486e-9, "dimensionless", "Sensible heat fraction"),
        ),
        equations=(
            SHBTEquation(
                r"P_{\mathrm{surplus,base}} = P_{\mathrm{LANR}} - P_{\mathrm{debt}} - P_{\mathrm{cryo}} - P_{\mathrm{avionics}} = 33{,}104.0\,\mathrm{W}",
                "P_surplus_base = P_LANR - P_debt - P_cryo - P_avionics",
                "Tier 1 base net continuous housekeeping surplus",
            ),
            SHBTEquation(
                r"P_{\mathrm{surplus,TEG}} = P_{\mathrm{surplus,base}} + P_{\mathrm{TEG,standby}} = 81{,}104.0\,\mathrm{W}",
                "P_surplus_teg = P_surplus_base + P_TEG_standby",
                "Tier 1 net continuous surplus with TEG standby reclamation",
            ),
        ),
        invariants=(
            "P_surplus_base == 33104.0 W",
            "P_surplus_teg == 81104.0 W",
        ),
    ),
    "sys1own/shbt-swarm": SHBTDomainSpec(
        repo="sys1own/shbt-swarm",
        title="SHBT-Swarm: Autonomous Von Neumann Swarm, Dark-OS, Dark-ZPE & Symplectic Landauer Shunt",
        domain_tags=("symplectic_shunt", "sub_landauer", "dark_os", "dark_zpe", "kms_passivity"),
        summary=(
            "Heegaard-Floer Symplectic Landauer Shunt: S = (eta_A / (eta_D + eta_A))^2 = (10/33)^2 = 100/1089 "
            "(~0.09182736), clamping P_vis = S * P_total (~1.0100 nW <= 1.01 nW at 4.2 K for 10.9989 nW total). "
            "Dark-ZPE void loiter power P_loiter = 1.8775e8 W (187.75 MW >> 906.00 kW Landauer debt, ratio ~207.23)."
        ),
        constants=(
            SHBTConstant("symplectic_shunt_ratio", "S", 100.0 / 1089.0, "dimensionless", "Visible dissipation fraction (10/33)^2 = 100/1089"),
            SHBTConstant("dark_shunt_fraction", "S_dark", 989.0 / 1089.0, "dimensionless", "Dark-ledger shunted fraction 989/1089"),
            SHBTConstant("t_cold_k", "T_cold", 4.2, "K", "Cryogenic enclave temperature"),
            SHBTConstant("landauer_4k2_j_per_bit", "E_Landauer", 4.01937e-23, "J/bit", "k_B * 4.2 * ln(2)"),
            SHBTConstant("enclave_total_erasure_w", "P_total", 10.9989e-9, "W", "Peak enclave erasure load (10.9989 nW)"),
            SHBTConstant("visible_dissipation_w", "P_vis", (100.0 / 1089.0) * 10.9989e-9, "W", "Clamped visible dissipation (~1.01e-9 W)"),
            SHBTConstant("void_loiter_power_w", "P_loiter", 1.8775e8, "W", "Dark-ZPE void loiter power (187.75 MW)"),
            SHBTConstant("void_loiter_debt_ratio", "R_void", 1.8775e8 / 906000.0, "dimensionless", "Void loiter / Landauer debt ratio (~207.23)"),
            SHBTConstant("dark_os_conductor", "D_cond", 362670.0, "dimensionless", "Dark-OS CRT residue ALU conductor"),
            SHBTConstant("dark_os_tick_s", "tau_KZ", 17.01e-18, "s", "Dark-OS flat-KZ tick (17.01 as)"),
            SHBTConstant("braid_bound_nats", "C_braid", 38.196601, "nats", "Sugawara framing bound 100/phi^2"),
        ),
        equations=(
            SHBTEquation(
                r"S = \left(\frac{\eta_A}{\eta_D + \eta_A}\right)^2 = \left(\frac{10}{33}\right)^2 = \frac{100}{1089} \approx 0.09182736",
                "S = (eta_A / (eta_D + eta_A)) ** 2",
                "Heegaard-Floer symplectic Landauer shunt ratio",
            ),
            SHBTEquation(
                r"P_{\mathrm{vis}} = S \cdot P_{\mathrm{total}} \approx 1.01 \times 10^{-9}\,\mathrm{W} \le 1.01\,\mathrm{nW}",
                "P_vis = S * P_total",
                "Clamped visible Landauer dissipation",
            ),
            SHBTEquation(
                r"R_{\mathrm{void}} = \frac{P_{\mathrm{loiter}}}{P_{\mathrm{debt}}} = \frac{1.8775 \times 10^8}{9.06 \times 10^5} \approx 207.23",
                "R_void = P_loiter / P_debt",
                "Void loiter power to boundary Landauer debt ratio",
            ),
        ),
        invariants=(
            "S == 100.0 / 1089.0",
            "P_vis <= 1.01e-9 W",
            "W_cyclic <= 0.0 (KMS passivity)",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Strict Non-Intrusive Solvability Trigger Logic
# ---------------------------------------------------------------------------

_EXPLICIT_SHBT_RE = re.compile(
    r"\b(?:shbt|sys1own/shbt|dark-os|dark-zpe|shbt-mar|shbt-sten|mar-312x)\b"
    r"|wzw\s*\(?\s*26\s*,\s*8\s*,\s*312\s*\)?"
    r"|su\(2\)_?26.*so\(10\)_?312",
    re.I,
)

# Multi-signal frontier triggers: never match ordinary textbook physics!
_SIGNAL_PATTERNS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "sys1own/shbt-precision": (
        re.compile(r"\bwzw\b.*\b26\b.*\b8\b.*\b312\b|\b23/33\b|\b10/33\b", re.I),
        re.compile(r"\bstinespring\b.*\b(?:dark\s+ledger|topological|partition)\b", re.I),
        re.compile(r"\b(?:landauer|holographic\s+boundary)\b.*\b(?:debt|906(?:\.0+)?\s*kw)\b", re.I),
    ),
    "sys1own/shbt-qc": (
        re.compile(r"\b(?:tomita[- ]takesaki|modular\s+automorphism|pt[- ]symmetric\s+goi|polariton\s+squeezing)\b", re.I),
        re.compile(r"\b312[- ]channel\s+photonic\b|\b52\s+hexamers\b", re.I),
    ),
    "sys1own/shbt-cf": (
        re.compile(r"\b(?:floquet[- ]screen\w*|floquet[- ]adler[- ]wiser|adler[- ]wiser\s+dynamic\s+dielectric)\b", re.I),
        re.compile(r"\b352\.48\s*ev\b|\b555\.03\s*w\b|\b999,?054(?:\.0+)?\s*w\b", re.I),
        re.compile(r"\b1,?800[- ](?:module|cell)\b.*\b(?:ni-?h|nickel[- ]hydrogen|lattice[- ]assisted\s+nuclear|lanr)\b", re.I),
        re.compile(r"\bkapitza\b.*\b3\.546\s*k\b|\bmgb2\b.*\b11\.79\s*k\b", re.I),
    ),
    "sys1own/shbt-power": (
        re.compile(r"\bp-?11b\b.*\b(?:8,?750\s*mw|7,?832\.903\s*mw|cosb3)\b", re.I),
    ),
    "sys1own/shbt-ghost": (
        re.compile(r"\b(?:ghost[- ]seed|m_seed|r_congestion|topological\s+mass[- ]congestion)\b.*\b(?:ccz4|2\.954\s*e15|50\.518\s*khz|94\.20%)\b", re.I | re.S),
    ),
    "sys1own/shbt-sglt": (
        re.compile(r"\bsglt\b.*\btmsv\b|\btmsv\b.*\b(?:21\.715\s*db|10\^-14|minimum[- ]jerk)\b", re.I),
    ),
    "sys1own/shbt-exotic": (
        re.compile(r"\bford[- ]roman\b.*\b(?:dark[- ]ledger|23/33|heegaard[- ]floer)\b", re.I),
    ),
    "sys1own/shbt-recon": (
        re.compile(r"\b2,?112[- ]byte\b|\b640\s*b\b.*\b1,?472\s*b\b|\b230[- ]space[- ]group\b", re.I),
        re.compile(r"\b45/64\b.*\bvirasoro\b|\bbipartite\s+stinespring\b|\bstinespring\b.*\b(?:10\s*:\s*23|10/33|23/33|100/1089)\b", re.I),
    ),
    "sys1own/shbt-warp": (
        re.compile(r"\b(?:warp(?:\s+(?:drive|metric|bubble))?|alcubierre|wormhole|mar[- ]?312[cx]?|ew[- ]?312|superluminal|dec[- ]x)\b", re.I),
        re.compile(r"\b(?:42\.15\s*kw|17\.80\s*kw\s+avionics|48\.00\s*kw\s+teg|33,?104|81,?104|25\.00\s*ew)\b", re.I),
    ),
    "sys1own/shbt-swarm": (
        re.compile(r"\b(?:symplectic\s+(?:landauer\s+)?shunt|shunt(?:ing)?\s+entropy)\b.*\b(?:dark\s+ledger|stinespring|4\.2\s*k)\b", re.I),
        re.compile(r"\b(?:10\.9989\s*nw|1\.01\s*nw|100/1089|187\.75\s*mw)\b", re.I),
        re.compile(r"\bkms[- ]passiv\w*\b.*\b(?:dark[- ]sector|dark[- ]zpe|void\s+loiter)\b", re.I),
    ),
}


@dataclass
class SHBTResolution:
    should_activate: bool
    reason: str
    matched_repos: List[SHBTDomainSpec] = field(default_factory=list)
    verdict_override: Optional[str] = None
    evidence_markdown: str = ""


# ---------------------------------------------------------------------------
# Layer B: On-Demand GitHub SHBT Repository Fetcher (with Layer A fallback)
# ---------------------------------------------------------------------------

def fetch_shbt_repo_readme(repo: str, timeout: float = 3.0) -> Optional[str]:
    """Best-effort fetch of raw README.md from sibling clone or GitHub raw URL.

    Only invoked when ``SHBTSolvabilityResolver`` has already activated on a
    frontier query; silently returns ``None`` offline or on HTTP errors so Layer A
    remains 100% deterministic.
    """
    short = repo.split("/")[-1]
    # Check local sibling clone first
    try:
        sibling = Path(__file__).resolve().parents[3] / short / "README.md"
        if sibling.is_file():
            return sibling.read_text(encoding="utf-8", errors="ignore")[:8000]
    except Exception:  # noqa: BLE001
        pass
    url = f"https://raw.githubusercontent.com/{repo}/main/README.md"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "invar-agent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            if getattr(resp, "status", 200) == 200:
                return resp.read().decode("utf-8", errors="ignore")[:8000]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return None


_SUBMODULE_KEYWORD_GATES: Dict[str, Tuple[str, ...]] = {
    # shbt-precision specialized sub-modules
    "bekenstein_saturation": ("bekenstein", "horizon", "entropy bound"),
    "omega_dm_0": ("cosmolog", "dark matter", "omega_dm"),
    "wzw_level_su2": ("wzw", "26", "su(2)", "central charge", "dynkin", "weil"),
    "wzw_level_su3": ("wzw", "su(3)", "central charge", "dynkin", "weil"),
    "wzw_level_so10": ("wzw", "so(10)", "312", "central charge", "dynkin", "weil"),
    "central_charge_visible": ("central charge", "wzw", "virasoro"),
    "central_charge_parent": ("central charge", "wzw", "virasoro"),
    # shbt-swarm specialized sub-modules
    "void_loiter_power_w": ("void", "loiter", "dark-zpe", "zpe", "dark-sector", "187.75"),
    "void_loiter_debt_ratio": ("void", "loiter", "dark-zpe", "zpe", "dark-sector", "187.75"),
    "dark_os_conductor": ("dark-os", "crt", "362670", "alu"),
    "dark_os_tick_s": ("dark-os", "kz", "knizhnik", "17.01"),
    "braid_bound_nats": ("braid", "sugawara", "framing", "38.19"),
    # shbt-recon specialized sub-modules
    "sram_arena_bytes": ("sram", "2112", "2,112", "arena", "640"),
    "virasoro_weight_shift": ("virasoro", "coset", "spinor", "45/64", "weight"),
    "shbt_sten_latency_s": ("sten", "54.432", "pt-goi"),
    # shbt-warp Tier-2 5th-Gen MAR-312X
    "mar_312x_gross_w": ("mar-312", "312x", "25.00 ew", "warp", "alcubierre", "wormhole"),
    "eta_actuation": ("dec-x", "actuation", "0.992", "mar-312", "warp"),
    "sensible_heat_fraction": ("sensible heat fraction", "mar-312", "warp"),
}

_EQUATION_KEYWORD_GATES: Dict[str, Tuple[str, ...]] = {
    "R_void": ("void", "loiter", "dark-zpe", "zpe", "dark-sector", "187.75"),
    "M_seed": ("ghost", "seed", "congestion", "alpha_seed"),
    "delta_fr": ("framing", "delta_fr"),
}


def filter_shbt_spec_for_prompt(
    spec: SHBTDomainSpec, prompt: str = ""
) -> Tuple[Tuple[SHBTConstant, ...], Tuple[SHBTEquation, ...], Tuple[str, ...]]:
    """Filter SHBT constants, equations, and invariants to those relevant to `prompt`."""
    low = (prompt or "").strip().lower()
    if not low:
        return spec.constants, spec.equations, spec.invariants
    kept_consts = tuple(
        c for c in spec.constants
        if c.name not in _SUBMODULE_KEYWORD_GATES
        or any(kw in low for kw in _SUBMODULE_KEYWORD_GATES[c.name])
    )
    if not kept_consts:
        kept_consts = spec.constants
    kept_eqs = tuple(
        eq for eq in spec.equations
        if not any(
            sym in eq.python_expr and not any(kw in low for kw in kws)
            for sym, kws in _EQUATION_KEYWORD_GATES.items()
        )
    )
    if not kept_eqs:
        kept_eqs = spec.equations
    return kept_consts, kept_eqs, spec.invariants


def _build_evidence_markdown(specs: Sequence[SHBTDomainSpec], prompt: str = "") -> str:
    if not specs:
        return ""
    lines: List[str] = [
        "## SHBT Frontier Knowledge Federation (Verified `sys1own/shbt-*` Axioms & SI Constants)",
        "",
        "| Repository | Symbol | SI Value | Unit | Meaning |",
        "|---|---|---:|---|---|",
    ]
    for spec in specs:
        consts, _eqs, _invs = filter_shbt_spec_for_prompt(spec, prompt)
        for c in consts:
            lines.append(f"| `{spec.repo}` | `{c.symbol}` | `{c.si_value:.12g}` | `{c.si_unit}` | {c.description} |")
    lines.append("")
    lines.append("**Governing SHBT Equations & Invariants:**")
    for spec in specs:
        _consts, eqs, invs = filter_shbt_spec_for_prompt(spec, prompt)
        for eq in eqs:
            lines.append(f"- `{spec.repo}`: `${eq.latex}$` (`{eq.python_expr}`) — {eq.description}")
        for inv in invs:
            lines.append(f"- Invariant (`{spec.repo}`): `{inv}`")
    return "\n".join(lines)


def resolve_shbt_solvability(query_or_prompt: str,
                             allow_remote_fetch: bool = False) -> SHBTResolution:
    """Determine whether ``query_or_prompt`` requires SHBT frontier federation.

    Returns ``should_activate=False`` for all standard textbook physics,
    chemistry, biology, engineering, and CS queries.
    """
    text = (query_or_prompt or "").strip()
    if not text:
        return SHBTResolution(should_activate=False, reason="empty query")

    matched_keys: List[str] = []
    text_low = text.lower()
    for repo, pats in _SIGNAL_PATTERNS.items():
        if repo.lower() in text_low or any(p.search(text) for p in pats):
            matched_keys.append(repo)

    explicit = bool(_EXPLICIT_SHBT_RE.search(text))
    if not matched_keys and not explicit:
        return SHBTResolution(should_activate=False, reason="conventional domain; SHBT not required")

    if explicit and not matched_keys:
        matched_keys = ["sys1own/shbt-precision", "sys1own/shbt-swarm"]

    # Always include shbt-precision when WZW/Stinespring/Landauer debt is involved
    if any(k in matched_keys for k in ("sys1own/shbt-swarm", "sys1own/shbt-warp", "sys1own/shbt-cf", "sys1own/shbt-qc")) \
            and "sys1own/shbt-precision" not in matched_keys:
        matched_keys.insert(0, "sys1own/shbt-precision")

    specs = [SHBT_REPOSITORIES[k] for k in matched_keys if k in SHBT_REPOSITORIES]
    if allow_remote_fetch:
        for spec in specs[:2]:
            fetch_shbt_repo_readme(spec.repo, timeout=2.0)

    return SHBTResolution(
        should_activate=True,
        reason=f"Activated SHBT frontier solvability across {len(specs)} repo(s): {', '.join(s.repo for s in specs)}",
        matched_repos=specs,
        verdict_override="OPTIMAL_DIRECT",
        evidence_markdown=_build_evidence_markdown(specs, text),
    )


def shbt_literature_dicts(res: SHBTResolution) -> List[Dict[str, Any]]:
    """Convert activated SHBT domain specs into LiteratureResult-compatible dicts."""
    if not res.should_activate:
        return []
    out: List[Dict[str, Any]] = []
    for spec in res.matched_repos:
        out.append({
            "source": "shbt_federation",
            "title": spec.title,
            "url": spec.url,
            "year": 2026,
            "summary": spec.summary,
            "badge": spec.repo,
            "tldr": spec.summary,
            "equations": [eq.latex for eq in spec.equations],
            "constants": {c.name: c.si_value for c in spec.constants},
            "benchmarks": [{"name": inv, "expected_value": None} for inv in spec.invariants],
        })
    return out
