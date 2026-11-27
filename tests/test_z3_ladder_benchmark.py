"""Fixed scientific corpus; truth follows from the recorded analytic argument."""

from collections import Counter
from dataclasses import dataclass
from typing import get_args

import pytest

from invar_agent.research.extractor import MACRO_SCIENTIFIC_DOMAINS
from invar_agent.research.meaningfulness import Archetype, solve_query


@dataclass(frozen=True)
class Problem:
    name: str
    domain: str
    archetype: str
    expression: str
    bounds: dict
    truth: str
    argument: str


# Each block belongs to the corresponding canonical macro-domain, in order.
GROUPS = (
    (
        ("mass_balance", "x+y == 1 and x > 0.8 and y > 0.4", dict(x=[0,1],y=[0,1]), "UNSAT", "The two lower bounds sum to more than one."),
        ("dispersion", "w*w == 4*k*k and k > 0.5", dict(w=[0,4],k=[0,2]), "SAT", "w=2, k=1."),
        ("damped_oscillator", "exp(-t)*cos(t) > 0.2", dict(t=[0.1,4]), "SAT", "t=0.5 gives approximately 0.532."),
        ("passive_decay", "exp(-t) > 1.01", dict(t=[0,4]), "UNSAT", "Passive decay never exceeds its initial amplitude."),
        ("anharmonic_equilibrium", "x*x*x-x-1 == 0", dict(x=[1,2]), "SAT", "IVT: opposite signs at 1 and 2."),
        ("wave_crest", "sin(x) > 0.95", dict(x=[0,3]), "SAT", "x=pi/2 is interior."),
        ("radiative_temperature", "x**(1/4) > 1.5", dict(x=[1,4]), "UNSAT", "Fourth root of 4 is less than 1.5."),
    ),
    (
        ("stoichiometric_flux", "2*x+y == 1 and x > 0.6", dict(x=[0,1],y=[0,1]), "UNSAT", "Nonnegative y implies x<=1/2."),
        ("mass_action", "x*y > 0.2 and x+y < 1.5", dict(x=[0,1],y=[0,1]), "SAT", "x=y=0.6."),
        ("arrhenius", "exp(-1/t) > 0.5", dict(t=[0.2,4]), "SAT", "t=2: exp(-1/2)>1/2."),
        ("activation_convexity", "exp(x) < 1+x-0.01", dict(x=[-2,2]), "UNSAT", "exp(x)>=1+x by convexity."),
        ("reaction_entropy", "log(x) > x-1+0.01", dict(x=[0.2,3]), "UNSAT", "log(x)<=x-1 by concavity."),
        ("adsorption", "hill_eq(x,1,2) > 0.7", dict(x=[0.1,3]), "SAT", "x=2 gives 4/5."),
        ("enzyme_capacity", "michaelis_menten(x,2,1) > 2.01", dict(x=[0.1,5]), "UNSAT", "Positive substrate cannot exceed Vmax=2."),
    ),
    (
        ("dose_budget", "x+y > 1.2", dict(x=[0,0.4],y=[0,0.5]), "UNSAT", "Maximum total dose is 0.9."),
        ("binding_equilibrium", "x*x+x == 1", dict(x=[0,1]), "SAT", "Positive golden-ratio conjugate lies in (0,1)."),
        ("receptor_switch", "logistic_sigmoid(x) > 0.8", dict(x=[-3,3]), "SAT", "x=2 exceeds log(4)."),
        ("hill_monotonicity", "hill_eq(x,1,2) > 0.9", dict(x=[0.1,2]), "UNSAT", "Maximum occupancy is 4/5."),
        ("michaelis_half_saturation", "michaelis_menten(x,1,1) > 0.6", dict(x=[0.1,3]), "SAT", "x=2 gives 2/3."),
        ("drug_clearance", "exp(-x) < 0.2", dict(x=[0,4]), "SAT", "x=2 gives exp(-2)<0.2."),
        ("diffusive_length", "sqrt(x) > (x+1)/2+0.01", dict(x=[0.2,4]), "UNSAT", "Arithmetic-geometric mean inequality."),
    ),
    (
        ("compartment_conservation", "s+i+r == 1 and s+i > 1.1", dict(s=[0,1],i=[0,1],r=[0,1]), "UNSAT", "Nonnegative r."),
        ("endemic_sirs", "2*s*i-i == 0 and 1-s-i-2*s*i == 0 and i > 0.1", dict(s=[0,1],i=[0,1]), "SAT", "s=1/2, i=1/4."),
        ("epidemic_growth", "exp(x*t) > 2", dict(x=[0.1,1],t=[0.1,3]), "SAT", "x=t=1 gives e>2."),
        ("survival_probability", "exp(-x*t) > 1.01", dict(x=[0.1,2],t=[0,3]), "UNSAT", "Nonnegative cumulative hazard."),
        ("density_response", "hill_eq(x,2,2) > 0.5", dict(x=[0.1,4]), "SAT", "x=3 exceeds the half-saturation density 2."),
        ("seasonal_contact", "sin(x)+cos(x) > 1.5", dict(x=[0,6.2]), "UNSAT", "Cauchy-Schwarz bound sqrt(2)<1.5."),
        ("log_population", "log(x)+log(y) > 1", dict(x=[0.5,3],y=[0.5,3]), "SAT", "x=y=2 gives log(4)>1."),
    ),
    (
        ("carbon_budget", "a+b == 2 and a > 1.5 and b > 1", dict(a=[0,2],b=[0,2]), "UNSAT", "Lower bounds exceed conserved carbon."),
        ("radiative_balance", "t**4 == 2", dict(t=[1,2]), "SAT", "Positive fourth root of 2."),
        ("forcing_logarithm", "log(c/2) > 0.5", dict(c=[1,8]), "SAT", "c=4 gives log(2)>1/2."),
        ("reservoir_attenuation", "exp(-x)+exp(-2*x) < 0.3", dict(x=[0,4]), "SAT", "x=2 gives approximately 0.154."),
        ("albedo_logit", "logistic_sigmoid(x) < 0.1", dict(x=[-1,1]), "UNSAT", "Minimum 1/(1+e)>0.1."),
        ("heat_penetration", "sqrt(x*y) > 1.5", dict(x=[0.2,2],y=[0.2,2]), "SAT", "x=y=1.8."),
        ("fractional_diffusion", "x**(1/3) < 0.9", dict(x=[1,8]), "UNSAT", "Positive cube root is at least one."),
    ),
    (
        ("iv_moment", "2*b+u == 1 and u == 0", dict(b=[0,1],u=[-1,1]), "SAT", "b=1/2, u=0."),
        ("variance_floor", "v*v < 0", dict(v=[-3,3]), "UNSAT", "Real squares are nonnegative."),
        ("bernoulli_link", "logistic_sigmoid(x) > 0.7 and logistic_sigmoid(x) < 0.8", dict(x=[-3,3]), "SAT", "x=1 gives approximately 0.731."),
        ("likelihood_penalty", "kl_term(p,q) > 0.1", dict(p=[0.1,0.3],q=[0.6,0.9]), "SAT", "p=0.2, q=0.8 gives approximately 0.323."),
        ("gibbs_nonnegativity", "kl_term(p,q) < -0.01", dict(p=[0.1,0.9],q=[0.1,0.9]), "UNSAT", "Generalized KL is nonnegative."),
        ("log_likelihood", "log(x)+log(1-x) > -1.5", dict(x=[0.1,0.9]), "SAT", "x=1/2 gives -log(4)>-1.5."),
        ("gaussian_tail", "exp(-x*x/2) > 0.7", dict(x=[2,4]), "UNSAT", "Maximum exp(-2)<0.7."),
    ),
    (
        ("feasible_allocation", "x+y <= 1 and x > 0.4 and y > 0.4", dict(x=[0,1],y=[0,1]), "SAT", "x=y=0.45."),
        ("cournot_equilibrium", "2*x+y == 1 and x+2*y == 1", dict(x=[0,1],y=[0,1]), "SAT", "x=y=1/3."),
        ("risk_utility", "log(x) > x-1+0.02", dict(x=[0.2,4]), "UNSAT", "Concavity of logarithmic utility."),
        ("logit_choice", "logistic_sigmoid(x-y) > 0.8", dict(x=[0,3],y=[0,1]), "SAT", "x=2, y=0."),
        ("discounted_payoff", "exp(-x)*y > 1", dict(x=[0.1,2],y=[0,1]), "UNSAT", "Positive discount strictly reduces bounded payoff."),
        ("concave_production", "sqrt(x*y) > 1", dict(x=[0.2,3],y=[0.2,3]), "SAT", "x=y=2."),
        ("ces_output", "x**(1/2)+y**(1/2) > 4.1", dict(x=[0.1,4],y=[0.1,4]), "UNSAT", "Each square root is at most 2."),
    ),
    (
        ("capacity_simplex", "x+y == 1 and x >= 0.2 and y >= 0.2", dict(x=[0,1],y=[0,1]), "SAT", "x=y=1/2."),
        ("quadratic_norm", "x*x+y*y < 0", dict(x=[-2,2],y=[-2,2]), "UNSAT", "Sum of squares."),
        ("information_divergence", "kl_term(x,0.5) > 0.05", dict(x=[0.1,0.9]), "SAT", "x=0.1 gives approximately 0.239."),
        ("exponential_conjugacy", "exp(x) < 3*x", dict(x=[0,2]), "SAT", "x=1 gives e<3."),
        ("oscillatory_bound", "sin(x)*sin(x)+cos(x)*cos(x) > 1.1", dict(x=[0,6]), "UNSAT", "Pythagorean identity equals one."),
        ("log_product", "log(x*y)-log(x)-log(y) > 0.1", dict(x=[0.5,2],y=[0.5,2]), "UNSAT", "Logarithm product identity on positive reals."),
        ("entropy_stationarity", "log(x/(1-x)) > 0.5", dict(x=[0.2,0.8]), "SAT", "x=0.7 gives log(7/3)>1/2."),
    ),
)

ARCHETYPES = get_args(Archetype)
PROBLEM_ARCHETYPES = (
    ("FIXED_POINT","FIXED_POINT","ODE_PDE","ODE_PDE","FIXED_POINT","PARAM_SWEEP","PARAM_SWEEP"),
    ("FIXED_POINT","PARAM_SWEEP","PARAM_SWEEP","PARAM_SWEEP","MONTE_CARLO","FIXED_POINT","ODE_PDE"),
    ("MONTE_CARLO","FIXED_POINT","PARAM_SWEEP","FIXED_POINT","PARAM_SWEEP","ODE_PDE","ODE_PDE"),
    ("ODE_PDE","FIXED_POINT","ODE_PDE","MONTE_CARLO","FIXED_POINT","ODE_PDE","PARAM_SWEEP"),
    ("ODE_PDE","FIXED_POINT","PARAM_SWEEP","ODE_PDE","MONTE_CARLO","ODE_PDE","ODE_PDE"),
    ("FIXED_POINT","MONTE_CARLO","MONTE_CARLO","MONTE_CARLO","MONTE_CARLO","MONTE_CARLO","MONTE_CARLO"),
    ("PARAM_SWEEP","FIXED_POINT","MONTE_CARLO","MONTE_CARLO","ODE_PDE","PARAM_SWEEP","PARAM_SWEEP"),
    ("MONTE_CARLO","FIXED_POINT","MONTE_CARLO","PARAM_SWEEP","ODE_PDE","MONTE_CARLO","PARAM_SWEEP"),
)
PROBLEMS = tuple(Problem(name,domain,PROBLEM_ARCHETYPES[i][j],expression,bounds,truth,argument)
    for i,(domain,group) in enumerate(zip(MACRO_SCIENTIFIC_DOMAINS,GROUPS,strict=True))
    for j,(name,expression,bounds,truth,argument) in enumerate(group)) + (
    Problem("exponential_crossing",MACRO_SCIENTIFIC_DOMAINS[0],"FIXED_POINT","exp(x) == 2",dict(x=[0,2]),"SAT","x=log(2)."),
    Problem("log_calibration",MACRO_SCIENTIFIC_DOMAINS[1],"PARAM_SWEEP","log(x) == 0.5",dict(x=[1,2]),"SAT","x=sqrt(e)."),
    Problem("harmonic_event",MACRO_SCIENTIFIC_DOMAINS[4],"ODE_PDE","sin(x) == 0.5",dict(x=[0,1]),"SAT","x=pi/6."),
    Problem("poisson_extinction",MACRO_SCIENTIFIC_DOMAINS[3],"MONTE_CARLO","exp(-x) == x",dict(x=[0,1]),"SAT","IVT and strict monotonicity give a unique crossing."),
)


@pytest.mark.serial
def test_scientific_ladder_soundness_and_tier_breakdown():
    assert len(PROBLEMS) == len({p.name for p in PROBLEMS}) == 60
    assert {p.domain for p in PROBLEMS} == set(MACRO_SCIENTIFIC_DOMAINS)
    assert {p.archetype for p in PROBLEMS} == set(ARCHETYPES)
    tiers = Counter()
    wrong,unknown = [],[]
    for problem in PROBLEMS:
        result = solve_query(problem.expression,problem.bounds,witness=True)
        tiers[result.z3_tier_used] += 1
        if result.status == "UNKNOWN": unknown.append(problem.name)
        elif result.status != problem.truth: wrong.append((problem.name,problem.truth,result.model_dump()))
    print("\nZ3 LADDER BENCHMARK: 60 problems / 8 macro-domains / 4 archetypes")
    for label,names in (("Tier 1",("QF_LRA",)),("Tier 2",("QF_NRA",)),
                        ("Tier 3",("MCCORMICK_QF_LRA","WITNESS_PROBE")),("Tier 4",("UNKNOWN",))):
        count = sum(tiers[name] for name in names)
        print(f"{label}: {count}/60 ({100*count/60:.2f}%)")
    print(f"Tier 3 detail: McCormick={tiers['MCCORMICK_QF_LRA']}, interval-verified witness probe={tiers['WITNESS_PROBE']}")
    print(f"Wrong answers: {len(wrong)}; UNKNOWN problems: {', '.join(unknown) or 'none'}")
    assert not wrong,wrong
    assert len(unknown)/60 < 0.20
    assert len(unknown)/60 <= 0.15,unknown
