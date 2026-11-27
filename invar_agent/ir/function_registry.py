"""UPG-006: Single canonical math function registry with 6 mandatory members.

This single table is the sole authority for:
1. Prompt vocabulary advertised to the SciIR synthesizer (`ir/prompt_templates.py`),
2. Formal gate allow-list and arities (`research/meaningfulness.py`),
3. SciIR -> UAST / Python / C++ / Rust / CSEL expression lowering (`ir/lower_uast.py`),
4. Strict numeric evaluation, outward-rounded interval extensions, NF-19 Lipschitz roundoff
   bounds (`L * e_in + 16 * ulp(M)`), symbolic partial derivatives in `invar.expr.v1`,
   and SI dimension rules.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .expr_ast import Binary, Call, Cond, Cmp, Const, Expr, Unary
from .units import Dim

ZERO_DIM: Dim = (
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
    Fraction(0),
)


class RegistryError(ValueError):
    """Raised when a FunctionSpec violates the 6-member completeness invariant."""


class DomainError(ValueError):
    """Raised on domain violations (e.g., TAN_POLE_IN_DOMAIN, log(<=0), sqrt(<0))."""


def _ulp(x: float) -> float:
    ax = abs(float(x))
    if not math.isfinite(ax):
        return float("inf")
    return math.ulp(max(ax, 1e-300))


def _up(x: float, ulps: int = 1) -> float:
    v = float(x)
    for _ in range(ulps):
        v = math.nextafter(v, math.inf)
    return v


def _down(x: float, ulps: int = 1) -> float:
    v = float(x)
    for _ in range(ulps):
        v = math.nextafter(v, -math.inf)
    return v


def _iv_lo_hi(iv: Any) -> Tuple[float, float]:
    if isinstance(iv, tuple) and len(iv) == 2:
        return (float(iv[0]), float(iv[1]))
    return (float(iv.lo), float(iv.hi))


def _nf19_unary_roundoff(
    lo: float,
    hi: float,
    e_in: float,
    iv_fn: Callable[[float, float], Tuple[float, float]],
    deriv_abs_sup: Callable[[float, float], float],
    kappa: int = 16,
) -> Tuple[Tuple[float, float], float]:
    """NF-19: Tight Lipschitz unary rounding bound `L * e_in + kappa * ulp(M)` upward-rounded."""
    blo = _down(lo - e_in)
    bhi = _up(hi + e_in)
    nom_lo, nom_hi = iv_fn(lo, hi)
    act_lo, act_hi = iv_fn(blo, bhi)
    L = deriv_abs_sup(blo, bhi)
    M = max(abs(act_lo), abs(act_hi))
    if math.isfinite(L) and math.isfinite(M):
        err = _up(L * e_in + kappa * _ulp(M))
    else:
        err = _up(max(abs(act_lo - nom_hi), abs(act_hi - nom_lo)))
    return ((nom_lo, nom_hi), err)


# ---------------------------------------------------------------------------
# Dimension rules
# ---------------------------------------------------------------------------

def _dim_dimensionless(dims: Sequence[Dim]) -> Dim:
    for d in dims:
        if any(x != 0 for x in d):
            raise ValueError(f"HIN_DIMENSIONAL_MISMATCH: expected dimensionless argument, got {d}")
    return ZERO_DIM


def _dim_same_as_args(dims: Sequence[Dim]) -> Dim:
    if not dims:
        return ZERO_DIM
    first = dims[0]
    for d in dims[1:]:
        if d != first:
            raise ValueError(f"HIN_DIMENSIONAL_MISMATCH: mismatched operand dimensions {first} vs {d}")
    return first


def _dim_sqrt(dims: Sequence[Dim]) -> Dim:
    if len(dims) != 1:
        raise ValueError("sqrt expects 1 argument")
    return tuple(x / 2 for x in dims[0])  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Interval extensions (returns (lo, hi) tuples, outward rounded)
# ---------------------------------------------------------------------------

def _iv_exp(lo: float, hi: float) -> Tuple[float, float]:
    if hi > 709.0:
        raise DomainError("EXP_OVERFLOW")
    return (_down(math.exp(lo), 16), _up(math.exp(hi), 16))


def _iv_log(lo: float, hi: float) -> Tuple[float, float]:
    if lo <= 0.0:
        raise DomainError("LOG_NONPOSITIVE_DOMAIN")
    return (_down(math.log(lo), 16), _up(math.log(hi), 16))


def _iv_log10(lo: float, hi: float) -> Tuple[float, float]:
    if lo <= 0.0:
        raise DomainError("LOG10_NONPOSITIVE_DOMAIN")
    return (_down(math.log10(lo), 16), _up(math.log10(hi), 16))


def _iv_sqrt(lo: float, hi: float) -> Tuple[float, float]:
    if lo < 0.0:
        raise DomainError("SQRT_NEGATIVE_DOMAIN")
    return (_down(math.sqrt(lo), 16) if lo > 0.0 else 0.0, _up(math.sqrt(hi), 16))


def _iv_sin(lo: float, hi: float) -> Tuple[float, float]:
    if hi - lo >= 2.0 * math.pi:
        return (-1.0, 1.0)
    pts = [math.sin(lo), math.sin(hi)]
    k0 = math.ceil((lo - math.pi / 2.0) / math.pi)
    k1 = math.floor((hi - math.pi / 2.0) / math.pi)
    for k in range(k0, k1 + 1):
        pts.append(1.0 if k % 2 == 0 else -1.0)
    return (max(-1.0, _down(min(pts), 16)), min(1.0, _up(max(pts), 16)))


def _iv_cos(lo: float, hi: float) -> Tuple[float, float]:
    if hi - lo >= 2.0 * math.pi:
        return (-1.0, 1.0)
    pts = [math.cos(lo), math.cos(hi)]
    k0 = math.ceil(lo / math.pi)
    k1 = math.floor(hi / math.pi)
    for k in range(k0, k1 + 1):
        pts.append(1.0 if k % 2 == 0 else -1.0)
    return (max(-1.0, _down(min(pts), 16)), min(1.0, _up(max(pts), 16)))


def _iv_tan(lo: float, hi: float) -> Tuple[float, float]:
    """Pole-safe interval tangent: cos(x) must exclude 0 over [lo, hi] and width < 3.14159."""
    if (hi - lo) >= 3.14159:
        raise DomainError("TAN_POLE_IN_DOMAIN: interval width >= pi")
    c_lo, c_hi = _iv_cos(lo, hi)
    if c_lo <= 0.0 <= c_hi:
        raise DomainError("TAN_POLE_IN_DOMAIN: cos(x) interval contains 0")
    t_lo = math.sin(lo) / math.cos(lo)
    t_hi = math.sin(hi) / math.cos(hi)
    return (_down(min(t_lo, t_hi), 16), _up(max(t_lo, t_hi), 16))


def _iv_tanh(lo: float, hi: float) -> Tuple[float, float]:
    return (max(-1.0, _down(math.tanh(lo), 16)), min(1.0, _up(math.tanh(hi), 16)))


def _iv_sinh(lo: float, hi: float) -> Tuple[float, float]:
    if max(abs(lo), abs(hi)) > 709.0:
        raise DomainError("SINH_OVERFLOW")
    return (_down(math.sinh(lo), 16), _up(math.sinh(hi), 16))


def _iv_cosh(lo: float, hi: float) -> Tuple[float, float]:
    if max(abs(lo), abs(hi)) > 709.0:
        raise DomainError("COSH_OVERFLOW")
    if lo <= 0.0 <= hi:
        return (1.0, _up(max(math.cosh(lo), math.cosh(hi)), 16))
    c1, c2 = math.cosh(lo), math.cosh(hi)
    return (_down(min(c1, c2), 16), _up(max(c1, c2), 16))


def _iv_abs(lo: float, hi: float) -> Tuple[float, float]:
    if lo <= 0.0 <= hi:
        return (0.0, max(abs(lo), abs(hi)))
    return (min(abs(lo), abs(hi)), max(abs(lo), abs(hi)))


# ---------------------------------------------------------------------------
# Strict numeric evaluators
# ---------------------------------------------------------------------------

def _eval_tan(x: float) -> float:
    c = math.cos(x)
    if abs(c) < 1e-15:
        raise DomainError("TAN_POLE_IN_DOMAIN")
    v = math.tan(x)
    if not math.isfinite(v):
        raise DomainError("TAN_POLE_IN_DOMAIN")
    return v


def _eval_log(x: float) -> float:
    if x <= 0.0:
        raise DomainError("LOG_NONPOSITIVE_DOMAIN")
    return math.log(x)


def _eval_log10(x: float) -> float:
    if x <= 0.0:
        raise DomainError("LOG10_NONPOSITIVE_DOMAIN")
    return math.log10(x)


def _eval_sqrt(x: float) -> float:
    if x < 0.0:
        raise DomainError("SQRT_NEGATIVE_DOMAIN")
    return math.sqrt(x)


# ---------------------------------------------------------------------------
# Six-member FunctionSpec & MathFunctionSpec compatibility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionSpec:
    parse_name: str
    arity: Tuple[int, Optional[int]]
    eval: Optional[Callable[..., float]]
    interval_ext: Optional[Callable[[Sequence[Any]], Tuple[float, float]]]
    roundoff_bound: Optional[Callable[[Sequence[Tuple[Tuple[float, float], float]]], Tuple[Tuple[float, float], float]]]
    symbolic_derivative: Optional[Callable[[Sequence[Expr]], List[Expr]]]
    dimension_rule: Optional[Callable[[Sequence[Dim]], Dim]]
    py_name: str
    cpp_name: str
    rust_name: str
    csel_name: Optional[str] = None
    domain: Optional[Callable[[Sequence[Any]], bool]] = None
    formal_supported: bool = True
    prompt_advertised: bool = True
    alias_of: Optional[str] = None
    is_stochastic: bool = False

    @property
    def name(self) -> str:
        return self.parse_name

    @property
    def min_arity(self) -> int:
        return self.arity[0]

    @property
    def max_arity(self) -> Optional[int]:
        return self.arity[1]


# Backward-compatible alias
MathFunctionSpec = FunctionSpec

FUNCTION_REGISTRY: Dict[str, FunctionSpec] = {}


def register(spec: FunctionSpec) -> FunctionSpec:
    """Register a FunctionSpec; enforces all 6 mandatory members when `formal_supported=True`."""
    if spec.formal_supported and not spec.is_stochastic and spec.alias_of is None:
        missing = [
            fld
            for fld in (
                "eval",
                "interval_ext",
                "roundoff_bound",
                "symbolic_derivative",
                "dimension_rule",
                "domain",
            )
            if getattr(spec, fld) is None
        ]
        if missing:
            raise RegistryError(
                f"FunctionSpec {spec.parse_name!r} with formal_supported=True is missing mandatory members: {missing}"
            )
    FUNCTION_REGISTRY[spec.parse_name] = spec
    return spec


def _c(n: int, d: int = 1) -> Const:
    return Const(num=str(n), den=str(d))


def _make_unary_iv(fn: Callable[[float, float], Tuple[float, float]]):
    def _wrap(args: Sequence[Any]) -> Tuple[float, float]:
        lo, hi = _iv_lo_hi(args[0])
        return fn(lo, hi)
    return _wrap


def _make_unary_ro(
    iv_fn: Callable[[float, float], Tuple[float, float]],
    deriv_sup: Callable[[float, float], float],
):
    def _wrap(args: Sequence[Tuple[Tuple[float, float], float]]) -> Tuple[Tuple[float, float], float]:
        (lo, hi), e_in = args[0]
        return _nf19_unary_roundoff(lo, hi, e_in, iv_fn, deriv_sup)
    return _wrap


# --- Core 13 formal functions (exact match with Rust `functions.rs`) ---

register(FunctionSpec(
    parse_name="exp",
    arity=(1, 1),
    eval=math.exp,
    interval_ext=_make_unary_iv(_iv_exp),
    roundoff_bound=_make_unary_ro(_iv_exp, lambda _lo, hi: math.exp(hi) if hi <= 709.0 else float("inf")),
    symbolic_derivative=lambda args: [Call(fn="exp", args=(args[0],))],
    dimension_rule=_dim_dimensionless,
    py_name="math.exp",
    cpp_name="std::exp",
    rust_name="f64::exp",
    csel_name="exp",
    domain=lambda args: _iv_lo_hi(args[0])[1] <= 709.0,
))

register(FunctionSpec(
    parse_name="log",
    arity=(1, 1),
    eval=_eval_log,
    interval_ext=_make_unary_iv(_iv_log),
    roundoff_bound=_make_unary_ro(_iv_log, lambda lo, _hi: (1.0 / lo) if lo > 0.0 else float("inf")),
    symbolic_derivative=lambda args: [Binary(op="div", lhs=_c(1), rhs=args[0])],
    dimension_rule=_dim_dimensionless,
    py_name="math.log",
    cpp_name="std::log",
    rust_name="f64::ln",
    csel_name="log",
    domain=lambda args: _iv_lo_hi(args[0])[0] > 0.0,
))

register(FunctionSpec(
    parse_name="log10",
    arity=(1, 1),
    eval=_eval_log10,
    interval_ext=_make_unary_iv(_iv_log10),
    roundoff_bound=_make_unary_ro(
        _iv_log10,
        lambda lo, _hi: (1.0 / (lo * math.log(10.0))) if lo > 0.0 else float("inf"),
    ),
    symbolic_derivative=lambda args: [
        Binary(op="div", lhs=_c(1), rhs=Binary(op="mul", lhs=args[0], rhs=Call(fn="log", args=(_c(10),))))
    ],
    dimension_rule=_dim_dimensionless,
    py_name="math.log10",
    cpp_name="std::log10",
    rust_name="f64::log10",
    csel_name="log10",
    domain=lambda args: _iv_lo_hi(args[0])[0] > 0.0,
))

register(FunctionSpec(
    parse_name="sqrt",
    arity=(1, 1),
    eval=_eval_sqrt,
    interval_ext=_make_unary_iv(_iv_sqrt),
    roundoff_bound=_make_unary_ro(_iv_sqrt, lambda lo, _hi: (0.5 / math.sqrt(lo)) if lo > 0.0 else float("inf")),
    symbolic_derivative=lambda args: [
        Binary(op="div", lhs=_c(1), rhs=Binary(op="mul", lhs=_c(2), rhs=Call(fn="sqrt", args=(args[0],))))
    ],
    dimension_rule=_dim_sqrt,
    py_name="math.sqrt",
    cpp_name="std::sqrt",
    rust_name="f64::sqrt",
    csel_name="sqrt",
    domain=lambda args: _iv_lo_hi(args[0])[0] >= 0.0,
))

register(FunctionSpec(
    parse_name="sin",
    arity=(1, 1),
    eval=math.sin,
    interval_ext=_make_unary_iv(_iv_sin),
    roundoff_bound=_make_unary_ro(_iv_sin, lambda _lo, _hi: 1.0),
    symbolic_derivative=lambda args: [Call(fn="cos", args=(args[0],))],
    dimension_rule=_dim_dimensionless,
    py_name="math.sin",
    cpp_name="std::sin",
    rust_name="f64::sin",
    csel_name="sin",
    domain=lambda args: math.isfinite(_iv_lo_hi(args[0])[0]) and math.isfinite(_iv_lo_hi(args[0])[1]),
))

register(FunctionSpec(
    parse_name="cos",
    arity=(1, 1),
    eval=math.cos,
    interval_ext=_make_unary_iv(_iv_cos),
    roundoff_bound=_make_unary_ro(_iv_cos, lambda _lo, _hi: 1.0),
    symbolic_derivative=lambda args: [Unary(op="neg", arg=Call(fn="sin", args=(args[0],)))],
    dimension_rule=_dim_dimensionless,
    py_name="math.cos",
    cpp_name="std::cos",
    rust_name="f64::cos",
    csel_name="cos",
    domain=lambda args: math.isfinite(_iv_lo_hi(args[0])[0]) and math.isfinite(_iv_lo_hi(args[0])[1]),
))

def _tan_deriv_sup(lo: float, hi: float) -> float:
    t_lo, t_hi = _iv_tan(lo, hi)
    m = max(abs(t_lo), abs(t_hi))
    return 1.0 + m * m

def _tan_domain(args: Sequence[Any]) -> bool:
    lo, hi = _iv_lo_hi(args[0])
    if (hi - lo) >= 3.14159:
        return False
    c_lo, c_hi = _iv_cos(lo, hi)
    return not (c_lo <= 0.0 <= c_hi)

register(FunctionSpec(
    parse_name="tan",
    arity=(1, 1),
    eval=_eval_tan,
    interval_ext=_make_unary_iv(_iv_tan),
    roundoff_bound=_make_unary_ro(_iv_tan, _tan_deriv_sup),
    symbolic_derivative=lambda args: [
        Binary(op="add", lhs=_c(1), rhs=Binary(op="pow", lhs=Call(fn="tan", args=(args[0],)), rhs=_c(2)))
    ],
    dimension_rule=_dim_dimensionless,
    py_name="math.tan",
    cpp_name="std::tan",
    rust_name="f64::tan",
    csel_name="tan",
    domain=_tan_domain,
))

register(FunctionSpec(
    parse_name="tanh",
    arity=(1, 1),
    eval=math.tanh,
    interval_ext=_make_unary_iv(_iv_tanh),
    roundoff_bound=_make_unary_ro(_iv_tanh, lambda _lo, _hi: 1.0),
    symbolic_derivative=lambda args: [
        Binary(op="sub", lhs=_c(1), rhs=Binary(op="pow", lhs=Call(fn="tanh", args=(args[0],)), rhs=_c(2)))
    ],
    dimension_rule=_dim_dimensionless,
    py_name="math.tanh",
    cpp_name="std::tanh",
    rust_name="f64::tanh",
    csel_name="tanh",
    domain=lambda args: math.isfinite(_iv_lo_hi(args[0])[0]) and math.isfinite(_iv_lo_hi(args[0])[1]),
))

register(FunctionSpec(
    parse_name="sinh",
    arity=(1, 1),
    eval=math.sinh,
    interval_ext=_make_unary_iv(_iv_sinh),
    roundoff_bound=_make_unary_ro(_iv_sinh, lambda lo, hi: max(math.cosh(lo), math.cosh(hi)) if max(abs(lo), abs(hi)) <= 709.0 else float("inf")),
    symbolic_derivative=lambda args: [Call(fn="cosh", args=(args[0],))],
    dimension_rule=_dim_dimensionless,
    py_name="math.sinh",
    cpp_name="std::sinh",
    rust_name="f64::sinh",
    csel_name="sinh",
    domain=lambda args: max(abs(_iv_lo_hi(args[0])[0]), abs(_iv_lo_hi(args[0])[1])) <= 709.0,
))

register(FunctionSpec(
    parse_name="cosh",
    arity=(1, 1),
    eval=math.cosh,
    interval_ext=_make_unary_iv(_iv_cosh),
    roundoff_bound=_make_unary_ro(_iv_cosh, lambda lo, hi: max(abs(math.sinh(lo)), abs(math.sinh(hi))) if max(abs(lo), abs(hi)) <= 709.0 else float("inf")),
    symbolic_derivative=lambda args: [Call(fn="sinh", args=(args[0],))],
    dimension_rule=_dim_dimensionless,
    py_name="math.cosh",
    cpp_name="std::cosh",
    rust_name="f64::cosh",
    csel_name="cosh",
    domain=lambda args: max(abs(_iv_lo_hi(args[0])[0]), abs(_iv_lo_hi(args[0])[1])) <= 709.0,
))

register(FunctionSpec(
    parse_name="abs",
    arity=(1, 1),
    eval=abs,
    interval_ext=_make_unary_iv(_iv_abs),
    roundoff_bound=lambda args: (_iv_abs(*args[0][0]), args[0][1]),
    symbolic_derivative=lambda args: [
        Cond(pred=Cmp(op="ge", lhs=args[0], rhs=_c(0)), then_expr=_c(1), else_expr=_c(-1))
    ],
    dimension_rule=_dim_same_as_args,
    py_name="abs",
    cpp_name="std::abs",
    rust_name="f64::abs",
    csel_name="abs",
    domain=lambda _args: True,
))

def _iv_min(args: Sequence[Any]) -> Tuple[float, float]:
    los, his = zip(*(_iv_lo_hi(a) for a in args))
    return (min(los), min(his))

def _iv_max(args: Sequence[Any]) -> Tuple[float, float]:
    los, his = zip(*(_iv_lo_hi(a) for a in args))
    return (max(los), max(his))

register(FunctionSpec(
    parse_name="min",
    arity=(2, None),
    eval=min,
    interval_ext=_iv_min,
    roundoff_bound=lambda args: (_iv_min([a[0] for a in args]), max(a[1] for a in args)),
    symbolic_derivative=lambda args: [
        Cond(pred=Cmp(op="le", lhs=args[0], rhs=args[1]), then_expr=_c(1), else_expr=_c(0)),
        Cond(pred=Cmp(op="le", lhs=args[0], rhs=args[1]), then_expr=_c(0), else_expr=_c(1)),
    ],
    dimension_rule=_dim_same_as_args,
    py_name="min",
    cpp_name="std::min",
    rust_name="f64::min",
    csel_name="min",
    domain=lambda _args: True,
))

register(FunctionSpec(
    parse_name="max",
    arity=(2, None),
    eval=max,
    interval_ext=_iv_max,
    roundoff_bound=lambda args: (_iv_max([a[0] for a in args]), max(a[1] for a in args)),
    symbolic_derivative=lambda args: [
        Cond(pred=Cmp(op="ge", lhs=args[0], rhs=args[1]), then_expr=_c(1), else_expr=_c(0)),
        Cond(pred=Cmp(op="ge", lhs=args[0], rhs=args[1]), then_expr=_c(0), else_expr=_c(1)),
    ],
    dimension_rule=_dim_same_as_args,
    py_name="max",
    cpp_name="std::max",
    rust_name="f64::max",
    csel_name="max",
    domain=lambda _args: True,
))

# --- Aliases & sugar (canonicalized before verification) ---
register(FunctionSpec(
    parse_name="ln", arity=(1, 1), eval=_eval_log, interval_ext=None, roundoff_bound=None,
    symbolic_derivative=None, dimension_rule=_dim_dimensionless,
    py_name="math.log", cpp_name="std::log", rust_name="f64::ln", csel_name="log",
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="log",
))
register(FunctionSpec(
    parse_name="step", arity=(1, 1), eval=lambda x: 1.0 if x >= 0.0 else 0.0,
    interval_ext=None, roundoff_bound=None, symbolic_derivative=None, dimension_rule=_dim_dimensionless,
    py_name="step", cpp_name="invar_step_fn", rust_name="invar_step_fn", csel_name="where",
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="cond",
))
register(FunctionSpec(
    parse_name="clamp", arity=(3, 3), eval=lambda x, lo, hi: min(hi, max(lo, x)),
    interval_ext=None, roundoff_bound=None, symbolic_derivative=None, dimension_rule=_dim_same_as_args,
    py_name="clamp", cpp_name="std::clamp", rust_name="f64::clamp", csel_name="smooth_clamp",
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="min_max",
))
register(FunctionSpec(
    parse_name="pow", arity=(2, 2), eval=pow,
    interval_ext=None, roundoff_bound=None, symbolic_derivative=None, dimension_rule=None,
    py_name="pow", cpp_name="std::pow", rust_name="f64::powf", csel_name="pow",
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="binary_pow",
))
register(FunctionSpec(
    parse_name="logistic_sigmoid", arity=(1, 1), eval=lambda x: 1.0 / (1.0 + math.exp(-x)),
    interval_ext=None, roundoff_bound=None, symbolic_derivative=None, dimension_rule=_dim_dimensionless,
    py_name="logistic_sigmoid", cpp_name="logistic_sigmoid", rust_name="logistic_sigmoid", csel_name="logistic_sigmoid",
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="exp_rational",
))
register(FunctionSpec(
    parse_name="kl_term", arity=(2, 2), eval=lambda p, q: p * math.log(p / q),
    interval_ext=None, roundoff_bound=None, symbolic_derivative=None, dimension_rule=_dim_dimensionless,
    py_name="kl_term", cpp_name="kl_term", rust_name="kl_term", csel_name="kl_term",
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="log_ratio",
))
register(FunctionSpec(
    parse_name="hill_eq", arity=(3, 3), eval=lambda x, k, n: (x ** n) / (k ** n + x ** n),
    interval_ext=None, roundoff_bound=None, symbolic_derivative=None, dimension_rule=_dim_dimensionless,
    py_name="hill_eq", cpp_name="hill_eq", rust_name="hill_eq", csel_name="hill_eq",
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="hill",
))
register(FunctionSpec(
    parse_name="michaelis_menten", arity=(3, 3), eval=lambda s, vmax, km: vmax * s / (km + s),
    interval_ext=None, roundoff_bound=None, symbolic_derivative=None, dimension_rule=None,
    py_name="michaelis_menten", cpp_name="michaelis_menten", rust_name="michaelis_menten", csel_name="michaelis_menten",
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="mm",
))
register(FunctionSpec(
    parse_name="implies", arity=(2, 2), eval=lambda a, b: float((not bool(a)) or bool(b)),
    interval_ext=None, roundoff_bound=None, symbolic_derivative=None, dimension_rule=None,
    py_name="implies", cpp_name="implies", rust_name="implies", csel_name=None,
    domain=None, formal_supported=True, prompt_advertised=True, alias_of="bool_implies",
))
# Seeded stochastic primitives (allowed in MONTE_CARLO / StochasticSimulation)
for _st_name, _py_fn, _c_fn in (
    ("rand_uniform01", "_splitmix64_uniform01", "invar_rand_uniform01"),
    ("rand_normal01", "_splitmix64_normal01", "invar_rand_normal01"),
    ("rand_exponential1", "_splitmix64_exponential1", "invar_rand_exponential1"),
):
    register(FunctionSpec(
        parse_name=_st_name, arity=(0, 1), eval=None, interval_ext=None, roundoff_bound=None,
        symbolic_derivative=None, dimension_rule=_dim_dimensionless,
        py_name=_py_fn, cpp_name=_c_fn, rust_name=_c_fn, csel_name=None,
        domain=None, formal_supported=True, prompt_advertised=True, is_stochastic=True,
    ))


BUILTIN_CONSTANTS: Tuple[str, ...] = ("pi", "e_euler")


def core_formal_functions() -> Dict[str, FunctionSpec]:
    """Return the non-alias, non-stochastic 6-member core formal functions (matches Rust `functions.rs`)."""
    return {
        name: spec
        for name, spec in FUNCTION_REGISTRY.items()
        if spec.formal_supported and not spec.is_stochastic and spec.alias_of is None
    }


def prompt_function_vocabulary() -> List[str]:
    """Exact ordered list of built-in functions advertised in synthesis prompts."""
    return sorted(
        name for name, spec in FUNCTION_REGISTRY.items()
        if spec.prompt_advertised
    )


def formal_gate_vocabulary() -> List[str]:
    """Exact ordered list of functions accepted by the formal verification gate."""
    return sorted(
        name for name, spec in FUNCTION_REGISTRY.items()
        if spec.formal_supported
    )


def formal_gate_arities() -> Dict[str, Tuple[int, Optional[int]]]:
    """Map of formal-gate function name -> (min_arity, max_arity_or_None)."""
    return {
        name: (spec.min_arity, spec.max_arity)
        for name, spec in FUNCTION_REGISTRY.items()
        if spec.formal_supported
    }


def lower_nary_binary_tree(fn_name: str, args: Sequence[str]) -> str:
    """Lower n-ary ``min``/``max`` into a balanced binary tree."""
    items = list(args)
    if len(items) < 2:
        raise ValueError(f"{fn_name} requires at least 2 arguments, got {len(items)}")
    while len(items) > 1:
        nxt: List[str] = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                nxt.append(f"{fn_name}({items[i]}, {items[i + 1]})")
            else:
                nxt.append(items[i])
        items = nxt
    return items[0]


def lower_nary_ast_tree(fn_name: str, args: Sequence[ast.expr]) -> ast.expr:
    """AST equivalent of ``lower_nary_binary_tree`` for balanced binary min/max."""
    items = list(args)
    if len(items) < 2:
        raise ValueError(f"{fn_name} requires at least 2 arguments")
    while len(items) > 1:
        nxt: List[ast.expr] = []
        for i in range(0, len(items), 2):
            if i + 1 < len(items):
                nxt.append(
                    ast.Call(
                        func=ast.Name(id=fn_name, ctx=ast.Load()),
                        args=[items[i], items[i + 1]],
                        keywords=[],
                    )
                )
            else:
                nxt.append(items[i])
        items = nxt
    return items[0]


class FormalFunctionNormalizer(ast.NodeTransformer):
    """Rewrite registered convenience functions (ln, clamp, n-ary min/max) into canonical formal AST."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.keywords:
            return node
        name = node.func.id
        if name == "ln" and len(node.args) == 1:
            return ast.copy_location(
                ast.Call(func=ast.Name(id="log", ctx=ast.Load()), args=list(node.args), keywords=[]),
                node,
            )
        if name == "clamp" and len(node.args) == 3:
            x_arg, lo_arg, hi_arg = node.args
            inner = ast.Call(func=ast.Name(id="max", ctx=ast.Load()), args=[lo_arg, x_arg], keywords=[])
            return ast.copy_location(
                ast.Call(func=ast.Name(id="min", ctx=ast.Load()), args=[hi_arg, inner], keywords=[]),
                node,
            )
        if name in ("min", "max") and len(node.args) > 2:
            return ast.copy_location(lower_nary_ast_tree(name, node.args), node)
        return node
