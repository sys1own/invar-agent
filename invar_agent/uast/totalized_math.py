"""Continuous Totalized Smooth Operators in Python and v9_transpiler_friendly emitter form."""

from __future__ import annotations

import math

SAFE_MAX_VAL = 1e300
SAFE_DIV_EPS = 1e-12
SAFE_LOG_EPS = 1e-30


def safe_div(x: float, y: float, eps: float = SAFE_DIV_EPS) -> float:
    """Continuous totalized division: x / copysign(max(|y|, eps), y), with copysign(eps, 0.0) = +eps."""
    denom_mag = abs(y) if abs(y) >= eps else eps
    denom = -denom_mag if y < 0.0 else denom_mag
    res = x / denom
    if res > SAFE_MAX_VAL:
        return SAFE_MAX_VAL
    if res < -SAFE_MAX_VAL:
        return -SAFE_MAX_VAL
    return res


def safe_exp(x: float) -> float:
    if x > 80.0:
        x = 80.0
    elif x < -80.0:
        x = -80.0
    return math.exp(x)


def safe_log(x: float) -> float:
    return math.log(x if x > SAFE_LOG_EPS else SAFE_LOG_EPS)


def safe_sqrt(x: float) -> float:
    return math.sqrt(x if x > 0.0 else 0.0)


def safe_acos(x: float) -> float:
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return math.acos(x)


def safe_asin(x: float) -> float:
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return math.asin(x)


def safe_sin(x: float) -> float:
    return math.sin(x)


def safe_cos(x: float) -> float:
    return math.cos(x)


def safe_tan(x: float) -> float:
    if x > 1.570796:
        x = 1.570796
    elif x < -1.570796:
        x = -1.570796
    return math.tan(x)


def safe_tanh(x: float) -> float:
    return math.tanh(x)


def safe_pow(x: float, y: float) -> float:
    """Totalized power: exact integer power for integer y (preserving negative bases),
    otherwise safe_exp(y * safe_log(abs(x)))."""
    if abs(y - round(y)) < 1e-12 and abs(y) <= 1024.0:
        exp_i = int(round(y))
        if exp_i == 0:
            return 1.0
        if exp_i > 0:
            val = x ** exp_i
            if val > SAFE_MAX_VAL:
                return SAFE_MAX_VAL
            if val < -SAFE_MAX_VAL:
                return -SAFE_MAX_VAL
            return float(val)
        return safe_div(1.0, x ** (-exp_i))
    return safe_exp(y * safe_log(abs(x)))


def log1p(x: float) -> float:
    return math.log1p(x if x > -1.0 + SAFE_LOG_EPS else -1.0 + SAFE_LOG_EPS)


def expm1(x: float) -> float:
    if x > 80.0:
        x = 80.0
    elif x < -80.0:
        x = -80.0
    return math.expm1(x)


def smooth_clamp(x: float, a: float, b: float) -> float:
    lo = a if a <= b else b
    hi = b if a <= b else a
    mid = 0.5 * (lo + hi)
    sig = 1.0 / (1.0 + safe_exp(-10.0 * (x - mid)))
    return lo + (hi - lo) * sig


def where(cond: float, t_val: float, f_val: float) -> float:
    return t_val if cond > 0.0 else f_val


def logistic_sigmoid(x: float) -> float:
    return 1.0 / (1.0 + safe_exp(-x))


def logit(p: float) -> float:
    pc = 1e-15 if p < 1e-15 else (1.0 - 1e-15 if p > 1.0 - 1e-15 else p)
    return math.log(pc / (1.0 - pc))


def hill_eq(c: float, ec50: float, n: float) -> float:
    if c <= 0.0:
        return 0.0
    cn = safe_pow(c if c > 0.0 else 0.0, n)
    kn = safe_pow(ec50 if ec50 > 1e-15 else 1e-15, n)
    return safe_div(cn, cn + kn)


def michaelis_menten(s: float, vmax: float, km: float) -> float:
    sc = s if s > 0.0 else 0.0
    return safe_div(vmax * sc, (km if km > 1e-15 else 1e-15) + sc)


def kl_term(p: float, q: float) -> float:
    qc = q if q > 1e-30 else 1e-30
    if p <= 0.0:
        return qc
    val = p * math.log(p / qc) - p + qc
    return val if val > 0.0 else 0.0


def shannon_term(p: float) -> float:
    if p <= 0.0:
        return 0.0
    return -p * math.log(p)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865475))


def norm_ppf(p: float) -> float:
    """Rational Acklam inverse normal CDF on clamp(p, 1e-15, 1.0 - 1e-15)."""
    pc = 1e-15 if p < 1e-15 else (1.0 - 1e-15 if p > 1.0 - 1e-15 else p)
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low
    if pc < p_low:
        q = math.sqrt(-2.0 * math.log(pc))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    if pc <= p_high:
        q = pc - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - pc))
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


def gammainc_reg(a: float, x: float) -> float:
    ac = a if a > 1e-12 else 1e-12
    if x <= 0.0:
        return 0.0
    ratio = (x / ac) ** (1.0 / 3.0)
    mu = 1.0 - 1.0 / (9.0 * ac)
    sigma = math.sqrt(1.0 / (9.0 * ac))
    return norm_cdf((ratio - mu) / (sigma if sigma > 1e-15 else 1e-15))


def betainc_reg(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ac = a if a > 1e-12 else 1e-12
    bc = b if b > 1e-12 else 1e-12
    mean = ac / (ac + bc)
    var = (ac * bc) / (((ac + bc) ** 2) * (ac + bc + 1.0))
    sigma = math.sqrt(var)
    return norm_cdf((x - mean) / (sigma if sigma > 1e-15 else 1e-15))


def welford_mean_update(mean: float, x: float, k: float) -> float:
    kc = k if k >= 1.0 else 1.0
    return mean + (x - mean) / kc


def welford_m2_update(m2: float, mean_old: float, mean_new: float, x: float) -> float:
    return m2 + (x - mean_old) * (x - mean_new)


def welford_cov_update(c_xy: float, mean_x_old: float, x: float, mean_y_new: float, y: float) -> float:
    return c_xy + (x - mean_x_old) * (y - mean_y_new)


# Pure v9_transpiler_friendly helper functions emitted into unified_module.py
TOTALIZED_HELPERS_PY = '''def safe_div(x, y):
    mag = abs(y) if abs(y) >= 1e-12 else 1e-12
    denom = -mag if y < 0.0 else mag
    res = x / denom
    if res > 1e300:
        return 1e300
    if res < -1e300:
        return -1e300
    return res


def safe_exp(x):
    if x > 80.0:
        x = 80.0
    elif x < -80.0:
        x = -80.0
    return math.exp(x)


def safe_log(x):
    return math.log(x if x > 1e-30 else 1e-30)


def safe_sqrt(x):
    return math.sqrt(x if x > 0.0 else 0.0)


def safe_acos(x):
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return math.acos(x)


def safe_asin(x):
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return math.asin(x)


def safe_sin(x):
    return math.sin(x)


def safe_cos(x):
    return math.cos(x)


def safe_tan(x):
    if x > 1.570796:
        x = 1.570796
    elif x < -1.570796:
        x = -1.570796
    return math.tan(x)


def safe_tanh(x):
    return math.tanh(x)


def safe_pow(x, y):
    y_round = math.floor(y + 0.5)
    if abs(y - y_round) < 1e-12 and abs(y) <= 1024.0:
        exp_i = int(y_round)
        if exp_i == 0:
            return 1.0
        if exp_i > 0:
            val = math.pow(x, exp_i) if x >= 0.0 or (exp_i % 2 == 0) else -math.pow(-x, exp_i)
            if val > 1e300:
                return 1e300
            if val < -1e300:
                return -1e300
            return val
        pos = math.pow(x, -exp_i) if x >= 0.0 or ((-exp_i) % 2 == 0) else -math.pow(-x, -exp_i)
        return safe_div(1.0, pos)
    return safe_exp(y * safe_log(abs(x)))


def safe_log1p(x):
    return math.log1p(x if x > -1.0 + 1e-30 else -1.0 + 1e-30)


def safe_expm1(x):
    if x > 80.0:
        x = 80.0
    elif x < -80.0:
        x = -80.0
    return math.expm1(x)


def smooth_clamp(x, a, b):
    lo = a if a <= b else b
    hi = b if a <= b else a
    mid = 0.5 * (lo + hi)
    sig = 1.0 / (1.0 + safe_exp(-10.0 * (x - mid)))
    return lo + (hi - lo) * sig


def safe_where(cond, t_val, f_val):
    return t_val if cond > 0.0 else f_val
'''
