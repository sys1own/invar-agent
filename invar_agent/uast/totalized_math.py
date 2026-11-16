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
