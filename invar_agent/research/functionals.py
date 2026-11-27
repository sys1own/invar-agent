"""UPG-004: Trajectory functionals (MAX, MIN, FINAL, PEAK_ABS, PERIOD) with cubic Hermite period estimator."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict

from invar_agent.research import meaningfulness as m

# Reserved semantic tags per UPG-004
TRAJECTORY_MAX_TAG = 252
TRAJECTORY_MIN_TAG = 251
TRAJECTORY_FINAL_TAG = 250
TRAJECTORY_PEAK_ABS_TAG = 249
TRAJECTORY_PERIOD_TAG = 248


class TrajectoryFunctional(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["MAX", "MIN", "FINAL", "PEAK_ABS", "PERIOD"]
    expr_text: str
    window: Optional[Tuple[float, float]] = None
    level_text: Optional[str] = None


class FunctionalError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class FunctionalEstimate:
    value: float
    eps_peak_loc: float
    advisories: Tuple[str, ...] = ()


def check_window_grid_aligned(
    window: Tuple[float, float],
    dt: float,
) -> Tuple[int, int]:
    """Verify window endpoints align to integer grid indices at dt (and 2*dt, dt/2)."""
    t0, t1 = window
    if t1 <= t0 or t0 < 0.0:
        raise FunctionalError("WINDOW_NOT_GRID_ALIGNED", f"Invalid window [{t0}, {t1}]")
    dt_f = Fraction(str(dt))
    r0 = Fraction(str(t0)) / dt_f
    r1 = Fraction(str(t1)) / dt_f
    if r0.denominator != 1 or r1.denominator != 1:
        raise FunctionalError("WINDOW_NOT_GRID_ALIGNED", f"Window [{t0}, {t1}] not aligned to dt={dt}")
    return int(r0.numerator), int(r1.numerator)


def _cubic_hermite_root(
    t0: float, t1: float, g0: float, g1: float, gp0: float, gp1: float
) -> float:
    """Find the unique root in [t0, t1] of the cubic Hermite interpolant for g(t0) <= 0 < g(t1)."""
    h = t1 - t0

    def _eval(u: float) -> float:
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        return h00 * g0 + h10 * h * gp0 + h01 * g1 + h11 * h * gp1

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _eval(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return t0 + 0.5 * (lo + hi) * h


def evaluate_functional(
    op: Literal["MAX", "MIN", "FINAL", "PEAK_ABS", "PERIOD"],
    times: Sequence[float],
    samples: Sequence[float],
    *,
    derivatives: Optional[Sequence[float]] = None,
    derivative_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    window: Optional[Tuple[float, float]] = None,
    level: Optional[float] = None,
    dt: Optional[float] = None,
) -> FunctionalEstimate:
    """Evaluate a trajectory functional and its localization error bound `eps_peak_loc`."""
    if len(times) != len(samples) or len(times) < 2:
        raise FunctionalError("INSUFFICIENT_SAMPLES", "At least 2 time samples are required")

    if window is not None:
        if dt is not None:
            i0, i1 = check_window_grid_aligned(window, dt)
            t_slice = list(times[i0 : i1 + 1])
            y_slice = list(samples[i0 : i1 + 1])
            d_slice = list(derivatives[i0 : i1 + 1]) if derivatives is not None else None
            db_slice = list(derivative_bounds[i0 : i1 + 1]) if derivative_bounds is not None else None
        else:
            idxs = [i for i, t in enumerate(times) if window[0] - 1e-12 <= t <= window[1] + 1e-12]
            t_slice = [times[i] for i in idxs]
            y_slice = [samples[i] for i in idxs]
            d_slice = [derivatives[i] for i in idxs] if derivatives is not None else None
            db_slice = [derivative_bounds[i] for i in idxs] if derivative_bounds is not None else None
    else:
        t_slice = list(times)
        y_slice = list(samples)
        d_slice = list(derivatives) if derivatives is not None else None
        db_slice = list(derivative_bounds) if derivative_bounds is not None else None

    if len(t_slice) < 2:
        raise FunctionalError("INSUFFICIENT_SAMPLES", "Window contains fewer than 2 samples")

    if op == "FINAL":
        return FunctionalEstimate(value=float(y_slice[-1]), eps_peak_loc=0.0)

    if op in ("MAX", "MIN"):
        val, _t_pk, eps_loc = m.quadratic_peak(tuple(t_slice), tuple(y_slice), minimum=(op == "MIN"))
        return FunctionalEstimate(value=float(val), eps_peak_loc=float(eps_loc))

    if op == "PEAK_ABS":
        vmax, _, emax = m.quadratic_peak(tuple(t_slice), tuple(y_slice), minimum=False)
        vmin, _, emin = m.quadratic_peak(tuple(t_slice), tuple(y_slice), minimum=True)
        return FunctionalEstimate(value=float(max(vmax, -vmin)), eps_peak_loc=float(max(emax, emin)))

    if op == "PERIOD":
        lvl = float(level) if level is not None else 0.5 * (min(y_slice) + max(y_slice))
        g = [y - lvl for y in y_slice]
        h = float(t_slice[1] - t_slice[0])
        # Estimate derivatives via central differences if not supplied from sealed RHS
        if d_slice is None:
            d_slice = []
            n = len(g)
            for i in range(n):
                if i == 0:
                    d_slice.append((g[1] - g[0]) / h)
                elif i == n - 1:
                    d_slice.append((g[-1] - g[-2]) / h)
                else:
                    d_slice.append((g[i + 1] - g[i - 1]) / (2.0 * h))

        # Fourth finite difference bound M4 = 2 * max |Delta^4 g| / h^4
        max_d4 = 0.0
        for i in range(len(g) - 4):
            d4 = abs(g[i] - 4.0 * g[i + 1] + 6.0 * g[i + 2] - 4.0 * g[i + 3] + g[i + 4]) / (h ** 4)
            if d4 > max_d4:
                max_d4 = d4
        m4 = 2.0 * max_d4
        e_h = (h ** 4) / 384.0 * m4

        crossings: list[float] = []
        max_dt_err = 0.0
        for i in range(len(g) - 1):
            if g[i] <= 0.0 < g[i + 1]:
                # Transversality check
                if db_slice is not None:
                    d_lo = min(db_slice[i][0], db_slice[i + 1][0])
                    d_hi = max(db_slice[i][1], db_slice[i + 1][1])
                    if d_lo <= 0.0 <= d_hi:
                        raise FunctionalError(
                            "CROSSING_NOT_TRANSVERSAL",
                            f"Derivative interval [{d_lo}, {d_hi}] contains zero on bracket [{t_slice[i]}, {t_slice[i+1]}]",
                        )
                    min_abs_gp = min(abs(d_lo), abs(d_hi))
                else:
                    min_abs_gp = min(abs(d_slice[i]), abs(d_slice[i + 1]))
                    if min_abs_gp <= 1e-12:
                        raise FunctionalError(
                            "CROSSING_NOT_TRANSVERSAL",
                            f"Near-zero crossing slope {min_abs_gp} on bracket [{t_slice[i]}, {t_slice[i+1]}]",
                        )
                t_cross = _cubic_hermite_root(
                    t_slice[i], t_slice[i + 1], g[i], g[i + 1], d_slice[i], d_slice[i + 1]
                )
                crossings.append(t_cross)
                dt_err = e_h / min_abs_gp
                if dt_err > max_dt_err:
                    max_dt_err = dt_err

        cycles = len(crossings) - 1
        if cycles < 3:
            raise FunctionalError(
                "INSUFFICIENT_CYCLES",
                f"PERIOD requires >= 3 complete cycles in window, found {max(0, cycles)}",
            )
        periods = [crossings[i + 1] - crossings[i] for i in range(cycles)]
        mean_period = sum(periods) / cycles
        eps_loc = 2.0 * max_dt_err / cycles
        spread = max(periods) - min(periods)
        if spread > max(5.0 * eps_loc, 1e-3 * mean_period):
            raise FunctionalError(
                "NOT_PERIODIC_IN_WINDOW",
                f"Period spread {spread:.4g} exceeds tolerance for mean period {mean_period:.4g}",
            )
        return FunctionalEstimate(
            value=float(mean_period),
            eps_peak_loc=float(eps_loc),
            advisories=("PERIOD_INTERPOLATION_BOUND_HEURISTIC",),
        )

    raise FunctionalError("UNSUPPORTED_FUNCTIONAL", f"Unknown op: {op!r}")
