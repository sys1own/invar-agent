"""Numerical core routines: RK4 stepping, Newton-Raphson, LU decomposition."""

from __future__ import annotations

from typing import Callable, List


def rk4_step(state: List[float], params: List[float], dt: float,
             deriv: Callable[[List[float], List[float], float], List[float]]) -> List[float]:
    """Classic 4th-order Runge-Kutta step.

    ``deriv(state, params, t)`` returns d(state)/dt. ``params[0]`` is the
    decay rate convention used by the research kernels.
    """
    n = len(state)
    t = 0.0

    def scaled(s: List[float], k: List[float], h: float) -> List[float]:
        return [s[i] + h * k[i] for i in range(n)]

    k1 = deriv(state, params, t)
    k2 = deriv(scaled(state, k1, dt / 2.0), params, t + dt / 2.0)
    k3 = deriv(scaled(state, k2, dt / 2.0), params, t + dt / 2.0)
    k4 = deriv(scaled(state, k3, dt), params, t + dt)
    out = [0.0] * n
    for i in range(n):
        out_state_i = state[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        out[i] = out_state_i
    return out


def newton_raphson(f: Callable[[float], float], f_prime: Callable[[float], float],
                   x0: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = f_prime(x)
        if dfx == 0.0:
            raise ZeroDivisionError("Newton-Raphson: zero derivative")
        x = x - fx / dfx
    raise ArithmeticError(f"Newton-Raphson did not converge in {max_iter} iterations")


def lu_decompose(a: List[List[float]]):
    """LU decomposition with partial pivoting.

    Returns ``(lu, pivots)`` where ``lu`` is the combined L/U matrix and
    ``pivots[i]`` is the row swapped with row ``i`` at step ``i``.
    """
    n = len(a)
    lu = [row[:] for row in a]
    perm = list(range(n))  # perm[i] = original row index currently at row i
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(lu[i][k]))
        if abs(lu[pivot][k]) < 1e-15:
            raise ZeroDivisionError("LU decomposition: singular matrix")
        if pivot != k:
            lu[k], lu[pivot] = lu[pivot], lu[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]
        for i in range(k + 1, n):
            lu[i][k] /= lu[k][k]
            for j in range(k + 1, n):
                lu[i][j] -= lu[i][k] * lu[k][j]
    return lu, perm


def lu_solve(lu_and_perm, b: List[float]) -> List[float]:
    lu, perm = lu_and_perm
    n = len(lu)
    # Apply the recorded row permutation to b.
    pb = [b[perm[i]] for i in range(n)]
    # Forward substitution (unit lower).
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(lu[i][j] * y[j] for j in range(i))
    # Back substitution.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(lu[i][j] * x[j] for j in range(i + 1, n))) / lu[i][i]
    return x
