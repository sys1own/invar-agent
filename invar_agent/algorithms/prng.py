"""Deterministic PRNG utilities (xoshiro256** style, reference implementation)."""

from __future__ import annotations

MASK = (1 << 64) - 1


def rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK


def xoshiro256plusplus_next(s0: int, s1: int, s2: int, s3: int):
    """Advance the xoshiro256** state and return (s0, s1, s2, s3, result)."""
    result = (rotl((s0 + s3) & MASK, 23) + s0) & MASK
    t = (s1 << 17) & MASK
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = rotl(s3, 45)
    return s0 & MASK, s1 & MASK, s2 & MASK, s3 & MASK, result


def uniform_f64(rand_u64: int) -> float:
    """Map a 64-bit integer to a uniform double in [0, 1)."""
    return (rand_u64 >> 11) * (1.0 / 9007199254740992.0)


def splitmix64(seed: int):
    """Seed a xoshiro256** state from a 64-bit seed."""
    state = seed & MASK

    def nxt() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    return nxt(), nxt(), nxt(), nxt()


def invar_splitmix64_stateless(seed: int | float, step: int | float, channel: int | float) -> int:
    """Collision-free stateless SplitMix64 hash mixing seed, step, and channel with distinct 64-bit Weyl constants."""
    s = int(seed) & MASK
    st = int(step) & MASK
    ch = int(channel) & MASK
    z = (s + ((st * 0x9E3779B97F4A7C15) & MASK) + ((ch * 0xBF58476D1CE4E5B9) & MASK)) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def invar_rand_u01(seed: int | float, step: int | float, channel: int | float) -> float:
    """Stateless uniform variate in [1e-15, 1.0 - 1e-15] (never 0.0 or 1.0)."""
    val = invar_splitmix64_stateless(seed, step, channel)
    u = (val >> 11) * (1.0 / 9007199254740992.0)
    if u <= 1e-15:
        return 1e-15
    if u >= 1.0 - 1e-15:
        return 1.0 - 1e-15
    return u


def invar_rand_normal(seed: int | float, step: int | float, channel: int | float) -> float:
    """Exact Box-Muller standard normal variate using collision-free clamped u1, u2."""
    import math
    ch = int(channel) & MASK
    u1 = invar_rand_u01(seed, step, (ch * 2) & MASK)
    u2 = invar_rand_u01(seed, step, (ch * 2 + 1) & MASK)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(6.2831853071795864769 * u2)


def invar_rand_poisson(seed: int | float, step: int | float, lam: float) -> float:
    """Deterministic Poisson variate (Knuth exact for lam < 30, Gaussian continuity-corrected for lam >= 30)."""
    import math
    if lam <= 0.0:
        return 0.0
    if lam < 30.0:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L and k < 256:
            k += 1
            p *= invar_rand_u01(seed, step, 1000 + k)
        return float(k - 1)
    z = invar_rand_normal(seed, step, 999)
    res = math.floor(lam + math.sqrt(lam) * z + 0.5)
    return res if res > 0.0 else 0.0


def invar_rand_binomial(seed: int | float, step: int | float, n: float, p: float) -> float:
    """Deterministic Binomial(n, p) variate."""
    import math
    if n <= 0.0 or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return math.floor(n)
    if n < 64.0:
        hits = 0
        ni = int(n)
        for i in range(ni):
            if invar_rand_u01(seed, step, 2000 + i) < p:
                hits += 1
        return float(hits)
    mu = n * p
    sigma = math.sqrt(n * p * (1.0 - p))
    z = invar_rand_normal(seed, step, 1999)
    val = math.floor(mu + sigma * z + 0.5)
    if val < 0.0:
        return 0.0
    n_floor = math.floor(n)
    return n_floor if val > n_floor else val

