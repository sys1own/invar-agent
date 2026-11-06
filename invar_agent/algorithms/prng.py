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
