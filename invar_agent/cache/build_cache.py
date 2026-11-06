"""Hermetic build cache: Merkle-keyed verification artifacts."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import blake3

from ..scheduler.hasher import SemanticASTHasher


@dataclass
class CachedVerification:
    key: str
    passed: bool
    stdout_tail: str = ""
    stderr_tail: str = ""
    exit_code: int = 0
    cached_at: float = 0.0
    hit_count: int = 0


class HermeticBuildCache:
    """Content-addressed cache of sandbox verification results."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS build_cache (
                key TEXT PRIMARY KEY,
                passed INTEGER NOT NULL,
                stdout_tail TEXT NOT NULL DEFAULT '',
                stderr_tail TEXT NOT NULL DEFAULT '',
                exit_code INTEGER NOT NULL DEFAULT 0,
                cached_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self._conn.commit()

    @staticmethod
    def compute_key(files: dict[str, str], toolchain: str = "python3", command: str = "pytest -q") -> str:
        h = blake3.blake3()
        h.update(toolchain.encode())
        h.update(b"\x00")
        h.update(command.encode())
        for path in sorted(files):
            h.update(path.encode())
            h.update(b"\x00")
            h.update(SemanticASTHasher.hash_ast_normalized(files[path]).encode())
            h.update(b"\x00")
        return h.hexdigest()

    def is_node_verification_cached(self, key: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM build_cache WHERE key = ?", (key,)).fetchone()
        return row is not None

    def get(self, key: str) -> Optional[CachedVerification]:
        row = self._conn.execute(
            "SELECT passed, stdout_tail, stderr_tail, exit_code, cached_at, hit_count FROM build_cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        self._conn.execute("UPDATE build_cache SET hit_count = hit_count + 1 WHERE key = ?", (key,))
        self._conn.commit()
        passed, stdout_tail, stderr_tail, exit_code, cached_at, hit_count = row
        return CachedVerification(key=key, passed=bool(passed), stdout_tail=stdout_tail,
                                  stderr_tail=stderr_tail, exit_code=exit_code,
                                  cached_at=cached_at, hit_count=hit_count + 1)

    def put(self, key: str, passed: bool, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self._conn.execute(
            """INSERT INTO build_cache (key, passed, stdout_tail, stderr_tail, exit_code, cached_at, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, 0)
               ON CONFLICT(key) DO UPDATE SET passed = excluded.passed,
                   stdout_tail = excluded.stdout_tail, stderr_tail = excluded.stderr_tail,
                   exit_code = excluded.exit_code, cached_at = excluded.cached_at""",
            (key, int(passed), stdout[-2000:], stderr[-2000:], exit_code, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class WavefrontArtifactCache:
    """In-memory cache of per-wavefront verification artifacts."""

    def __init__(self) -> None:
        self._artifacts: dict[str, dict] = {}

    def store(self, wavefront_id: str, artifact: dict) -> None:
        self._artifacts[wavefront_id] = artifact

    def get(self, wavefront_id: str) -> Optional[dict]:
        return self._artifacts.get(wavefront_id)

    def invalidate(self, wavefront_id: str) -> None:
        self._artifacts.pop(wavefront_id, None)

    def clear(self) -> None:
        self._artifacts.clear()
