"""Content-addressable fix cache (SQLite-backed) for verified reflex repairs."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

import blake3


@dataclass(frozen=True)
class FixSignature:
    error_category: str
    normalized_diagnostic: str
    target_ast_pattern: str

    @staticmethod
    def compute(error_category: str, diagnostic_msg: str, source_snippet: str = "") -> "FixSignature":
        normalized = " ".join(diagnostic_msg.split())[:200]
        # Strip line numbers and paths for stability.
        import re

        normalized = re.sub(r"line \d+", "line <N>", normalized)
        normalized = re.sub(r"/[^\s]+/", "<PATH>/", normalized)
        pattern = blake3.blake3(source_snippet.encode()).hexdigest()[:16] if source_snippet else ""
        return FixSignature(error_category=error_category, normalized_diagnostic=normalized, target_ast_pattern=pattern)

    def key(self) -> str:
        raw = f"{self.error_category}|{self.normalized_diagnostic}|{self.target_ast_pattern}"
        return blake3.blake3(raw.encode()).hexdigest()


@dataclass
class FixRecord:
    signature: FixSignature
    patch_payload: dict
    confidence: float = 0.95
    use_count: int = 0
    last_used: float = 0.0


class ContentAddressableFixCache:
    """SQLite-backed store mapping failure signatures to verified patches."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS fixes (
                key TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                diagnostic TEXT NOT NULL,
                pattern TEXT NOT NULL,
                payload TEXT NOT NULL,
                confidence REAL NOT NULL,
                use_count INTEGER NOT NULL DEFAULT 0,
                last_used REAL NOT NULL DEFAULT 0
            )"""
        )
        self._conn.commit()

    def lookup_reflex_patch(
        self,
        error_category: str,
        diagnostic_msg: str,
        source_snippet: str = "",
        min_confidence: float = 0.5,
    ) -> Optional[FixRecord]:
        sig = FixSignature.compute(error_category, diagnostic_msg, source_snippet)
        row = self._conn.execute(
            "SELECT payload, confidence, use_count, last_used FROM fixes WHERE key = ?", (sig.key(),)
        ).fetchone()
        if row is None:
            return None
        payload, confidence, use_count, last_used = row
        if confidence < min_confidence:
            return None
        # Update usage stats.
        self._conn.execute(
            "UPDATE fixes SET use_count = use_count + 1, last_used = ? WHERE key = ?",
            (time.time(), sig.key()),
        )
        self._conn.commit()
        return FixRecord(
            signature=sig,
            patch_payload=json.loads(payload),
            confidence=confidence,
            use_count=use_count + 1,
            last_used=time.time(),
        )

    def record_verified_fix(
        self,
        error_category: str,
        diagnostic_msg: str,
        patch_payload: dict,
        source_snippet: str = "",
        confidence: float = 0.95,
    ) -> str:
        sig = FixSignature.compute(error_category, diagnostic_msg, source_snippet)
        self._conn.execute(
            """INSERT INTO fixes (key, category, diagnostic, pattern, payload, confidence, use_count, last_used)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)
               ON CONFLICT(key) DO UPDATE SET payload = excluded.payload,
                   confidence = MAX(confidence, excluded.confidence)""",
            (sig.key(), sig.error_category, sig.normalized_diagnostic, sig.target_ast_pattern,
             json.dumps(patch_payload), confidence, time.time()),
        )
        self._conn.commit()
        return sig.key()

    def compute_signature(self, error_category: str, diagnostic_msg: str, source_snippet: str = "") -> str:
        return FixSignature.compute(error_category, diagnostic_msg, source_snippet).key()

    def close(self) -> None:
        self._conn.close()
