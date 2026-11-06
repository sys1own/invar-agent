"""Deterministic hashing: AST normalization, Merkle state trees, wavefront digests."""
from __future__ import annotations

import ast
import hashlib
from typing import Optional

import blake3


def _normalize_source(source: str) -> str:
    try:
        tree = ast.parse(source)
        return ast.dump(tree, annotate_fields=False)
    except SyntaxError:
        return source


class SemanticASTHasher:
    """Hashes AST-normalized source so cosmetic edits do not change digests."""

    @staticmethod
    def hash_ast_normalized(source: str) -> str:
        return blake3.blake3(_normalize_source(source).encode("utf-8")).hexdigest()

    @staticmethod
    def hash_raw(source: str) -> str:
        return blake3.blake3(source.encode("utf-8")).hexdigest()


class MerkleStateTree:
    """Binary Merkle tree over file content hashes."""

    def __init__(self, leaves: list[str]) -> None:
        self.levels: list[list[str]] = [list(leaves)] if leaves else [[]]
        while len(self.levels[-1]) > 1:
            level = self.levels[-1]
            if len(level) % 2 == 1:
                level = level + [level[-1]]
            self.levels[-1] = level
            parents = [
                hashlib.sha256((level[i] + level[i + 1]).encode()).hexdigest()
                for i in range(0, len(level), 2)
            ]
            self.levels.append(parents)
        self.root = self.levels[-1][0] if self.levels[-1] else hashlib.sha256(b"").hexdigest()

    @classmethod
    def from_files(cls, files: dict[str, str]) -> "MerkleStateTree":
        leaves = [SemanticASTHasher.hash_ast_normalized(files[p]) for p in sorted(files)]
        return cls(leaves)

    def root_digest(self) -> str:
        return self.root


def compute_wavefront_digest(node_ids: list[str], patch_hashes: list[str]) -> str:
    h = hashlib.sha256()
    for nid, ph in zip(sorted(node_ids), sorted(patch_hashes)):
        h.update(nid.encode())
        h.update(b"\x00")
        h.update(ph.encode())
        h.update(b"\x00")
    return h.hexdigest()


def file_content_hash(content: str) -> str:
    return blake3.blake3(content.encode("utf-8")).hexdigest()
