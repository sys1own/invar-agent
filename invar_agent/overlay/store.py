"""Content-addressable overlay blob store (BLAKE3-keyed, in-memory + optional disk)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import blake3


@dataclass(frozen=True)
class BlobEntry:
    hash: str
    size: int
    path: str = ""


@dataclass
class SnapshotManifest:
    tx_id: str
    files: dict[str, str] = field(default_factory=dict)  # path -> content hash
    boundary_digest: str = ""


class ContentAddressableOverlayStore:
    """BLAKE3 content-addressed store for staged overlay buffers."""

    def __init__(self, root: Optional[os.PathLike] = None) -> None:
        self._blobs: dict[str, str] = {}
        self._meta: dict[str, BlobEntry] = {}
        self.root = Path(root) if root else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_content(content: str | bytes) -> str:
        data = content.encode("utf-8") if isinstance(content, str) else content
        return blake3.blake3(data).hexdigest()

    def put_ast_blob(self, content: str, path: str = "") -> str:
        digest = self.hash_content(content)
        self._blobs[digest] = content
        self._meta[digest] = BlobEntry(hash=digest, size=len(content.encode("utf-8")), path=path)
        if self.root is not None:
            blob_path = self.root / digest[:2] / digest
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            if not blob_path.exists():
                blob_path.write_text(content, encoding="utf-8")
        return digest

    def get_blob(self, digest: str) -> Optional[str]:
        if digest in self._blobs:
            return self._blobs[digest]
        if self.root is not None:
            blob_path = self.root / digest[:2] / digest
            if blob_path.exists():
                content = blob_path.read_text(encoding="utf-8")
                self._blobs[digest] = content
                return content
        return None

    def has_blob(self, digest: str) -> bool:
        return digest in self._blobs or self.get_blob(digest) is not None

    def restore_snapshot(self, manifest: SnapshotManifest,
                         reader) -> dict[str, str]:
        """Restore staged buffers to snapshot contents.

        ``reader(path) -> str`` supplies current-on-disk content for files not
        present in the manifest.
        """
        restored: dict[str, str] = {}
        for path, digest in manifest.files.items():
            content = self.get_blob(digest)
            if content is None:
                content = reader(path)
                self.put_ast_blob(content, path)
            restored[path] = content
        return restored

    def snapshot_files(self, files: dict[str, str]) -> dict[str, str]:
        return {path: self.put_ast_blob(content, path) for path, content in files.items()}

    def __len__(self) -> int:
        return len(self._blobs)
