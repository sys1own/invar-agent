"""Validate every archive member before opening any output file."""
from __future__ import annotations

import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

ALLOWLIST = {".pdf", ".tex", ".md", ".ipynb", ".py", ".csv", ".tsv", ".json", ".zip"}
MAX_ENTRIES = 500
MAX_BYTES = 100 * 1024 * 1024
MAX_RATIO = 100.0


class IngestionError(ValueError):
    status_code = 422

    def __init__(self, code: str, message: str = "", *, details=None):
        self.code = code
        self.details = details
        super().__init__(f"{code}: {message}")


def validate_and_unpack_upload(upload_path: Path, dest_dir: Path) -> list[Path]:
    upload_path, dest_dir = Path(upload_path), Path(dest_dir)
    if upload_path.is_symlink():
        raise IngestionError("SANDBOX_ZIP_SYMLINK_REJECTED", str(upload_path))
    if upload_path.suffix.lower() not in ALLOWLIST:
        raise IngestionError("UPLOAD_DISALLOWED_TYPE", upload_path.name)
    if upload_path.stat().st_size > MAX_BYTES:
        raise IngestionError("SANDBOX_ZIP_BOMB_REJECTED", "Upload size limit exceeded")
    if upload_path.suffix.lower() != ".zip":
        return [upload_path.resolve()]
    root = dest_dir.resolve()
    try:
        with zipfile.ZipFile(upload_path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ENTRIES:
                raise IngestionError("SANDBOX_ZIP_BOMB_REJECTED", "Entry limit exceeded")
            planned = []
            names = set()
            for info in infos:
                name = info.filename.replace("\\", "/")
                target = (root / name).resolve()
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise IngestionError("SANDBOX_ZIP_SYMLINK_REJECTED", name)
                if (name.startswith("/") or re.match(r"^[A-Za-z]:", name)
                        or ".." in name.split("/") or not target.is_relative_to(root)):
                    raise IngestionError("SANDBOX_ZIP_SLIP_REJECTED", name)
                if target in names or target.exists():
                    raise IngestionError("SANDBOX_ZIP_SLIP_REJECTED", "Duplicate or existing member")
                names.add(target)
                if info.is_dir():
                    continue
                if target.suffix.lower() not in ALLOWLIST or target.suffix.lower() == ".zip":
                    raise IngestionError("UPLOAD_DISALLOWED_TYPE", name)
                planned.append((info, target))
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            # Stage bounded streams so rejected archives leave no partial documents.
            with tempfile.TemporaryDirectory(dir=dest_dir.parent) as staging:
                staged = []
                total = 0
                for index, (info, target) in enumerate(planned):
                    temporary = Path(staging) / str(index)
                    size = 0
                    with archive.open(info) as source, temporary.open("wb") as output:
                        while chunk := source.read(65536):
                            size += len(chunk)
                            total += len(chunk)
                            if total > MAX_BYTES or size > MAX_RATIO * max(1, info.compress_size):
                                raise IngestionError("SANDBOX_ZIP_BOMB_REJECTED", info.filename)
                            output.write(chunk)
                    staged.append((temporary, target))
                for temporary, target in staged:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temporary), target)
            return [target for _, target in planned]
    except (zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        raise IngestionError("SANDBOX_ZIP_INVALID", str(exc)) from exc
