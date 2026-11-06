"""Rust-aware AST merge helpers: Cargo.toml and impl-block aware merging."""
from __future__ import annotations

import re
from typing import Optional


class RustSynMerger:
    """Merges concurrent Rust deltas with Cargo.toml and impl-block awareness."""

    # -- Cargo.toml ----------------------------------------------------------------

    _SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    _KV_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*=\s*(.+?)\s*$")

    def merge_cargo_toml(self, base: str, ours: str, theirs: str) -> tuple[str, list[str]]:
        """Three-way merge of Cargo.toml: section-aware key union."""
        conflicts: list[str] = []
        base_sections = self._parse_toml(base)
        ours_sections = self._parse_toml(ours)
        theirs_sections = self._parse_toml(theirs)

        all_sections = sorted(set(base_sections) | set(ours_sections) | set(theirs_sections))
        out: list[str] = []
        for section in all_sections:
            b = base_sections.get(section, {})
            o = ours_sections.get(section, {})
            t = theirs_sections.get(section, {})
            merged: dict[str, str] = {}
            for key in sorted(set(b) | set(o) | set(t)):
                bv, ov, tv = b.get(key), o.get(key), t.get(key)
                if ov == tv:
                    merged[key] = ov if ov is not None else (bv or "")
                elif bv == ov:
                    merged[key] = tv or ""
                elif bv == tv:
                    merged[key] = ov or ""
                else:
                    conflicts.append(f"{section}.{key}: ours={ov!r} theirs={tv!r}")
                    merged[key] = ov or tv or ""
            if merged:
                out.append(f"[{section}]")
                for key, value in merged.items():
                    out.append(f"{key} = {value}")
                out.append("")
        return "\n".join(out).rstrip() + "\n", conflicts

    def _parse_toml(self, source: str) -> dict[str, dict[str, str]]:
        sections: dict[str, dict[str, str]] = {}
        current = ""
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = self._SECTION_RE.match(line)
            if m:
                current = m.group(1)
                sections.setdefault(current, {})
                continue
            kv = self._KV_RE.match(line)
            if kv and current:
                sections[current][kv.group(1)] = kv.group(2)
        return sections

    # -- impl blocks -----------------------------------------------------------------

    _IMPL_RE = re.compile(r"^impl(?:<[^>]*>)?\s+([A-Za-z0-9_:]+)", re.MULTILINE)

    def merge_rust_impl_blocks(self, base: str, ours: str, theirs: str) -> tuple[str, list[str]]:
        """Merges impl blocks by (type, method-name) keys; ours wins on conflict."""
        conflicts: list[str] = []
        base_blocks = self._extract_impl_blocks(base)
        ours_blocks = self._extract_impl_blocks(ours)
        theirs_blocks = self._extract_impl_blocks(theirs)
        keys = sorted(set(base_blocks) | set(ours_blocks) | set(theirs_blocks))
        merged_blocks: list[str] = []
        for key in keys:
            b = base_blocks.get(key, "")
            o = ours_blocks.get(key, "")
            t = theirs_blocks.get(key, "")
            if o and t and o != t and b != o and b != t:
                conflicts.append(f"impl block {key}: concurrent modification")
                merged_blocks.append(o)
            elif o and o != b:
                merged_blocks.append(o)
            elif t and t != b:
                merged_blocks.append(t)
            elif b:
                merged_blocks.append(b)
        # Preserve non-impl content from ours.
        prefix = self._strip_impl_blocks(ours)
        body = "\n\n".join(b.rstrip() for b in merged_blocks if b)
        return (prefix.rstrip() + "\n\n" + body + "\n") if body else (prefix if prefix.endswith("\n") else prefix + "\n"), conflicts

    def _extract_impl_blocks(self, source: str) -> dict[str, str]:
        blocks: dict[str, str] = {}
        lines = source.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m = re.match(r"^(impl(?:<[^>]*>)?\s+)([A-Za-z0-9_:]+)", line.strip())
            if m and line.startswith("impl"):
                type_name = m.group(2)
                # Capture until closing brace at column 0.
                start = i
                depth = 0
                j = i
                while j < len(lines):
                    depth += lines[j].count("{") - lines[j].count("}")
                    j += 1
                    if depth <= 0 and j > start:
                        break
                block = "\n".join(lines[start:j])
                # Key by type + method names inside.
                methods = re.findall(r"fn\s+([A-Za-z0-9_]+)", block)
                key = f"{type_name}::{','.join(sorted(methods)) or '<impl>'}"
                blocks[key] = block
                i = j
            else:
                i += 1
        return blocks

    def _strip_impl_blocks(self, source: str) -> str:
        lines = source.splitlines()
        out: list[str] = []
        i = 0
        while i < len(lines):
            if re.match(r"^impl(?:<[^>]*>)?\s+", lines[i].strip()) and lines[i].startswith("impl"):
                depth = 0
                j = i
                while j < len(lines):
                    depth += lines[j].count("{") - lines[j].count("}")
                    j += 1
                    if depth <= 0 and j > i:
                        break
                i = j
            else:
                out.append(lines[i])
                i += 1
        return "\n".join(out)

    # -- use statements ----------------------------------------------------------------

    def deduplicate_use_statements(self, source: str) -> str:
        seen: set[str] = set()
        out: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("use "):
                if stripped in seen:
                    continue
                seen.add(stripped)
            out.append(line)
        return "\n".join(out) + ("\n" if source.endswith("\n") else "")
