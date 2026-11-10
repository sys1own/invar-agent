"""Compacted dual-context engine for the Co-Pilot Planner.

Provides two strictly partitioned context blocks:

* ``<software_codebase_context>`` — real software project files ONLY. When no
  software project has been built yet the block explicitly states
  ``SOFTWARE_CODEBASE_STATE: GREENFIELD`` so the Builder Planner never mistakes
  research artifacts for an existing codebase.
* ``<research_vault_context>`` — a compacted digest of every completed research
  run in the Research Vault, plus the full verified unified kernel code for any
  explicitly referenced ``@research:<id>`` item.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .research.vault import ResearchVaultManager

# Directories that never belong to the software codebase context.
_SOFTWARE_SKIP_DIRS = {
    ".invar", ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "research", "checkpoints", "node_modules", "target", "dist", "build",
}
_SOFTWARE_SKIP_SUFFIXES = (
    ".so", ".dll", ".dylib", ".o", ".a", ".pyc", ".pyd", ".whl",
)
_SOFTWARE_SKIP_NAMES = {"research_report.md", ".env", ".gitignore"}
_MANIFEST_NAMES = {"pyproject.toml", "Cargo.toml", "setup.py", "setup.cfg",
                   "package.json", "requirements.txt"}
_PLACEHOLDER_NAMES = {"hello.py", "main.py", "app.py"}

GREENFIELD_BLOCK = """<software_codebase_context>
STATUS: GREENFIELD_WORKSPACE (No existing software project built yet).
IMPORTANT: Do NOT assume any research kernels or checkpoints are part of the software codebase. Plan new software builds as clean, standalone architectures at the workspace root.
</software_codebase_context>"""


def _is_software_file(path: Path) -> bool:
    name = path.name
    if name in _SOFTWARE_SKIP_NAMES or name.endswith(_SOFTWARE_SKIP_SUFFIXES):
        return False
    if name.startswith("kernel_") and name.endswith(".cpp"):
        return False
    return True


def _iter_software_files(workspace: Path, limit: int = 400):
    count = 0
    for path in sorted(workspace.rglob("*")):
        if count >= limit:
            break
        rel = path.relative_to(workspace)
        parts = rel.parts
        if any(part in _SOFTWARE_SKIP_DIRS or part.startswith(".invar") for part in parts[:-1]):
            continue
        if any(part.startswith(".") and part != "." for part in parts):
            continue
        if path.is_dir() or not _is_software_file(path):
            continue
        count += 1
        yield path


def _py_signatures(path: Path, max_chars: int = 900) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return ""
    lines: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = ", ".join(a.arg for a in node.args.args)
            lines.append(f"def {node.name}({args})")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body
                       if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)][:8]
            lines.append(f"class {node.name}: methods={methods}")
    return "\n".join(lines)[:max_chars]


def build_software_codebase_context(workspace_dir: str | Path,
                                    max_chars: int = 6000) -> str:
    """Compact summary of REAL software files only (strictly excludes the vault)."""
    workspace = Path(workspace_dir)
    files = list(_iter_software_files(workspace))
    if not files:
        return GREENFIELD_BLOCK

    manifests: List[str] = []
    tree_lines: List[str] = []
    signatures: List[str] = []
    for path in files:
        rel = path.relative_to(workspace).as_posix()
        tree_lines.append(rel)
        if path.name in _MANIFEST_NAMES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:600]
            except OSError:
                text = ""
            manifests.append(f"--- {rel} ---\n{text}")
        elif path.suffix == ".py":
            sig = _py_signatures(path)
            if sig:
                signatures.append(f"--- {rel} ---\n{sig}")

    # A workspace with only untouched placeholder files is effectively greenfield.
    meaningful = [p for p in files if p.name not in _PLACEHOLDER_NAMES]
    if not meaningful and len(files) <= 2:
        return GREENFIELD_BLOCK

    body_parts = [
        f"workspace_root: {workspace}",
        f"state: ACTIVE ({len(files)} software files)",
        "file_tree:\n" + "\n".join(tree_lines[:120]),
    ]
    if manifests:
        body_parts.append("manifests:\n" + "\n".join(manifests[:6]))
    if signatures:
        body_parts.append("signatures (imports + defs/classes):\n" + "\n".join(signatures[:12]))
    body = "\n\n".join(body_parts)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n…(truncated)"
    return f"<software_codebase_context>\n{body}\n</software_codebase_context>"


# ---------------------------------------------------------------------------
# Research vault context
# ---------------------------------------------------------------------------

_RESEARCH_REF_RE = re.compile(r"@research:([A-Za-z0-9_.-]+)")


def _referenced_ids(user_query: str, explicit: Optional[List[str]]) -> List[str]:
    ids = list(explicit or [])
    ids.extend(_RESEARCH_REF_RE.findall(user_query or ""))
    # De-duplicate, preserve order.
    seen: List[str] = []
    for item in ids:
        if item and item not in seen:
            seen.append(item)
    return seen


def _keyword_match(run: Dict[str, Any], user_query: str) -> bool:
    """True when the query strongly matches a run's title keywords."""
    query = (user_query or "").lower()
    if not query or len(query) < 12:
        return False
    title = str(run.get("title", "")).lower()
    words = [w for w in re.split(r"[^a-z0-9]+", title) if len(w) >= 5]
    if not words:
        return False
    hits = sum(1 for w in words if w in query)
    return hits >= max(2, len(words) // 2)


def _metric_lines(metrics: Dict[str, Any], limit: int = 8) -> List[str]:
    lines: List[str] = []
    for name, entry in list(metrics.items())[:limit]:
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        unit = entry.get("unit") or ""
        status = "verified" if entry.get("verified") else "unverified"
        lines.append(f"  - {name} = {value} {unit} ({status})".rstrip())
    return lines


def _contracts_digest_lines(contracts: List[Dict[str, Any]],
                            indent: str = "  ") -> List[str]:
    """Render contracts_summary (initial_state / dt / steps / variable_map)."""
    lines: List[str] = []
    for contract in contracts[:4]:
        cid = contract.get("contract_id") or "contract"
        initial = contract.get("initial_state") or []
        lines.append(
            f"{indent}- {cid}: dt={contract.get('dt')} steps={contract.get('steps')}")
        lines.append(f"{indent}  initial_state={initial}")
        variable_map = contract.get("variable_map") or {}
        if variable_map:
            compact = {slot: (entry.get("name") if isinstance(entry, dict) else entry)
                       for slot, entry in variable_map.items()}
            lines.append(f"{indent}  variable_map={compact}")
    return lines


def build_research_vault_context(workspace_dir: str | Path,
                                 referenced_ids: Optional[List[str]] = None,
                                 user_query: str = "",
                                 max_chars: int = 8000) -> str:
    """Compacted digest of all vault runs + full code for referenced runs."""
    vault = ResearchVaultManager(workspace_dir)
    runs = vault.list_runs()
    if not runs:
        return ("<research_vault_context>\nSTATUS: EMPTY (no completed research runs "
                "in the Research Vault yet).\n</research_vault_context>")

    explicit = _referenced_ids(user_query, referenced_ids)
    digest_lines: List[str] = []
    details: List[str] = []
    for run in runs:
        run_id = str(run.get("run_id", ""))
        title = str(run.get("title", run_id))
        verdict = run.get("verdict", "")
        status = run.get("status", "")
        backend = run.get("target_backend", "")
        digest_lines.append(
            f"- @research:{run_id} | {title} | verdict={verdict or 'n/a'} | "
            f"status={status} | backend={backend or 'n/a'}")
        metrics = _metric_lines(run.get("key_metrics") or {})
        if metrics:
            digest_lines.append("  key_metrics:")
            digest_lines.extend(metrics)
        constants = run.get("extracted_constants") or {}
        if constants:
            digest_lines.append("  extracted_constants:")
            digest_lines.extend(
                f"  - {k} = {v}" for k, v in list(constants.items())[:12])
        equations = run.get("governing_equations") or []
        if equations:
            digest_lines.append("  governing_equations:")
            digest_lines.extend(f"  - {e}" for e in equations[:8])
        functions = run.get("verified_functions") or []
        if functions:
            digest_lines.append(f"  verified_functions: {', '.join(functions[:10])}")
        digest_lines.extend(_contracts_digest_lines(run.get("contracts_summary") or []))

        referenced = run_id in explicit or _keyword_match(run, user_query)
        if referenced:
            detail = vault.get_run(run_id)
            if detail is not None:
                code = detail.get("unified_module_code") or ""
                spec_parts = [f"title: {title}", f"verdict: {verdict}",
                              f"status: {status}"]
                metrics_block = "\n".join(_metric_lines(run.get("key_metrics") or {}, limit=16))
                if metrics_block:
                    spec_parts.append("verified_metrics:\n" + metrics_block)
                constants_block = "\n".join(
                    f"- {k} = {v}" for k, v in list(constants.items())[:24])
                if constants_block:
                    spec_parts.append("extracted_constants:\n" + constants_block)
                equations_block = "\n".join(f"- {e}" for e in equations[:16])
                if equations_block:
                    spec_parts.append("governing_equations:\n" + equations_block)
                contracts_block = "\n".join(
                    _contracts_digest_lines(run.get("contracts_summary") or [], indent="  "))
                if contracts_block:
                    spec_parts.append("simulation_contracts (initial_state, "
                                      "dt, steps, variable_map — reproduce these "
                                      "EXACTLY, never bisect):\n" + contracts_block)
                details.append(
                    f'<referenced_research_detail id="{run_id}">\n'
                    + "\n".join(spec_parts)
                    + "\n\n# unified_module.py (verified)\n```python\n"
                    + (code[:6000] or "# (no unified module archived)")
                    + "\n```\n</referenced_research_detail>")

    body = "<research_vault_context>\nCOMPACTED RESEARCH DIGEST:\n" \
        + "\n".join(digest_lines)
    if details:
        body += "\n\nAUTO-EXPANDED REFERENCED RESEARCH:\n" + "\n\n".join(details)
    body += "\n</research_vault_context>"
    if len(body) > max_chars:
        body = body[:max_chars] + "\n…(truncated)\n</research_vault_context>"
    return body


def dual_context(workspace_dir: str | Path,
                 referenced_ids: Optional[List[str]] = None,
                 user_query: str = "") -> str:
    """Both partitioned blocks concatenated for planner prompts."""
    return (build_software_codebase_context(workspace_dir)
            + "\n\n"
            + build_research_vault_context(workspace_dir, referenced_ids, user_query))
