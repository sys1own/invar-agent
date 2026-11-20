"""Domain-separated interface/body identities and verified artifact checkpoints."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Literal

import blake3
from pydantic import Field, model_validator

from .ingestion.schemas import StrictModel
from .uast_models import (
    BuilderUAST, DependencyEdgeModel, DeterministicModuleNodeModel, InterfacePortNodeModel,
    KernelNodeModel, LLMExtensionSlotNodeModel, WavefrontExecutionPlan, builder_backend, canonical_json,
)


def digest(context: str, payload) -> str:
    return builder_backend().compute_his_digest_v2(context, canonical_json(payload))


def port_signature(port: InterfacePortNodeModel) -> dict:
    return port.model_dump(mode="json", include={"inputs", "outputs", "z3_preconditions", "z3_postconditions"})


def interface_digest(signature) -> str:
    return digest("invar.his.interface.v2", signature)


def target_relpaths(node) -> list[str]:
    if isinstance(node, DeterministicModuleNodeModel):
        return node.target_relpaths
    if isinstance(node, LLMExtensionSlotNodeModel):
        return [node.target_relpath]
    return []


def node_fingerprints(uast: BuilderUAST) -> dict[str, tuple[str, str, str]]:
    ports = {node.node_id: port_signature(node) for node in uast.nodes if isinstance(node, InterfacePortNodeModel)}
    result = {}
    for node in uast.nodes:
        if isinstance(node, InterfacePortNodeModel):
            public, body, declared = ports[node.node_id], {}, node.his_digest
        elif isinstance(node, KernelNodeModel):
            public, declared = ports[node.output_port_id], node.his_digest
            body = node.model_dump(mode="json", exclude={"his_digest", "output_port_id", "node_id", "kind"})
        elif isinstance(node, LLMExtensionSlotNodeModel):
            public, declared = ports[node.frozen_port_id], node.frozen_his_digest
            body = node.model_dump(mode="json", exclude={"frozen_his_digest", "frozen_port_id", "node_id", "kind"})
        else:
            identifiers = sorted(set(node.required_port_ids + node.provided_port_ids))
            public = ports[identifiers[0]] if len(identifiers) == 1 else {"interfaces": [ports[name] for name in identifiers]}
            declared = node.his_digest
            body = node.model_dump(mode="json", exclude={"his_digest", "node_id", "kind", "required_port_ids", "provided_port_ids"})
        result[node.node_id] = (interface_digest(public), digest("invar.his.body.v2", body), declared)
    return result


class HISCacheError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


class HISCacheEntry(StrictModel):
    module_id: str
    his_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    body_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    declared_his_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_digests: dict[str, str] = Field(default_factory=dict)
    verified_status: Literal["VERIFIED", "FALLBACK_VERIFIED", "UNVERIFIED"] = "UNVERIFIED"


class HISCacheState(StrictModel):
    schema_version: Literal["builder_his_cache.v1"] = "builder_his_cache.v1"
    build_id: str
    project_id: str
    artifact_root: str
    uast_digest: str
    node_order: list[str]
    resolvent_flat: list[float]
    edges: list[DependencyEdgeModel]
    entries: dict[str, HISCacheEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_checkpoint(self):
        if len(set(self.node_order)) != len(self.node_order) or len(self.resolvent_flat) != len(self.node_order) ** 2:
            raise ValueError("Invalid cache resolvent shape or node order")
        if any(key != value.module_id or key not in self.node_order for key, value in self.entries.items()):
            raise ValueError("Cache entries do not match graph nodes")
        if any(edge.from_node not in self.node_order or edge.to_node not in self.node_order for edge in self.edges):
            raise ValueError("Cache edge references an unknown node")
        return self


def artifact_path(root: Path, relpath: str) -> Path:
    path = Path(relpath)
    if (not relpath or path.is_absolute() or any(c in relpath for c in "\\:\0")
            or any(part in {"", ".", ".."} for part in relpath.split("/"))):
        raise HISCacheError("HIS_CACHE_UNSAFE_PATH", relpath)
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()) or any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise HISCacheError("HIS_CACHE_UNSAFE_PATH", relpath)
    return target


def artifacts_match(entry: HISCacheEntry, node, artifact_root: Path) -> bool:
    if entry.verified_status not in {"VERIFIED", "FALLBACK_VERIFIED"} or not set(target_relpaths(node)) <= entry.artifact_digests.keys():
        return False
    try:
        for relpath, expected in entry.artifact_digests.items():
            target = artifact_path(artifact_root, relpath)
            if not target.is_file() or blake3.blake3(target.read_bytes()).hexdigest() != expected:
                return False
    except (OSError, HISCacheError):
        return False
    return True


class HISCache:
    def __init__(self, workspace_root: Path, build_id: str, artifact_root: Path | None = None):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", build_id):
            raise HISCacheError("HIS_CACHE_INVALID_BUILD_ID", build_id)
        self.workspace_root = Path(workspace_root).resolve()
        self.build_id = build_id
        self.artifact_root = Path(artifact_root).resolve() if artifact_root is not None else self.workspace_root
        self.path = self.workspace_root / ".invar" / "builds" / build_id / "his_cache.json"
        if not self.artifact_root.is_relative_to(self.workspace_root) or not self.path.resolve().is_relative_to(self.workspace_root):
            raise HISCacheError("HIS_CACHE_UNSAFE_PATH", str(self.path))

    def load(self) -> HISCacheState | None:
        if self.path.is_symlink() or not self.path.resolve().is_relative_to(self.workspace_root):
            raise HISCacheError("HIS_CACHE_UNSAFE_PATH", str(self.path))
        if not self.path.exists():
            return None
        try:
            state = HISCacheState.model_validate_json(self.path.read_text())
            if state.build_id != self.build_id or Path(state.artifact_root).resolve() != self.artifact_root:
                raise ValueError("Cache belongs to another build or artifact root")
            return state
        except (ValueError, OSError) as exc:
            raise HISCacheError("HIS_CACHE_INVALID", str(exc)) from exc

    def new_state(self, uast: BuilderUAST, plan: WavefrontExecutionPlan) -> HISCacheState:
        if plan.project_id != uast.project_id or plan.uast_digest != digest("invar.builder.uast.v1", uast.model_dump(mode="json")):
            raise HISCacheError("HIS_CACHE_PLAN_MISMATCH", uast.project_id)
        return HISCacheState(build_id=self.build_id, project_id=uast.project_id, artifact_root=str(self.artifact_root),
            uast_digest=plan.uast_digest, node_order=plan.node_order, resolvent_flat=plan.resolvent_flat, edges=uast.edges)

    def record_verified(self, state: HISCacheState, uast: BuilderUAST, module_id: str,
                        artifact_relpaths: list[str] | None = None, *, fallback: bool = False) -> HISCacheEntry:
        if state.project_id != uast.project_id or state.uast_digest != digest("invar.builder.uast.v1", uast.model_dump(mode="json")):
            raise HISCacheError("HIS_CACHE_PLAN_MISMATCH", module_id)
        node = next((node for node in uast.nodes if node.node_id == module_id), None)
        if node is None:
            raise HISCacheError("HIS_CACHE_UNKNOWN_NODE", module_id)
        paths = target_relpaths(node) if artifact_relpaths is None else artifact_relpaths
        if not set(target_relpaths(node)) <= set(paths):
            raise HISCacheError("HIS_CACHE_MISSING_ARTIFACT", module_id)
        artifacts = {}
        for relpath in paths:
            target = artifact_path(self.artifact_root, relpath)
            if not target.is_file():
                raise HISCacheError("HIS_CACHE_MISSING_ARTIFACT", relpath)
            artifacts[relpath] = blake3.blake3(target.read_bytes()).hexdigest()
        interface, body, declared = node_fingerprints(uast)[module_id]
        entry = HISCacheEntry(module_id=module_id, his_digest=interface, body_digest=body, declared_his_digest=declared,
            artifact_digests=artifacts, verified_status="FALLBACK_VERIFIED" if fallback else "VERIFIED")
        state.entries[module_id] = entry
        return entry

    def save(self, state: HISCacheState) -> None:
        state = HISCacheState.model_validate(state.model_dump(mode="json"))
        if state.build_id != self.build_id or Path(state.artifact_root).resolve() != self.artifact_root:
            raise HISCacheError("HIS_CACHE_INVALID", "Cache identity mismatch")
        if self.path.is_symlink() or not self.path.resolve().is_relative_to(self.workspace_root):
            raise HISCacheError("HIS_CACHE_UNSAFE_PATH", str(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, prefix=".his_cache.", delete=False) as output:
                temporary = Path(output.name)
                output.write(canonical_json(state.model_dump(mode="json")))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
