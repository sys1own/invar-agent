"""GoI planning and HIS-gated rebuilds; planning never mutates artifacts or caches."""
from __future__ import annotations

import json
from pathlib import Path

from .his_cache import HISCacheError, HISCacheState, artifacts_match, digest, node_fingerprints
from .uast_models import (
    BuilderUAST, SmwDeltaResult, WavefrontExecutionPlan, WavefrontScheduleResult,
    builder_backend, canonical_json,
)


class GoICriticalError(ValueError):
    code = "GOI_IRREDUCIBLE_CYCLE"
    tier = "CRITICAL_NON_RETRYABLE"
    action = "ABORT_BUILD_CRITICAL"
    llm_retries = 0

    def __init__(self, details):
        self.details = details
        self.cycle_witness = details.get("cycle_witness", [])
        self.spectral_radius = details.get("spectral_radius", 1.0)
        self.capacitance_det = details.get("capacitance_det", 0.0)
        super().__init__(canonical_json(details))


def _call(function, *args):
    try:
        return function(*args)
    except ValueError as exc:
        try:
            details = json.loads(str(exc))
        except ValueError:
            raise exc
        if isinstance(details, dict) and details.get("error_code") == "GOI_IRREDUCIBLE_CYCLE":
            raise GoICriticalError(details) from exc
        raise


def _pairs(edges):
    return {(edge.from_node, edge.to_node) for edge in edges}


def _waves(order, pairs):
    indegree = dict.fromkeys(order, 0)
    outgoing = {name: [] for name in order}
    for source, target in pairs:
        indegree[target] += 1
        outgoing[source].append(target)
    ready = sorted(name for name in order if not indegree[name])
    waves = []
    while ready:
        waves.append(ready)
        following = []
        for source in ready:
            for target in outgoing[source]:
                indegree[target] -= 1
                if not indegree[target]:
                    following.append(target)
        ready = sorted(following)
    if sum(map(len, waves)) != len(order):
        raise ValueError("GoI backend accepted a cyclic graph")
    return waves


def _check_resolvent(state):
    order, flat = state.node_order, state.resolvent_flat
    n = len(order)
    index = {name: i for i, name in enumerate(order)}
    incoming = {name: [] for name in order}
    for source, target in _pairs(state.edges):
        incoming[target].append(index[source])
    for row, name in enumerate(order):
        for col in range(n):
            value = flat[row * n + col]
            total = sum(flat[source * n + col] for source in incoming[name])
            if abs(value - total - int(row == col)) > 1e-9 * max(1, abs(value), abs(total)):
                raise HISCacheError("HIS_CACHE_INVALID", "Cached resolvent does not satisfy (I-M)R=I")


class WavefrontScheduler:
    def __init__(self, artifact_root: Path | None = None):
        self.artifact_root = Path(artifact_root).resolve() if artifact_root is not None else None

    def plan(self, uast: BuilderUAST, previous_cache: HISCacheState | None = None) -> WavefrontExecutionPlan:
        uast = BuilderUAST.model_validate(uast.model_dump(mode="json"))
        backend = builder_backend()
        order = sorted(node.node_id for node in uast.nodes)
        current_pairs = _pairs(uast.edges)
        fingerprints = node_fingerprints(uast)
        graph_digest = digest("invar.builder.uast.v1", uast.model_dump(mode="json"))
        if previous_cache is not None:
            previous_cache = HISCacheState.model_validate(previous_cache.model_dump(mode="json"))
            if previous_cache.project_id != uast.project_id:
                raise HISCacheError("HIS_CACHE_PROJECT_MISMATCH", uast.project_id)
            if self.artifact_root is not None and self.artifact_root != Path(previous_cache.artifact_root).resolve():
                raise HISCacheError("HIS_CACHE_INVALID", "Artifact root mismatch")
            _check_resolvent(previous_cache)
        entries = previous_cache.entries if previous_cache is not None else {}
        changed_his = {name for name, (his, _, declared) in fingerprints.items() if name not in entries
                       or entries[name].his_digest != his or entries[name].declared_his_digest != declared}
        if previous_cache is not None:
            current_edges = {(e.from_node, e.to_node, e.edge_kind, e.port_id) for e in uast.edges}
            old_edges = {(e.from_node, e.to_node, e.edge_kind, e.port_id) for e in previous_cache.edges}
            changed_his.update(target for _, target, _, _ in current_edges.symmetric_difference(old_edges) if target in order)
        compatible = previous_cache is not None and order == previous_cache.node_order
        determinant = 1.0
        roots = set(changed_his)
        if compatible:
            old_pairs = _pairs(previous_cache.edges)
            deltas = [dict(from_node=a, to_node=b, weight=weight)
                      for pairs, weight in ((current_pairs - old_pairs, 1.0), (old_pairs - current_pairs, -1.0))
                      for a, b in sorted(pairs)]
            updated = SmwDeltaResult.model_validate_json(_call(backend.compute_builder_smw_delta,
                canonical_json(previous_cache.resolvent_flat), canonical_json(order),
                canonical_json(deltas), canonical_json(sorted(changed_his))))
            if not updated.is_valid_dag or abs(updated.capacitance_det) < 1e-9:
                raise GoICriticalError(dict(error_code=GoICriticalError.code, llm_retries=0,
                    capacitance_det=updated.capacitance_det, cycle_witness=[]))
            flat, determinant = updated.updated_resolvent_flat, updated.capacitance_det
            waves, rho = _waves(order, current_pairs), 0.0
            rebuild = set(updated.affected_node_ids)
        else:
            scheduled = WavefrontScheduleResult.model_validate_json(_call(backend.schedule_builder_wavefronts, uast.model_dump_json()))
            if not scheduled.is_nilpotent or scheduled.spectral_radius >= 1 - 1e-9:
                raise GoICriticalError(dict(error_code=GoICriticalError.code, llm_retries=0,
                    spectral_radius=scheduled.spectral_radius, cycle_witness=scheduled.cycle_witness))
            flat, waves, rho = scheduled.resolvent_flat, scheduled.wavefronts, scheduled.spectral_radius
            if previous_cache is not None:
                changed_pairs = current_pairs.symmetric_difference(_pairs(previous_cache.edges))
                roots.update(target for _, target in changed_pairs if target in order)
            index = {name: i for i, name in enumerate(order)}
            rebuild = {name for row, name in enumerate(order)
                       if any(abs(flat[row * len(order) + index[root]]) > 1e-9 for root in roots)}
        if previous_cache is None:
            rebuild.update(order)
        else:
            root = Path(previous_cache.artifact_root)
            for node in uast.nodes:
                entry = entries.get(node.node_id)
                if entry is None or entry.body_digest != fingerprints[node.node_id][1] or not artifacts_match(entry, node, root):
                    rebuild.add(node.node_id)
            if previous_cache.uast_digest != graph_digest or rebuild:
                if "mod_receipt" in order:
                    rebuild.add("mod_receipt")
        cached = set(order) - rebuild
        return WavefrontExecutionPlan(project_id=uast.project_id, uast_digest=graph_digest, is_nilpotent=True,
            spectral_radius=rho, wavefronts=waves, node_order=order, resolvent_flat=flat,
            rebuild_nodes=[name for name in order if name in rebuild], cached_nodes=[name for name in order if name in cached],
            smw_capacitance_det=determinant, used_smw=compatible,
            node_status={name: "CACHE_HIT_SKIPPED" if name in cached else "PENDING" for name in order})
