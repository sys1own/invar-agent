"""Causal action DAG and topological wavefront utilities."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DAGNode:
    node_id: str
    weight: float = 1.0
    resources: set[str] = field(default_factory=set)
    deps: set[str] = field(default_factory=set)


class CausalActionDAG:
    """Dependency DAG over wavefront nodes with topological iteration."""

    def __init__(self) -> None:
        self.nodes: dict[str, DAGNode] = {}
        self.edges: dict[str, set[str]] = {}  # src -> dsts

    def add_node(self, node: DAGNode) -> None:
        self.nodes[node.node_id] = node
        self.edges.setdefault(node.node_id, set())
        for dep in node.deps:
            self.edges.setdefault(dep, set()).add(node.node_id)

    def predecessors(self, node_id: str) -> set[str]:
        return set(self.nodes.get(node_id, DAGNode(node_id)).deps)

    def successors(self, node_id: str) -> set[str]:
        return set(self.edges.get(node_id, set()))

    def topological_order(self) -> list[str]:
        indegree = {n: len(self.nodes[n].deps) for n in self.nodes}
        heap = [n for n, d in indegree.items() if d == 0]
        heapq.heapify(heap)
        order: list[str] = []
        while heap:
            node = heapq.heappop(heap)
            order.append(node)
            for succ in sorted(self.successors(node)):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    heapq.heappush(heap, succ)
        if len(order) != len(self.nodes):
            raise ValueError("DAG contains a cycle")
        return order


class TopologicalWavefrontIterator:
    """Yields wavefront generations (Kahn levels) over a CausalActionDAG."""

    def __init__(self, dag: CausalActionDAG) -> None:
        self.dag = dag

    def __iter__(self):
        return iter(self.generations())

    def generations(self) -> list[list[str]]:
        indegree = {n: len(self.dag.nodes[n].deps) for n in self.dag.nodes}
        current = sorted(n for n, d in indegree.items() if d == 0)
        gens: list[list[str]] = []
        while current:
            gens.append(current)
            nxt: list[str] = []
            for node in current:
                for succ in self.dag.successors(node):
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        nxt.append(succ)
            current = sorted(nxt)
        if sum(len(g) for g in gens) != len(self.dag.nodes):
            raise ValueError("DAG contains a cycle")
        return gens


def compute_critical_path(dag: CausalActionDAG) -> tuple[float, list[str]]:
    """Longest weighted path through the DAG (critical path)."""
    order = dag.topological_order()
    dist = {n: dag.nodes[n].weight for n in order}
    prev: dict[str, Optional[str]] = {n: None for n in order}
    for node in order:
        for succ in dag.successors(node):
            cand = dist[node] + dag.nodes[succ].weight
            if cand > dist[succ]:
                dist[succ] = cand
                prev[succ] = node
    if not order:
        return 0.0, []
    end = max(order, key=lambda n: dist[n])
    path: list[str] = []
    cur: Optional[str] = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[end], path


def detect_resource_contention(dag: CausalActionDAG) -> list[tuple[str, str, str]]:
    """Pairs of nodes sharing a resource with no dependency ordering."""
    by_resource: dict[str, list[str]] = {}
    for node in dag.nodes.values():
        for res in node.resources:
            by_resource.setdefault(res, []).append(node.node_id)
    conflicts: list[tuple[str, str, str]] = []
    for res, holders in sorted(by_resource.items()):
        for i in range(len(holders)):
            for j in range(i + 1, len(holders)):
                a, b = holders[i], holders[j]
                if b not in dag.successors(a) and a not in dag.successors(b):
                    conflicts.append((res, a, b))
    return conflicts
