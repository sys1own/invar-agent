"""Pure-Python fallback GoI solver (used when the native extension is absent)."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable


class PurePythonGoiSolver:
    """Matrix-based GoI nilpotency check and wavefront partitioner."""

    def __init__(self) -> None:
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._nodes: list[str] = []

    def add_node(self, node_id: str, dependencies: Iterable[str] = ()) -> None:
        if node_id not in self._nodes:
            self._nodes.append(node_id)
        for dep in dependencies:
            self._adjacency[node_id].add(dep)
            self._adjacency.setdefault(dep, set())

    def add_edge(self, src: str, dst: str) -> None:
        self._adjacency.setdefault(src, set()).add(dst)
        self._adjacency.setdefault(dst, set())
        for n in (src, dst):
            if n not in self._nodes:
                self._nodes.append(n)

    def matrix_nilpotency_check(self, max_k: int = 10_000) -> bool:
        """Nilpotency via reachability: any cycle reachable from a node means
        the interaction operator is not nilpotent."""
        # Iterative DFS cycle detection over the dependency graph.
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._nodes}

        def visit(start: str) -> bool:
            stack = [(start, iter(sorted(self._adjacency.get(start, ()))))]
            color[start] = GRAY
            path = {start}
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if color.get(nxt, WHITE) == GRAY:
                        return False  # back edge -> cycle
                    if color.get(nxt, WHITE) == WHITE:
                        color[nxt] = GRAY
                        path.add(nxt)
                        stack.append((nxt, iter(sorted(self._adjacency.get(nxt, ())))))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    path.discard(node)
                    stack.pop()
            return True

        for node in self._nodes:
            if color[node] == WHITE:
                if not visit(node):
                    return False
        return True

    def partition_wavefronts(self) -> list[list[str]]:
        """Kahn topological levels as wavefront generations."""
        indegree = {n: 0 for n in self._nodes}
        dependents: dict[str, list[str]] = defaultdict(list)
        for node, deps in self._adjacency.items():
            for dep in deps:
                if dep in indegree and node in indegree:
                    indegree[node] += 1
                    dependents[dep].append(node)
        queue = deque(sorted(n for n, d in indegree.items() if d == 0))
        levels: list[list[str]] = []
        level_of: dict[str, int] = {}
        while queue:
            level: list[str] = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node)
                for dep in dependents.get(node, []):
                    indegree[dep] -= 1
                    if indegree[dep] == 0:
                        queue.append(dep)
            levels.append(sorted(level))
            for n in level:
                level_of[n] = len(levels) - 1
        if len(level_of) != len(self._nodes):
            raise ValueError("GoI task network contains a cyclic dependency; cannot schedule wavefronts")
        return levels
