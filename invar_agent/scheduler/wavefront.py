"""Wavefront executor: dispatches generations with resource lock arbitration."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class WavefrontGeneration:
    generation: int
    node_ids: list[str]


class ResourceLockTable:
    """Tracks write-set locks to enforce Bernstein disjointness at runtime."""

    def __init__(self) -> None:
        self._locks: dict[str, str] = {}
        self._guard = threading.Lock()

    def acquire(self, node_id: str, write_set: list[str]) -> bool:
        with self._guard:
            for path in write_set:
                holder = self._locks.get(path)
                if holder is not None and holder != node_id:
                    return False
            for path in write_set:
                self._locks[path] = node_id
            return True

    def release(self, node_id: str, write_set: list[str]) -> None:
        with self._guard:
            for path in write_set:
                if self._locks.get(path) == node_id:
                    del self._locks[path]

    def holder(self, path: str) -> Optional[str]:
        return self._locks.get(path)

    def snapshot(self) -> dict[str, str]:
        with self._guard:
            return dict(self._locks)


@dataclass
class NodeResult:
    node_id: str
    success: bool
    output: object = None
    error: Optional[str] = None
    latency_ms: float = 0.0


class WavefrontExecutor:
    """Executes wavefront generations sequentially, nodes in parallel."""

    def __init__(self, max_workers: int = 4) -> None:
        self.locks = ResourceLockTable()
        self.max_workers = max_workers

    def dispatch_wavefront_async(
        self,
        generation: WavefrontGeneration,
        handlers: dict[str, Callable[[], object]],
        write_sets: Optional[dict[str, list[str]]] = None,
    ) -> list[NodeResult]:
        write_sets = write_sets or {}
        results: dict[str, NodeResult] = {}

        def run_one(node_id: str) -> NodeResult:
            start = time.perf_counter()
            ws = write_sets.get(node_id, [])
            acquired = self.locks.acquire(node_id, ws)
            if not acquired:
                return NodeResult(node_id=node_id, success=False,
                                  error=f"resource lock contention on {ws}")
            try:
                handler = handlers[node_id]
                output = handler()
                return NodeResult(node_id=node_id, success=True, output=output,
                                  latency_ms=(time.perf_counter() - start) * 1000.0)
            except Exception as e:  # noqa: BLE001
                return NodeResult(node_id=node_id, success=False, error=str(e),
                                  latency_ms=(time.perf_counter() - start) * 1000.0)
            finally:
                self.locks.release(node_id, ws)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for res in pool.map(run_one, generation.node_ids):
                results[res.node_id] = res
        return [results[n] for n in generation.node_ids]

    def execute_generations(
        self,
        generations: list[list[str]],
        handlers: dict[str, Callable[[], object]],
        write_sets: Optional[dict[str, list[str]]] = None,
    ) -> list[list[NodeResult]]:
        out: list[list[NodeResult]] = []
        for i, gen in enumerate(generations):
            out.append(self.dispatch_wavefront_async(WavefrontGeneration(i, gen), handlers, write_sets))
        return out
