"""GoI task scheduler: native-first with pure-Python fallback."""
from __future__ import annotations

from typing import Optional

from invar_agent.contracts import TaskWavefrontNode

try:
    from invar_agent._native import PyGoiSolver as _NativeGoi
except Exception:  # pragma: no cover
    _NativeGoi = None

from invar_agent._fallback.goi_solver import PurePythonGoiSolver


class GoiTaskScheduler:
    """Wraps PyGoiSolver (native) or PurePythonGoiSolver (fallback)."""

    def __init__(self) -> None:
        self.native_available = _NativeGoi is not None
        self._native: Optional[object] = _NativeGoi() if _NativeGoi else None
        self._fallback = PurePythonGoiSolver()

    @property
    def native(self):
        return self._native

    def build_from_nodes(self, nodes: list[TaskWavefrontNode]) -> None:
        for node in nodes:
            if self._native is not None:
                self._native.add_task_node(
                    node.node_id,
                    list(node.read_set_paths),
                    list(node.write_set_paths),
                    [p.port_name for p in node.input_ports],
                    [p.port_name for p in node.output_ports],
                )
            else:
                deps = [p.source_node_id for p in node.input_ports if p.source_node_id]
                self._fallback.add_node(node.node_id, deps)

    def add_dependency_edge(self, src_node: str, src_port: str, dst_node: str, dst_port: str) -> None:
        if self._native is not None:
            self._native.add_dependency_edge(src_node, src_port, dst_node, dst_port)
        else:
            self._fallback.add_edge(src_node, dst_node)

    def check_nilpotency(self) -> bool:
        if self._native is not None:
            return bool(self._native.check_nilpotency())
        return self._fallback.matrix_nilpotency_check()

    def solve_wavefronts(self) -> list[list[str]]:
        if self._native is not None:
            return [list(w) for w in self._native.solve_wavefronts()]
        return self._fallback.partition_wavefronts()
