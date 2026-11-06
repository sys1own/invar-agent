"""Causal fault interaction graph: HIN-backed root-cause tracing."""
from __future__ import annotations

from typing import Optional

try:
    from invar_agent._native import PyHinEngine as _NativeHin
except Exception:  # pragma: no cover
    _NativeHin = None


class _PurePythonHinFallback:
    """Minimal deterministic fallback mirroring the native HIN semantics."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._agents: dict[int, dict] = {}
        self._next_id = 1
        self._wires: dict[tuple[int, str], tuple[int, str]] = {}

    def spawn_agent(self, symbol_name: str, principal_port: str, aux_ports: list[str]) -> int:
        aid = self._next_id
        self._next_id += 1
        self._agents[aid] = {"symbol": symbol_name, "principal": principal_port, "aux": list(aux_ports), "alive": True}
        return aid

    def connect_ports(self, agent_a: int, a_is_principal: bool, agent_b: int, b_is_principal: bool) -> None:
        pa = "principal" if a_is_principal else "aux"
        pb = "principal" if b_is_principal else "aux"
        self._wires[(agent_a, pa)] = (agent_b, pb)
        self._wires[(agent_b, pb)] = (agent_a, pa)

    def reduce_to_normal_form(self) -> int:
        return 0

    def trace_causal_fault_slice(self, fault_agent_id: int, max_hops: int) -> list[str]:
        return self.trace_fault_propagation(fault_agent_id, max_hops)

    def trace_fault_propagation(self, fault_agent_id: int, max_hops: int = 3) -> list[str]:
        seen = {fault_agent_id}
        frontier = [fault_agent_id]
        ordered: list[str] = []
        hops = 0
        while frontier and hops < max_hops:
            nxt: list[int] = []
            for aid in frontier:
                for (src, _port), (dst, _dport) in list(self._wires.items()):
                    if src == aid and dst not in seen:
                        seen.add(dst)
                        nxt.append(dst)
                        ordered.append(self._agents.get(dst, {}).get("symbol", str(dst)))
            frontier = nxt
            hops += 1
        return ordered

    def active_agent_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.get("alive", True))


class CausalFaultInteractionGraph:
    """Wraps PyHinEngine (or fallback) for fault-slice tracing."""

    def __init__(self, capacity: int = 1024) -> None:
        self.native_available = _NativeHin is not None
        self._engine = _NativeHin(capacity) if _NativeHin else _PurePythonHinFallback(capacity)
        self._symbol_to_agent: dict[str, int] = {}

    @property
    def engine(self):
        return self._engine

    def register_agent(self, symbol_name: str, principal_port: str = "in", aux_ports: Optional[list[str]] = None) -> int:
        aid = self._engine.spawn_agent(symbol_name, principal_port, aux_ports or [])
        self._symbol_to_agent[symbol_name] = aid
        return aid

    def register_wire(self, symbol_a: str, a_principal: bool, symbol_b: str, b_principal: bool) -> None:
        a = self._symbol_to_agent.get(symbol_a)
        b = self._symbol_to_agent.get(symbol_b)
        if a is None or b is None:
            return
        self._engine.connect_ports(a, a_principal, b, b_principal)

    def reduce(self) -> int:
        return int(self._engine.reduce_to_normal_form())

    def trace_fault_propagation(self, fault_symbol: str, max_hops: int = 3) -> list[str]:
        aid = self._symbol_to_agent.get(fault_symbol)
        if aid is None:
            return []
        return list(self._engine.trace_causal_fault_slice(aid, max_hops))

    def isolate_root_cause_port(self, fault_symbol: str, max_hops: int = 3) -> str:
        """Returns the earliest causal predecessor symbol (root cause candidate)."""
        slice_symbols = self.trace_fault_propagation(fault_symbol, max_hops)
        return slice_symbols[-1] if slice_symbols else fault_symbol

    def active_agent_count(self) -> int:
        return int(self._engine.active_agent_count())
