"""Closed registry of verified deterministic leaf emitters."""
from __future__ import annotations

from .emitters.fallback_slots import emit_slot
from .his_cache import interface_digest, port_signature
from .uast_models import FALLBACK_EMITTERS


class FallbackEmitterRegistry:
    def __init__(self):
        self.emitters = {identifier: emit_slot for identifier in FALLBACK_EMITTERS.values()}

    def emit(self, emitter_id, port_contract, ir):
        if emitter_id not in self.emitters:
            raise ValueError(f"Unregistered deterministic fallback: {emitter_id}")
        if interface_digest(port_signature(port_contract)) != port_contract.his_digest:
            raise ValueError("Frozen interface digest does not match contract")
        return self.emitters[emitter_id](emitter_id, port_contract, ir)
