"""Single-authority RFC 8785 JCS sealing and IHCG v1 compatibility bridge.

All cryptographic commitments delegate exclusively to the Rust native core
(`invar_agent/_native/src/his.rs`) to guarantee byte-identical cross-language
RFC 8785 UTF-16 JCS canonicalization and 1-byte domain-separated BLAKE3 hashes:
  - 0x01 || JCS(SciIR)
  - 0x02 || JCS(IHCG_v1)
  - 0x03 || raw_kernel_source
  - 0x04 || JCS(EvidenceRecord)
  - 0x05 || JCS(VerdictReceipt)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from invar_agent.ir.schema import SciIR, SlotRole

DOMAIN_SCIIR = 0x01
DOMAIN_IHCG_V1 = 0x02
DOMAIN_KERNEL_SOURCE = 0x03
DOMAIN_EVIDENCE_RECORD = 0x04
DOMAIN_VERDICT_RECEIPT = 0x05


@dataclass(frozen=True)
class SealedSciIR:
    ir: SciIR
    schema_version: str
    sciir_commitment_blake3: str
    ihcg_v1_commitment_blake3: str
    canonical_sciir_jcs: str
    ihcg_v1_projection: dict[str, Any]


def _get_native():
    from invar_agent import _native  # type: ignore[attr-defined]
    return _native


def jcs_canonicalize(obj_or_json: Any) -> str:
    """Canonicalize any JSON-serializable object or JSON string via Rust RFC 8785 JCS."""
    native = _get_native()
    raw = obj_or_json if isinstance(obj_or_json, str) else json.dumps(obj_or_json, ensure_ascii=False)
    return str(native.py_jcs_canonicalize(raw))


def domain_commit_json(domain_byte: int, obj_or_json: Any) -> str:
    """Compute `BLAKE3(domain_byte || JCS(obj))` in Rust."""
    native = _get_native()
    raw = obj_or_json if isinstance(obj_or_json, str) else json.dumps(obj_or_json, ensure_ascii=False)
    return str(native.py_domain_commit_json(int(domain_byte), raw))


def domain_commit_raw(domain_byte: int, raw_bytes: bytes | str) -> str:
    """Compute `BLAKE3(domain_byte || raw_bytes)` in Rust."""
    native = _get_native()
    payload = raw_bytes.encode("utf-8") if isinstance(raw_bytes, str) else raw_bytes
    return str(native.py_domain_commit_raw(int(domain_byte), payload))


def project_to_ihcg_v1(ir: SciIR, sciir_commitment_blake3: str = "") -> dict[str, Any]:
    """Project a sealed `SciIR` (`invar.sciir.v2`) into a backward-compatible
    `invar.ihcg.v1` dictionary for legacy Research Vault & certificate consumers."""
    states = [s.name for s in ir.symbols if s.role == SlotRole.STATE]
    control_params = [s.name for s in ir.symbols if s.role == SlotRole.CONTROL_PARAM]
    constants = [s.name for s in ir.symbols if s.role == SlotRole.CONSTANT]
    observables = [s.name for s in ir.symbols if s.role == SlotRole.DERIVED_OBSERVABLE]

    return {
        "schema_version": "invar.ihcg.v1",
        "sciir_commitment_blake3": sciir_commitment_blake3,
        "ir_id": ir.ir_id,
        "archetype": ir.archetype.value,
        "domain": ir.domain.value,
        "macro_domain": ir.to_macro_domain(),
        "user_prompt_digest": ir.user_prompt_digest,
        "states": states,
        "control_params": control_params,
        "constants": constants,
        "derived_observables": observables,
        "let_bindings": [lb.model_dump(mode="json") for lb in ir.let_bindings],
        "dynamics_or_residuals": [eq.model_dump(mode="json") for eq in ir.dynamics_or_residuals],
        "observables": [obs.model_dump(mode="json") for obs in ir.observables],
        "obligations": [ob.model_dump(mode="json") for ob in ir.obligations],
        "conservation_pools": [p.model_dump(mode="json") for p in ir.conservation_pools],
        "assumptions_ledger": [a.model_dump(mode="json") for a in ir.assumptions_ledger],
    }


def seal_sciir_ihcg(ir: SciIR) -> SealedSciIR:
    """Seal a `SciIR` through the single-authority Rust RFC 8785 JCS + BLAKE3 sealer."""
    native = _get_native()
    # Clear ihcg_commitment_blake3 before sealing so commitment is self-consistent
    unsealed = ir.model_copy(update={"ihcg_commitment_blake3": None})
    receipt = json.loads(native.seal_sciir_ihcg(unsealed.to_rust_json()))
    sciir_hash = receipt["sciir_commitment_blake3"]
    ihcg_hash = receipt["ihcg_v1_commitment_blake3"]
    sealed_ir = unsealed.model_copy(update={"ihcg_commitment_blake3": sciir_hash})
    return SealedSciIR(
        ir=sealed_ir,
        schema_version=receipt["schema_version"],
        sciir_commitment_blake3=sciir_hash,
        ihcg_v1_commitment_blake3=ihcg_hash,
        canonical_sciir_jcs=receipt["canonical_sciir_jcs"],
        ihcg_v1_projection=project_to_ihcg_v1(sealed_ir, sciir_commitment_blake3=sciir_hash),
    )
