"""UPG-019 (BUG-021, NF-07, §5.1): 40-equation ReferenceRegistry audit & domain-free reference_fingerprint."""
from __future__ import annotations

from invar_agent.research.uniqueness import ReferenceRegistry, reference_fingerprint


def test_section51_reference_registry_zero_unresolved():
    reg = ReferenceRegistry()
    assert reg.unresolved_references == (), f"Unresolved references: {reg.unresolved_references}"
    assert reg.coverage_complete is True
    assert len(reg.model_hashes) >= 40


def test_nf07_reference_fingerprint_matches_renamed_ode_and_rejects_sign_flip():
    # Lorenz ODE: dx/dt = sig*(y - x), renamed to a*(b - c) -> identical fingerprint
    t1 = {
        "sig": {"role": "CONTROL_PARAM", "si_dim": [0] * 7, "positive": True},
        "x": {"role": "DYNAMIC_STATE", "si_dim": [0] * 7, "positive": False},
        "y": {"role": "DYNAMIC_STATE", "si_dim": [0] * 7, "positive": False},
    }
    t2 = {
        "a": {"role": "CONTROL_PARAM", "si_dim": [0] * 7, "positive": True},
        "c": {"role": "DYNAMIC_STATE", "si_dim": [0] * 7, "positive": False},
        "b": {"role": "DYNAMIC_STATE", "si_dim": [0] * 7, "positive": False},
    }
    fp1 = reference_fingerprint([("ODE_RHS", "sig*(y - x)")], t1)
    fp2 = reference_fingerprint([("ODE_RHS", "a*(b - c)")], t2)
    fp_flip = reference_fingerprint([("ODE_RHS", "sig*(y + x)")], t1)

    assert fp1 == fp2
    assert fp1 != fp_flip
