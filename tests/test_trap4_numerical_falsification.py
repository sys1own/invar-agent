"""g(x,y) is explicitly model-defined; the sealed hypothesis is ∃g≥0.

The definition is in the prompt and sealed model, not a silent replacement of
the user claim. Q2 and entailment feasibility remain mandatory.
"""
import pytest

from phase3b_cases import blueprint,run_case,sir_model


@pytest.mark.asyncio
async def test_sir_universal_peak_counterexample(tmp_path):
    state,certificate,_ = await run_case(tmp_path,sir_model("trap4a_sir_peak",false_universal=True),"VERIFIED_FALSIFICATION")
    assert certificate.fidelity_audit.passed
    assert certificate.fidelity_audit.epsilon_disc < 1e-5
    proof = certificate.falsification_certificate
    assert proof.mode == "MODE2A_COUNTEREXAMPLE"
    assert -proof.signed_margin_at_witness-proof.epsilon_total > .25
    assert state.heal_count == 0


@pytest.mark.asyncio
async def test_nonlinear_existential_exhaustive_enclosure(tmp_path):
    bp = blueprint("trap4b_nonlinear_existential",{"g":"x*y-0.5*x**2-0.5*y**2-0.05"},
        dict(x=[.5,2.5],y=[.5,2.5],g=[-10,10]),[1.5,1.5,0],"g >= 0",
        controls=("x","y"),mechanism=("x",))
    state,certificate,_ = await run_case(tmp_path,bp,"VERIFIED_FALSIFICATION")
    proof = certificate.falsification_certificate
    assert proof.mode == "MODE2B_EXHAUSTIVE_ENCLOSURE"
    assert proof.certified_sup_upper_bound < 0
    assert 0 < proof.certified_subbox_count <= 256
    assert not proof.budget_exhausted
    assert state.heal_count == 0
