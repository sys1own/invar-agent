"""Carnot: First/Second Laws refute efficiency ≥.65 at 300K/600K."""
import pytest

from trap_control_cases import blueprint,run_case


@pytest.mark.asyncio
async def test_carnot_quickxplain_mus(tmp_path):
    equations = dict(W="Qin*(1-Tc/Th)",Qout="Qin-W",eta="W/Qin",
        first_law="Qin == W+Qout",second_law="Qout/Tc >= Qin/Th",
        auxiliary_a="a >= 0",auxiliary_b="b >= 0",auxiliary_c="c >= 0")
    bp = blueprint("trap3_carnot",equations,
        dict(Tc=[300,300],Th=[600,600],Qin=[1,200],a=[0,10],b=[0,10],c=[0,10],W=[0,200],Qout=[0,200],eta=[0,1]),
        [300,600,100,1,2,3,0,0,0],"eta >= 0.65",controls=("Tc","Th","Qin","a","b","c"),mode="FORALL")
    state,certificate,_ = await run_case(tmp_path,bp,"VERIFIED_FALSIFICATION")
    proof = certificate.falsification_certificate
    assert proof.mode == "MODE1_DEDUCTIVE_MUS"
    assert proof.proof_checker == "z3_quickxplain"
    assert "h_user" in proof.minimal_unsat_core
    assert not any("auxiliary_" in label for label in proof.minimal_unsat_core)
    assert state.heal_count == 0
