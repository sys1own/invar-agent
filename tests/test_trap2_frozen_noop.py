"""A real SIR foundation cannot justify a dead extension or frozen ODE."""
import copy

import pytest

from trap_control_cases import run_case,sir_model


@pytest.mark.asyncio
async def test_2a_uncoupled_lockdown_stage(tmp_path):
    bp = sir_model("trap2a_noop")
    bp.target_research_stages = 2
    bp.contracts[0] = bp.contracts[0].model_copy(update={"contract_id":"c1_sir","initial_state":[*bp.contracts[0].initial_state,1,0]})
    bp.contracts.append(bp.contracts[0].model_copy(update={"contract_id":"c2_lockdown"}))
    bp.variable_map.update(c_lockdown=7,scratch=8)
    bp.slot_roles.update(c_lockdown="read_only_parameter",scratch="diagnostic")
    extension = copy.deepcopy(bp.ihcg_evidence)
    extension["domain"].update(c_lockdown=[0,1],scratch=[0,.1])
    evidence = extension["evidence"]
    evidence["slots"].extend([dict(symbol="c_lockdown",slot_idx=7,role="CONTROL_PARAM",si_dim=[0]*7),dict(symbol="scratch",slot_idx=8,role="DERIVED_OBSERVABLE",si_dim=[0]*7)])
    evidence["mechanism_parameters"].append("c_lockdown")
    evidence["equations"]["scratch"] = dict(expression="0.1*c_lockdown",si_dim=[0]*7)
    bp.ihcg_evidence["extensions"] = [extension]
    bp.ihcg_evidence["execution"] = {"stage_code_templates":{"2":"def layer_2_step(buf,dt):\n    layer_1_step(buf,dt)\n    buf.data[8] = 0.1*buf.data[7]\ndef simulation_step(buf,dt):\n    layer_2_step(buf,dt)\n"}}
    state,certificate,events = await run_case(tmp_path,bp,"INCONCLUSIVE_UNVERIFIED")
    assert state.research_stage == 2
    assert certificate.identifiability_audit.dead_parameters == ("c_lockdown",)
    assert certificate.identifiability_audit.column_norms["c_lockdown"] == 0
    assert certificate.fidelity_audit.passed
    assert any(e.get("action") == "REFLEX_HEAL" for e in events)
    assert not any(e.get("halt_healing") for e in events)


@pytest.mark.asyncio
async def test_2b_frozen_sir_is_not_a_conserved_discovery(tmp_path):
    _,certificate,events = await run_case(tmp_path,sir_model("trap2b_frozen",frozen=True),"INCONCLUSIVE_UNVERIFIED")
    assert not certificate.fidelity_audit.algebraic_equiv_passed
    assert certificate.identifiability_audit.equilibrium_restoring_force == 0
    assert min(certificate.identifiability_audit.singular_values) == 0
    assert not any(e.get("halt_healing") for e in events)
