"""Opt-in, unseeded DeepSeek research-to-Builder acceptance runs.

Run with an existing DEEPSEEK_API_KEY:
    .venv/bin/pytest -m live -s -v tests/test_live_meaningfulness_deepseek.py

Artifacts survive under ~/.invar-live-tests/meaningfulness/, outside the repo.
No offline blueprints, kernels, formal evidence or verification runners are
injected. A live failure is an acceptance failure, not an expected failure.
SHBT success concerns the registry's conditional closure, not physical truth.
"""
import asyncio
import json
import os
from pathlib import Path
import uuid

import pytest


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"),
]


SCENARIOS = [
    pytest.param(
        "seir_v_boundary",
        "Carry out one bounded computational study of the herd-immunity boundary "
        "in a standard SEIR model with imperfect vaccination. Use population "
        "fractions, a basic reproduction number R0 between 2.5 and 4, vaccine "
        "efficacy e=0.85, and vaccination coverage pv between 0 and 1. The "
        "effective reproduction number is Reff=R0*(1-e*pv). Find and bracket "
        "the coverage boundary for the strict criterion Reff < 1 at R0=3, "
        "and certify a safe coverage interval across the stated R0 uncertainty. "
        "Keep the strict criterion unchanged, report both sides of the boundary, "
        "and distinguish the threshold calculation from clinical evidence. "
        "Use dimensionless quantities and explicit finite parameter domains.",
        "VERIFIED_BOUNDARY_DISCOVERY",
        id="seir-v-herd-immunity-boundary",
    ),
    pytest.param(
        "carnot_ceiling",
        "Audit, rather than repair, the following falsifiable research claim "
        "in a single computational study: a cyclic heat engine operating "
        "between a hot reservoir at 600 K and a cold reservoir at 300 K can "
        "have efficiency eta >= 0.65 while obeying both thermodynamic laws. "
        "Take positive heat input Qin between 1 and 200 J per cycle, work W "
        "and rejected heat Qout nonnegative and at most 200 J, and eta in "
        "[0,1]. Use Qin=W+Qout, eta=W/Qin and the Clausius inequality "
        "Qout/300 >= Qin/600; the reversible Carnot engine provides the "
        "ceiling benchmark. Retain eta >= 0.65 as the claim being tested, "
        "even if false. Seek a justified refutation with a checkable proof, "
        "not a redesigned engine, a weakened target, or a failed-search claim. "
        "State SI units explicitly and keep all original bounds sealed.",
        "VERIFIED_FALSIFICATION",
        id="carnot-ceiling-falsification",
    ),
    pytest.param(
        "shbt_open_boundary",
        "Perform one computational SHBT frontier solvability study using the "
        "shbt-cf, shbt-precision and shbt-qc registry equations and their stated "
        "assumptions. This is an open-boundary conditional closure calculation, "
        "not a claim of experimentally established SHBT physics or a closed "
        "perpetual-motion device. Account explicitly for the external 906000 W "
        "Landauer boundary debt. For 1800 cells, vary per-cell power from 0 to "
        "1000 W and operating temperature from 0 to 35 K, with an MgB2 critical "
        "temperature of 39 K. Find a nontrivial operating point with positive "
        "net power and positive MgB2 temperature margin, using the actual "
        "registry generation, debt and margin equations. Evaluate the reversible "
        "forward/inverse framing and internal heat contributions, so that their "
        "zero-dissipation invariants arise from cancellation rather than assigning "
        "or clamping a diagnostic to zero. Separate these conservation auxiliaries "
        "from mechanisms whose activity must be measured. Include boundary work "
        "and heat costs, the registry propagation time and power scales, SI "
        "dimensions, finite domains, and a comparison baseline. Do not reinterpret "
        "zero internal dissipation as zero external boundary cost.",
        "VERIFIED_DISCOVERY",
        id="shbt-frontier-open-boundary-closure",
    ),
]


def _check_builder_admission(workspace, certificate):
    from invar_agent.builder.ingestion.pipeline import seal_ir
    from invar_agent.builder.ingestion.vault_adapter import load_vault_bundle
    from invar_agent.builder.research_bridge import validate_research_ir

    ir = seal_ir(load_vault_bundle(workspace, certificate.run_id))
    assert ir.meaningfulness_certificate == certificate.model_dump(mode="json")
    assert ir.blake3_receipt
    negative = certificate.epistemic_verdict.value == "VERIFIED_FALSIFICATION"
    assert validate_research_ir(ir, positive_import=not negative) is ir
    if negative:
        assert ir.research_import_role == "NEGATIVE_CONSTRAINT_ONLY"
        assert {entry["run_id"] for entry in ir.negative_constraints} == {certificate.run_id}
        assert not (ir.equations or ir.constants or ir.pinned_kernel_code or ir.runtime_predicates)
        with pytest.raises(ValueError, match="NEGATIVE_CONSTRAINT_ONLY"):
            validate_research_ir(ir)
    else:
        assert ir.research_import_role != "NEGATIVE_CONSTRAINT_ONLY"
        assert ir.certified_valid_domain == certificate.certified_valid_domain
    tampered = ir.model_copy(update={"blake3_receipt": "0" * 64})
    with pytest.raises(ValueError, match="receipt"):
        validate_research_ir(tampered, positive_import=not negative)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario,prompt,expected", SCENARIOS)
async def test_live_meaningfulness_deepseek(scenario, prompt, expected, monkeypatch):
    from invar_agent.config import RETIRED_DEEPSEEK_MODELS, Tier, resolve_tier_model
    from invar_agent.llm.deepseek import DeepSeekChatClient
    from invar_agent.research import meaningfulness as m
    from invar_agent.research.broadcaster import ResearchEventBroadcaster
    from invar_agent.research.default_verification import DefaultVerificationRunner
    from invar_agent.research.epistemic import compile_blueprint_ihcg, parse_run_certificate
    from invar_agent.research.search import LiteratureSearchClient
    from invar_agent.research.state_machine import ResearchStateMachine
    from invar_agent.research.vault import ResearchVaultManager

    monkeypatch.setenv("INVAR_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "1")
    monkeypatch.delenv("INVAR_OFFLINE", raising=False)
    workspace = Path.home()/".invar-live-tests"/"meaningfulness"/f"{scenario}-{uuid.uuid4().hex}"
    assert not workspace.resolve().is_relative_to(Path(__file__).resolve().parents[1])
    workspace.mkdir(parents=True)
    (workspace/"prompt.txt").write_text(prompt)
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cache_hit_tokens": 0}
    clients = {}
    for tier in (Tier.REASONING, Tier.FAST):
        model = resolve_tier_model("deepseek", tier)
        assert model not in RETIRED_DEEPSEEK_MODELS, f"Retired model configured for {tier.value}"
        clients[tier] = DeepSeekChatClient(tier=tier, model=model)

    async def complete(tier, question):
        response = await asyncio.to_thread(
            clients[tier].complete, question,
            max_tokens=65536 if tier == Tier.REASONING else 32768,
            thinking=tier == Tier.REASONING,
        )
        usage["calls"] += 1
        for key in usage:
            if key != "calls": usage[key] += getattr(response, key, 0) or 0
        with (workspace/"llm_responses.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(tier=tier.value,model=clients[tier].model,content=getattr(response,"content",None)))+"\n")
        assert response and response.content, f"Empty {tier.value} response"
        return response.content

    async def reasoning(question): return await complete(Tier.REASONING, question)
    async def fast(question): return await complete(Tier.FAST, question)

    broadcaster = ResearchEventBroadcaster()
    queue = broadcaster.subscribe()
    events = []

    async def progress(channel, payload):
        broadcaster.publish(channel, payload["event"], payload)
        while not queue.empty():
            event = queue.get_nowait()
            events.append(event)
            with (workspace/"sse_events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, default=str)+"\n")

    state = certificate = None
    run_dir = workspace
    error_type = None
    try:
        machine = ResearchStateMachine(
            workspace/"research", workspace_root=workspace,
            progress_cb=progress, llm_planner=reasoning, llm_fast=fast,
            search_client=LiteratureSearchClient(workspace/"literature_cache", offline=False, timeout=10.0, max_results=3),
        )
        assert isinstance(machine.verification_runner, DefaultVerificationRunner)
        state = await machine.plan_and_run(prompt)
        if hasattr(machine,"planner_result"):
            (workspace/"planner_result.json").write_text(json.dumps(machine.planner_result, default=str, indent=2))
        run_dir = ResearchVaultManager(workspace).run_dir(state.blueprint.blueprint_id if state.blueprint else state.run_id)
        certificate = parse_run_certificate((run_dir/"meaningfulness_certificate.json").read_text())
        assert certificate == parse_run_certificate(state.meaningfulness_certificate)
        assert isinstance(certificate, m.MeaningfulnessCertificate), certificate.model_dump_json(indent=2)
        assert m.verify_certificate_receipt(certificate)
        assert certificate.epistemic_verdict.value == state.epistemic_verdict == expected, certificate.model_dump_json(indent=2)
        assert usage["calls"] > 0 and usage["input_tokens"] + usage["output_tokens"] > 0
        assert events and all(event.get("event_version") == "2.0" for event in events)
        purity_events = [event["audit"] for event in events if event["event"] == "purity_audit"]
        assert purity_events, "No runtime preserved-vs-enforced telemetry emitted"
        for raw in purity_events:
            audit = m.PurityAudit.model_validate(raw)
            assert audit.preserved_steps + audit.projected_steps + audit.stabilized_steps == audit.total_steps
            assert 0 <= audit.enforcement_ratio <= 1
        assert certificate.purity_audit.passed and certificate.purity_audit.enforcement_ratio == 0
        assert "| Enforcement ratio |" in (run_dir/"report.md").read_text()
        rho = certificate.fidelity_audit.convergence_reliability_ratio
        if certificate.ihcg_anchor.archetype in {"ODE_PDE", "PARAM_SWEEP"}: assert rho is not None
        if rho is not None: assert rho <= 1.0
        if expected == "VERIFIED_FALSIFICATION":
            assert state.heal_count == 0
            assert certificate.falsification_certificate is not None
            proof = m.FalsificationCertificate.model_validate(certificate.falsification_certificate.model_dump())
            assert proof.passed and proof.fidelity_passed and proof.reliability_passed
            assert proof.ihcg_digest == certificate.ihcg_anchor.ihcg_digest
        elif expected == "VERIFIED_BOUNDARY_DISCOVERY":
            assert certificate.boundary_certificate and certificate.boundary_certificate.passed
            assert certificate.certified_valid_domain and certificate.domain_compression_bits > 0
        else:
            graph = compile_blueprint_ihcg(state.blueprint, stage=len(certificate.ihcg_anchor.stage_chain_digests)+1)
            invariants = {slot.symbol for slot in graph.slots if slot.semantic_tag == m.ZERO_DISSIPATION_INVARIANT}
            assert len(invariants) >= 2, "Framing and internal heat invariants must both be represented"
            assert invariants <= set(certificate.identifiability_audit.excluded_certified_invariants)
            assert certificate.identifiability_audit.coupling_passed
            assert certificate.uniqueness_audit.shbt_open_boundary_closure
        _check_builder_admission(workspace, certificate)
    except Exception as error:
        error_type = type(error).__name__
        raise
    finally:
        broadcaster.unsubscribe(queue)
        proof = getattr(certificate, "falsification_certificate", None)
        fidelity = getattr(certificate, "fidelity_audit", None)
        purity = getattr(certificate, "purity_audit", None)
        summary = dict(
            scenario=scenario, verdict=state.epistemic_verdict if state else "NO_RUN",
            reason=getattr(certificate, "failure_reason", None) or (proof.mode if proof else error_type or "EIGHT_GATES_PASSED"),
            diagnostics=getattr(certificate, "diagnostics", {}),
            rho_conv=getattr(fidelity, "convergence_reliability_ratio", None),
            enforcement_ratio=getattr(purity, "enforcement_ratio", None),
            blake3_receipt=getattr(certificate, "blake3_receipt", None),
            run_dir=str(run_dir), token_usage=usage,
            models={tier.value: client.model for tier, client in clients.items()},
        )
        line = json.dumps(summary, sort_keys=True)
        (workspace/"summary.json").write_text(line+"\n")
        print("LIVE_MEANINGFULNESS "+line, flush=True)
