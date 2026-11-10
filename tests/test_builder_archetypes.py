"""Builder archetype classification, deterministic fallback, Plan Mode
archetype synthesis and research-parity injection (hermetic — mocked LLM,
no DeepSeek calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from invar_agent.builder.archetypes import (
    ARCHETYPE_REGISTRY, SoftwareArchetype, classify_software_archetype,
    ensure_archetype_tag, archetype_scaffold_text,
)
from invar_agent.builder.research_bridge import (
    PARITY_TEST_RELPATH, compile_research_build_packet,
)
from invar_agent.copilot.agent import CopilotAgent

INDEX_HTML = Path(__file__).resolve().parents[1] / "invar_agent" / "static" / "index.html"

ALL_ARCHETYPES = [a.value for a in SoftwareArchetype]


def test_classify_software_archetype_explicit_tags():
    for tag in ("[TARGET_ARCHETYPE: MULTI_SOLVER_BENCHMARK]",
                "[ARCHETYPE: TELEMETRY_DIGITAL_TWIN]"):
        result = classify_software_archetype(f"{tag}\nBuild the tool")
        assert result.confidence == 1.0
        assert result.is_fallback is False
    assert classify_software_archetype(
        "[TARGET_ARCHETYPE: MULTI_SOLVER_BENCHMARK] x").archetype == \
        SoftwareArchetype.MULTI_SOLVER_BENCHMARK
    assert classify_software_archetype(
        "[ARCHETYPE: TELEMETRY_DIGITAL_TWIN] x").archetype == \
        SoftwareArchetype.TELEMETRY_DIGITAL_TWIN
    # An explicit tag beats keyword noise.
    assert classify_software_archetype(
        "[TARGET_ARCHETYPE: CLI_SIMULATOR_SWEEP] interactive web dashboard"
    ).archetype == SoftwareArchetype.CLI_SIMULATOR_SWEEP


def test_classify_software_archetype_heuristics():
    cases = {
        "Build a parameter sweep CLI tool with CSV export for the model":
            SoftwareArchetype.CLI_SIMULATOR_SWEEP,
        "Build an interactive web dashboard with real-time Canvas rendering":
            SoftwareArchetype.INTERACTIVE_WEB_SIMULATOR,
        "Add Nelder-Mead parameter optimization to fit the constants":
            SoftwareArchetype.INVERSE_DESIGN_OPTIMIZER,
        "Benchmark RK4 versus symplectic Verlet solvers and compare accuracy":
            SoftwareArchetype.MULTI_SOLVER_BENCHMARK,
        "Package the kernel as a standalone Rust crate with no Python":
            SoftwareArchetype.STANDALONE_NATIVE_CRATE,
        "Build a digital twin that ingests live telemetry streams":
            SoftwareArchetype.TELEMETRY_DIGITAL_TWIN,
        "Build a Python library with bindings for the verified kernel":
            SoftwareArchetype.SCIENTIFIC_SDK_LIBRARY,
    }
    for prompt, expected in cases.items():
        result = classify_software_archetype(prompt)
        assert result.archetype == expected, (prompt, result)
        assert result.is_fallback is False
        assert result.confidence >= 0.35
        assert result.recommended_manifest and result.toolchains


def test_classify_software_archetype_fallback():
    result = classify_software_archetype("Build some code for this", has_research=True)
    assert result.archetype == SoftwareArchetype.SCIENTIFIC_SDK_LIBRARY
    assert result.is_fallback is True
    assert result.confidence == pytest.approx(0.35)
    assert "SCIENTIFIC_SDK_LIBRARY" in result.rationale
    # Fallback never aborts: manifest/toolchains are still populated.
    assert result.recommended_manifest and result.toolchains
    # Without research the fallback rationale stays generic.
    plain = classify_software_archetype("Build some code for this")
    assert plain.is_fallback is True
    assert "Research context" not in plain.rationale


def test_plan_mode_archetype_synthesis(tmp_path, monkeypatch):
    """Plan Mode returns an archetype-tagged, manifest-rich build prompt."""
    raw = json.dumps({
        "display_text": "An interactive simulator dashboard.",
        "action": {"type": "SUGGEST_BUILD_PROMPT",
                   "params": {"prompt": "Build the interactive simulator."}},
    })

    class _FakeLLM:
        model = "fake-fast"

        def generate(self, messages, temperature=0.2, max_tokens=None, **_):
            return raw

    monkeypatch.setattr(CopilotAgent, "_llm_client", lambda self: _FakeLLM())
    agent = CopilotAgent(workspace_path=str(tmp_path))
    result = agent.chat(
        "Build an interactive web simulator with real-time Canvas rendering")

    assert result["status"] == "success"
    assert result["archetype"] == "INTERACTIVE_WEB_SIMULATOR"
    assert result["is_fallback"] is False
    assert result["available_archetypes"] == ALL_ARCHETYPES
    prompt = result["clean_prompt"]
    assert prompt.startswith("[TARGET_ARCHETYPE: INTERACTIVE_WEB_SIMULATOR]")
    # Archetype-tailored manifest + verification invariants are embedded.
    assert "FastAPI" in prompt and "HTML5" in prompt
    assert "tests/test_research_parity.py" in prompt or \
        "test_research_parity" in prompt
    assert result["expert_prompt"] == prompt
    assert result["action_obj"]["clean_prompt"] == prompt

    # Ambiguous Plan Mode prompt falls back safely and says so.
    fallback = agent.chat("Build some code for this")
    assert fallback["archetype"] == "SCIENTIFIC_SDK_LIBRARY"
    assert fallback["is_fallback"] is True
    assert fallback["clean_prompt"].startswith(
        "[TARGET_ARCHETYPE: SCIENTIFIC_SDK_LIBRARY]")

    # The system prompt carries the Builder Archetype Codex.
    system = agent.system_prompt(user_query="build something")
    assert "[BUILDER ARCHETYPE CODEX]" in system
    assert "[TARGET_ARCHETYPE: <NAME>]" in system


def test_research_parity_injection(tmp_path):
    packet = compile_research_build_packet(
        f"Build a CLI sweep for @research:test_run",
        [{"run_id": "test_run", "title": "Test cooling run"}],
        workspace_dir=tmp_path)
    assert packet.classification_result is not None
    assert packet.classification_result["archetype"] == "CLI_SIMULATOR_SWEEP"
    assert PARITY_TEST_RELPATH in packet.prompt_block
    assert "tests/test_research_parity.py" in packet.prompt_block
    # The scaffold embeds the archetype tag and manifest for Plan Mode reuse.
    assert "[TARGET_ARCHETYPE: CLI_SIMULATOR_SWEEP]" in packet.prompt_block


def test_ui_archetype_badge_and_selector():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "badge-archetype" in html
    assert "archetype-selector-strip" in html
    assert "updatePromptArchetypeTag" in html
    assert "available_archetypes" in html
    assert "is_fallback" in html
    assert "(Default Fallback Applied)" in html
    # All seven archetypes are selectable client-side.
    for name in ALL_ARCHETYPES:
        assert f"'{name}'" in html
    # The registry covers every enum member with scaffold content.
    assert set(ARCHETYPE_REGISTRY) == set(SoftwareArchetype)
    for blueprint in ARCHETYPE_REGISTRY.values():
        assert blueprint.toolchains and blueprint.default_manifest
        assert blueprint.prompt_scaffold_instructions
        assert blueprint.test_requirements
        scaffold = archetype_scaffold_text(blueprint.archetype)
        assert f"[TARGET_ARCHETYPE: {blueprint.archetype.value}]" in scaffold
    # Tag helper is idempotent.
    once = ensure_archetype_tag("Build X", SoftwareArchetype.CLI_SIMULATOR_SWEEP)
    assert once.startswith("[TARGET_ARCHETYPE: CLI_SIMULATOR_SWEEP]")
    assert ensure_archetype_tag(once, SoftwareArchetype.CLI_SIMULATOR_SWEEP) == once
