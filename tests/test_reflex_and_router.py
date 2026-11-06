"""Reflex healer and three-tier cognitive router tests."""
from __future__ import annotations

import time

import pytest

from invar_agent.healing.error_classifier import DeterministicErrorClassifier, FailureCategory
from invar_agent.orchestrator.router import CognitiveModelRouter, CognitiveTier, TaskEventType

invar_agent_native = pytest.importorskip("invar_agent._native")
from invar_agent._native import PyDeterministicHealer


def test_tier0_reflex_heals_under_2ms_zero_tokens():
    healer = PyDeterministicHealer()
    source = "def broken(x)\n    return x\n"
    start = time.perf_counter()
    result = healer.match_and_heal("SyntaxError", "expected ':'", source, 1)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert result is not None
    assert "def broken(x):" in result[0]
    assert elapsed_ms < 2.0
    assert healer.can_heal("SyntaxError", "expected ':'") is True
    assert healer.can_heal("E0382", "use of moved value") is True
    assert healer.can_heal("WeirdError", "no idea") is False


def test_router_routes_rust_rules_and_cache_to_tier0():
    router = CognitiveModelRouter()
    decision = router.route(TaskEventType.SELF_HEAL_DIAGNOSTIC, rust_healer_matched=True)
    assert decision.tier == CognitiveTier.TIER_0_REFLEX
    assert decision.max_input_tokens == 0 and decision.max_output_tokens == 0
    decision2 = router.route(TaskEventType.SELF_HEAL_DIAGNOSTIC, fix_cache_confidence=0.95)
    assert decision2.tier == CognitiveTier.TIER_0_REFLEX


def test_router_localized_to_flash_architectural_to_pro():
    router = CognitiveModelRouter()
    flash = router.route(TaskEventType.WAVEFRONT_NODE_EXEC)
    assert flash.tier == CognitiveTier.TIER_1_FLASH
    assert flash.model_identifier == "gemini-3.8-flash"

    pro = router.route(TaskEventType.GOI_DAG_PLANNING)
    assert pro.tier == CognitiveTier.TIER_2_PRO
    assert pro.model_identifier == "gemini-3.1-pro-preview"

    escalation = router.route(TaskEventType.WAVEFRONT_NODE_EXEC, tier_1_failures=2)
    assert escalation.tier == CognitiveTier.TIER_2_PRO

    boundary = router.route(TaskEventType.WAVEFRONT_NODE_EXEC, modifies_holographic_boundary=True)
    assert boundary.tier == CognitiveTier.TIER_2_PRO

    # Budget exhaustion raises for adjoint rollback.
    with pytest.raises(RuntimeError):
        router.route(TaskEventType.WAVEFRONT_NODE_EXEC, tier_2_failures=2)


def test_error_classifier_categories():
    classifier = DeterministicErrorClassifier()
    syntax = classifier.classify_stderr_trace('  File "calc.py", line 1\nSyntaxError: expected \':\'')
    assert syntax.category == FailureCategory.SYNTAX_ERROR
    name = classifier.classify_stderr_trace("NameError: name 'Optional' is not defined")
    assert name.category == FailureCategory.NAME_ERROR
    imp = classifier.classify_stderr_trace("ModuleNotFoundError: No module named 'foo'")
    assert imp.category == FailureCategory.IMPORT_ERROR
    rust = classifier.classify_stderr_trace("error[E0382]: use of moved value: `x`")
    assert rust.category == FailureCategory.RUST_BORROW_ERROR
    assert rust.error_code == "E0382"
