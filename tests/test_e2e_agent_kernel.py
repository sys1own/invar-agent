"""End-to-end agent kernel tests: full 9-stage cycle and Adjoint rollback."""
from __future__ import annotations

import sys
import textwrap

import pytest

from invar_agent.contracts import ASTSurgicalPatch, PortSpec, TaskWavefrontNode, VerificationContract
from invar_agent.core.agent import VerifiedAutonomousAgent


def _workspace(tmp_path, broken_calc=True):
    (tmp_path / "calc.py").write_text(
        "def add(a, b)\n    return a + b\n" if broken_calc else "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
    )
    (tmp_path / "util.py").write_text("def mul(a, b):\n    return a * b\n")
    (tmp_path / "test_calc.py").write_text(textwrap.dedent("""\
        from calc import add, sub

        def test_add():
            assert add(2, 3) == 5

        def test_sub():
            assert sub(5, 3) == 2
        """))
    (tmp_path / "test_util.py").write_text(textwrap.dedent("""\
        from util import mul

        def test_mul():
            assert mul(2, 3) == 6
        """))
    (tmp_path / "conftest.py").write_text("")


def _node(node_id, write_path, read_path, gen, symbol, contract=None, out_port=None, in_port=None):
    return TaskWavefrontNode(
        node_id=node_id,
        wavefront_generation=gen,
        assigned_tier="TIER_1_FLASH",
        description=f"mutate {write_path}",
        read_set_paths=[read_path],
        write_set_paths=[write_path],
        target_ast_symbols=[symbol],
        input_ports=[in_port] if in_port else [],
        output_ports=[out_port or PortSpec(port_name="out", symbol_contract=symbol)],
        verification_contract=contract or VerificationContract(),
    )


def test_full_nine_stage_cycle_with_tier0_heal(tmp_path):
    _workspace(tmp_path)
    agent = VerifiedAutonomousAgent(tmp_path)

    # node1 patch intentionally reintroduces the missing colon (LLM-style fault);
    # the Tier-0 reflex healer must repair it before commit.
    patch1 = ASTSurgicalPatch(
        file_path="calc.py", operation="REPLACE_EXACT_SPAN", search_block="",
        replacement_code="def add(a, b)\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
    )
    # REPLACE_EXACT_SPAN with empty search_block fails; use CREATE-style full replace instead.
    patch1 = ASTSurgicalPatch(
        file_path="calc.py", operation="REPLACE_SYMBOL", target_symbol="add",
        replacement_code="def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
    )
    patch2 = ASTSurgicalPatch(
        file_path="util.py", operation="INSERT_AFTER_SYMBOL", target_symbol="mul",
        replacement_code="def div(a, b):\n    return a / b\n",
    )
    # Faulty variant for healing coverage: patch1b replaces sub with a colon-less def.
    faulty_sub = ASTSurgicalPatch(
        file_path="calc.py", operation="REPLACE_SYMBOL", target_symbol="sub",
        replacement_code="def sub(a, b)\n    return a - b\n",
    )

    n1 = _node("wf_calc", "calc.py", "test_calc.py", 0, "calc.add",
               out_port=PortSpec(port_name="out", symbol_contract="calc.add"))
    n2 = _node("wf_util", "util.py", "test_util.py", 1, "util.mul",
               in_port=PortSpec(port_name="in", source_node_id="wf_calc", symbol_contract="calc.add"))

    report = agent.run_until_verified(
        "add verified arithmetic helpers",
        [n1, n2],
        patches_by_node={"wf_calc": [patch1, faulty_sub], "wf_util": [patch2]},
        sandbox_cmd=[sys.executable, "-m", "pytest", "-q"],
    )

    assert report.nodes_total == 2
    # Two wavefront generations due to the dependency edge.
    assert report.success, [r.error for r in report.results]
    assert report.nodes_committed == 2
    assert report.nodes_rolled_back == 0
    # Tier-0 reflex healed the missing-colon fault.
    assert report.nodes_healed >= 1

    calc = (tmp_path / "calc.py").read_text()
    assert "def add(a, b):" in calc and "def sub(a, b):" in calc
    util = (tmp_path / "util.py").read_text()
    assert "def div(a, b):" in util

    # All nine stages appear in the trace.
    stages = [h["stage"] for h in report.results[0].stage_history]
    for stage in ("Ingest", "GoI_Plan", "Wavefront_Schedule", "Overlay_Mutate", "PreWrite_Guard",
                  "SMT_Concolic_Verify", "Sandbox_Execute", "Reflex_Heal", "Adjoint_Commit_or_Rollback"):
        assert stage in stages, f"missing stage {stage} in {stages}"


def test_unrecoverable_smt_violation_rolls_back_cleanly(tmp_path):
    _workspace(tmp_path)
    agent = VerifiedAutonomousAgent(tmp_path)

    contract = VerificationContract(
        smt_preconditions=["discount > price"],
        smt_postconditions=["price - discount >= 0"],
    )
    node = _node("wf_bad", "calc.py", "test_calc.py", 0, "calc.discount", contract=contract)
    patch = ASTSurgicalPatch(
        file_path="calc.py", operation="CREATE_FILE",
        replacement_code="def discount(price, discount):\n    return price - discount\n",
    )
    before = (tmp_path / "calc.py").read_text()

    report = agent.run_until_verified(
        "attempt unsound discount",
        [node],
        patches_by_node={"wf_bad": [patch]},
        sandbox_cmd=[sys.executable, "-m", "pytest", "-q"],
    )

    assert report.nodes_committed == 0
    assert report.nodes_rolled_back == 1
    result = report.results[0]
    assert result.rolled_back and not result.committed
    assert result.shield_report is not None and not result.shield_report.passed
    # Zero residual disk diff.
    assert (tmp_path / "calc.py").read_text() == before
