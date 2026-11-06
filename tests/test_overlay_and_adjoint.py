"""Overlay transaction, Adjoint rollback, and three-way AST merge tests."""
from __future__ import annotations

import textwrap

from invar_agent.contracts import ASTSurgicalPatch, PortSpec, TaskWavefrontNode, VerificationContract
from invar_agent.overlay.manager import OverlayTransactionManager
from invar_agent.overlay.structural_merger import ThreeWayASTMerger


def _node(node_id="n1", write_set=("calc.py",)):
    return TaskWavefrontNode(
        node_id=node_id,
        wavefront_generation=0,
        assigned_tier="TIER_1_FLASH",
        description="test node",
        read_set_paths=[],
        write_set_paths=list(write_set),
        target_ast_symbols=[],
        output_ports=[PortSpec(port_name="out", symbol_contract="calc.add")],
        verification_contract=VerificationContract(),
    )


def _workspace(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b)\n    return a + b\n")
    (tmp_path / "util.py").write_text("def mul(a, b):\n    return a * b\n")
    (tmp_path / "extra.py").write_text("VALUE = 1\n")


def test_three_sequential_patches_staged_disk_untouched(tmp_path):
    _workspace(tmp_path)
    manager = OverlayTransactionManager(tmp_path)
    node = _node(write_set=("calc.py", "util.py", "extra.py"))
    tx = manager.begin_transaction(node)

    patches = [
        ASTSurgicalPatch(file_path="calc.py", operation="REPLACE_SYMBOL", target_symbol="add",
                         replacement_code="def add(a, b):\n    return a + b + 0\n"),
        ASTSurgicalPatch(file_path="util.py", operation="INSERT_AFTER_SYMBOL", target_symbol="mul",
                         replacement_code="def div(a, b):\n    return a / b\n"),
        ASTSurgicalPatch(file_path="extra.py", operation="REPLACE_EXACT_SPAN",
                         search_block="VALUE = 1\n", replacement_code="VALUE = 2\n"),
    ]
    for p in patches:
        manager.stage_mutation(tx.tx_id, p)

    staged = manager.staged_files(tx.tx_id)
    assert len(staged) == 3
    assert "div" in staged["util.py"]
    # Live disk untouched.
    assert "div" not in (tmp_path / "util.py").read_text()
    assert (tmp_path / "extra.py").read_text() == "VALUE = 1\n"


def test_rollback_via_adjoint_restores_bit_for_bit(tmp_path):
    _workspace(tmp_path)
    manager = OverlayTransactionManager(tmp_path)
    node = _node(write_set=("calc.py", "extra.py"))
    tx = manager.begin_transaction(node)
    digest_pre = tx.pre_holographic_boundary

    manager.stage_mutation(tx.tx_id, ASTSurgicalPatch(
        file_path="calc.py", operation="REPLACE_SYMBOL", target_symbol="add",
        replacement_code="def add(a, b):\n    return a - b\n"))
    manager.stage_mutation(tx.tx_id, ASTSurgicalPatch(
        file_path="extra.py", operation="CREATE_FILE", replacement_code="NEW = True\n"))

    digest_mid = manager.compute_post_boundary(tx.tx_id)
    assert digest_mid != digest_pre

    ok = manager.rollback_via_adjoint(tx.tx_id)
    assert ok is True
    tx_after = manager.get_transaction(tx.tx_id)
    assert tx_after.rolled_back is True
    # Staged buffers restored: created file gone, calc.py back to disk content.
    assert manager.read_file(tx.tx_id, "calc.py") == "def add(a, b)\n    return a + b\n"
    assert manager.read_file(tx.tx_id, "extra.py") == "VALUE = 1\n"
    digest_post = manager.compute_post_boundary(tx.tx_id)
    assert digest_post == digest_pre
    # Disk was never touched at any point.
    assert (tmp_path / "calc.py").read_text() == "def add(a, b)\n    return a + b\n"
    assert not (tmp_path / "NEW").exists() and "NEW = True" not in (tmp_path / "extra.py").read_text()


def test_three_way_ast_merger_concurrent_functions():
    base = textwrap.dedent("""\
        import os
        import sys

        def helper():
            return 1
        """)
    ours = base + textwrap.dedent("""\

        def wavefront_alpha():
            return "alpha"
        """)
    theirs = base + textwrap.dedent("""\

        def wavefront_beta():
            return "beta"
        """)
    merger = ThreeWayASTMerger()
    result = merger.merge_ast_branches(base, ours, theirs, file_path="pkg/mod.py")
    assert not result.conflicts
    assert "def wavefront_alpha" in result.merged_source
    assert "def wavefront_beta" in result.merged_source
    assert "def helper" in result.merged_source
    # Imports deduplicated.
    assert result.merged_source.count("import os") == 1
    assert result.merged_source.count("import sys") == 1
    import ast

    ast.parse(result.merged_source)  # merged output must parse
