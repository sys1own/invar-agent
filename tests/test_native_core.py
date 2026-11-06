"""Native core tests: PyGoiSolver, PyHinEngine, PyDeterministicHealer, PyHisRegister."""
from __future__ import annotations

import time

import pytest

invar_agent_native = pytest.importorskip("invar_agent._native")
from invar_agent._native import PyDeterministicHealer, PyGoiSolver, PyHinEngine, PyHisRegister
from invar_agent.verification.holographic import HolographicVerifier, extract_boundary_signature


def _node(node_id, reads, writes):
    return dict(node_id=node_id, read_set=list(reads), write_set=list(writes),
                in_ports=["in"], out_ports=["out"])


def test_goi_cyclic_graph_is_not_nilpotent():
    solver = PyGoiSolver()
    for n in (_node("A", [], ["x.py"]), _node("B", ["x.py"], ["y.py"]), _node("C", ["y.py"], ["x.py"])):
        solver.add_task_node(n["node_id"], n["read_set"], n["write_set"], n["in_ports"], n["out_ports"])
    for src, dst in (("A", "B"), ("B", "C"), ("C", "A")):
        solver.add_dependency_edge(src, "out", dst, "in")
    assert solver.check_nilpotency() is False


def test_goi_acyclic_wavefronts_are_bernstein_disjoint():
    solver = PyGoiSolver()
    nodes = [
        _node("A", [], ["a.py"]),
        _node("B", [], ["b.py"]),
        _node("C", ["a.py"], ["c.py"]),
        _node("D", ["b.py"], ["d.py"]),
    ]
    for n in nodes:
        solver.add_task_node(n["node_id"], n["read_set"], n["write_set"], n["in_ports"], n["out_ports"])
    solver.add_dependency_edge("A", "out", "C", "in")
    solver.add_dependency_edge("B", "out", "D", "in")
    assert solver.check_nilpotency() is True
    wavefronts = solver.solve_wavefronts()
    assert len(wavefronts) == 2
    assert sorted(wavefronts[0]) == ["A", "B"]
    assert sorted(wavefronts[1]) == ["C", "D"]


def test_hin_engine_annihilation_and_fault_slice():
    net = PyHinEngine(128)
    ids = [net.spawn_agent(f"agent_{i}", "mutate", ["consumed"]) for i in range(100)]
    assert net.active_agent_count() == 100
    for i in range(0, 100, 2):
        net.connect_ports(ids[i], True, ids[i + 1], True)
    steps = net.reduce_to_normal_form()
    assert steps == 50
    assert net.active_agent_count() == 0

    # Causal fault slice: A -> B -> C -> D(fault), principal-to-auxiliary.
    net2 = PyHinEngine(16)
    a = net2.spawn_agent("A", "mutate", ["a_out"])
    b = net2.spawn_agent("B", "mutate", ["b_in"])
    c = net2.spawn_agent("C", "mutate", ["c_in"])
    d = net2.spawn_agent("D_fault", "verify", ["d_in"])
    net2.connect_ports(a, True, b, False)
    net2.connect_ports(b, True, c, False)
    net2.connect_ports(c, True, d, False)
    assert net2.trace_causal_fault_slice(d, 3) == ["C", "B", "A"]
    assert net2.trace_causal_fault_slice(d, 1) == ["C"]


def test_deterministic_healer_repairs_three_fault_classes_under_2ms():
    healer = PyDeterministicHealer()

    broken_def = "def foo(x)\n    return x + 1\n"
    start = time.perf_counter()
    healed = healer.match_and_heal("SyntaxError", "expected ':'", broken_def, 1)
    elapsed = (time.perf_counter() - start) * 1000.0
    assert healed is not None
    new_source, rule_id = healed
    assert "def foo(x):" in new_source
    assert elapsed < 2.0

    unused_import = "import os\n\nprint('hello')\n"
    start = time.perf_counter()
    healed2 = healer.match_and_heal("F401", "'os' imported but unused", unused_import, 1)
    elapsed2 = (time.perf_counter() - start) * 1000.0
    assert healed2 is not None
    assert "import os" not in healed2[0]
    assert elapsed2 < 2.0

    missing_typing = "def f(x: Optional[int]) -> int:\n    return x or 0\n"
    start = time.perf_counter()
    healed3 = healer.match_and_heal("NameError", "name 'Optional' is not defined", missing_typing, 1)
    elapsed3 = (time.perf_counter() - start) * 1000.0
    assert healed3 is not None
    assert "from typing import Optional" in healed3[0]
    assert elapsed3 < 2.0


def test_his_register_public_boundary_sensitivity():
    reg = PyHisRegister()
    reg.update_module_boundary("pkg/a.py", "def f(x: int) -> int")
    reg.update_module_boundary("pkg/b.py", "class B: method()")
    digest_before = reg.project_boundary_digest()

    # Private body edits never move the boundary digest.
    sig_private = extract_boundary_signature(
        "def public_api(x):\n    helper = 1\n    return x + helper\n\ndef _private():\n    return 'changed body'\n"
    )
    sig_private2 = extract_boundary_signature(
        "def public_api(x):\n    helper = 999\n    return x * helper\n\ndef _private():\n    return 'different body entirely'\n"
    )
    assert sig_private == sig_private2

    verifier = HolographicVerifier()
    files_v1 = {"pkg/a.py": "def public_api(x):\n    return x + 1\n"}
    files_v2 = {"pkg/a.py": "def public_api(x):\n    return x + 999999\n"}
    files_v3 = {"pkg/a.py": "def public_api(x, y):\n    return x + y\n"}
    d1 = verifier.encode_bulk_ast_to_boundary(files_v1)
    d2 = verifier.encode_bulk_ast_to_boundary(files_v2)
    d3 = verifier.encode_bulk_ast_to_boundary(files_v3)
    assert d1 == d2  # private body change: identical digest
    assert d1 != d3  # public signature change: digest moves
    verifier.encode_bulk_ast_to_boundary(files_v1)
    assert verifier.register.verify_boundary_parity(d1)
    verifier.encode_bulk_ast_to_boundary(files_v3)
    assert not verifier.register.verify_boundary_parity(d1)
    assert len(digest_before) == 64
