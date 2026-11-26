"""Comprehensive offline tests for the deterministic expression front end, typed channel,
observability, targeted meaning-locked repair, security caps, and raw-plan replay."""
from __future__ import annotations

import ast
import glob
import json
import time
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from invar_agent.research import meaningfulness as m
from invar_agent.research.blueprint_compiler import (
    BlueprintDiagnostic,
    compile_and_preflight_blueprint,
)
from invar_agent.research.expr_frontend import normalize_expression
from invar_agent.research.formalization import (
    FormalizationError,
    _check_meaning_lock,
    _evidence_pack_model,
    formalize_blueprint,
    persist_formalization_diagnostics,
)
from invar_agent.research.planner import _extract_json

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# 1. Table-driven front-end tests over every captured failure string
# ---------------------------------------------------------------------------

def _load_failure_cases():
    data = json.loads((FIXTURES_DIR / "formalization_failures.json").read_text(encoding="utf-8"))
    cases = []
    for group in ("live_captured_prompt_a", "fixture_corpus_unparseable"):
        for item in data[group]:
            cases.append((item["expression"], item["expected_outcome"]))
    return cases


@pytest.mark.parametrize("expr,expected_outcome", _load_failure_cases())
def test_fixture_failure_table(expr: str, expected_outcome: str):
    res = normalize_expression(expr)
    if expected_outcome == "normalize":
        assert res.ok and res.canonical is not None, f"Expected normalize for {expr!r}, got {res.diagnostics}"
        m._parse(res.canonical)
    else:
        assert not res.ok and res.canonical is None, f"Expected diagnostic for {expr!r}, got {res.canonical!r}"
        assert res.diagnostics, f"Missing diagnostics for {expr!r}"
        assert res.diagnostics[0].column is not None and res.diagnostics[0].column >= 1


def test_implication_precedence_and_right_associativity():
    res = normalize_expression("a > 0 and b > 0 -> c < 1 => d != 0")
    assert res.ok and res.canonical is not None
    assert ast.dump(m._parse(res.canonical)) == ast.dump(
        m._parse("implies(a > 0 and b > 0, implies(c < 1, d != 0))")
    )

    live = normalize_expression("pv == pv_star implies Reff == 1.0")
    assert live.ok and live.canonical == "implies(pv == pv_star, Reff == 1.0)"


def test_quantifier_binder_domain_check_and_mode_recording():
    ok_res = normalize_expression(
        "For all R0 in [2.5, 4] and pv in (0.5, 1.0], Reff < 1.0",
        declared_symbols={"R0", "pv", "Reff"},
        domain={"R0": (2.0, 4.0), "pv": (0.0, 1.0)},
        quantifier_mode="EXISTS",
    )
    assert ok_res.ok
    assert ok_res.canonical == "Reff < 1.0"
    assert "2.5 <= R0 <= 4" in ok_res.binder_assumptions
    assert "0.5 < pv <= 1.0" in ok_res.binder_assumptions
    assert any("quantifier_mode_mismatch" in t for t in ok_res.transforms)

    bad_res = normalize_expression(
        "For all R0 in [1.0, 5.0]: Reff < 1.0",
        declared_symbols={"R0", "Reff"},
        domain={"R0": (2.0, 4.0)},
    )
    assert not bad_res.ok
    assert bad_res.diagnostics[0].code == "BINDER_OUTSIDE_DOMAIN"


def test_unicode_and_implicit_mul_and_si_unit_strip():
    r1 = normalize_expression("2R0 + √x - y² ≤ 10 ∧ ¬(z ≠ 0)")
    assert r1.ok and r1.canonical is not None
    assert ast.dump(m._parse(r1.canonical)) == ast.dump(
        m._parse("2 * R0 + sqrt(x) - y ** 2 <= 10 and not (z != 0)")
    )

    # Unit K stripped when K is NOT a declared symbol
    r2 = normalize_expression("T_h = 600 K", declared_symbols={"T_h"})
    assert r2.ok and r2.canonical == "T_h == 600"
    assert ("600", "K") in r2.stripped_units

    # When K IS a declared symbol, do NOT silently strip it
    r3 = normalize_expression("T_h = 600 K", declared_symbols={"T_h", "K"})
    assert not r3.ok


# ---------------------------------------------------------------------------
# 2. Hypothesis property tests
# ---------------------------------------------------------------------------

_SYM = st.sampled_from(["x", "y", "z", "R0", "pv", "Reff", "T"])
_NUM = st.integers(min_value=1, max_value=50).map(str)
_CMP = st.sampled_from(["<", "<=", ">", ">=", "==", "!="])


def _arith_strategy():
    leaf = st.one_of(_SYM, _NUM)
    return st.recursive(
        leaf,
        lambda children: st.one_of(
            st.tuples(children, st.sampled_from(["+", "-", "*", "/"]), children).map(
                lambda t: f"({t[0]} {t[1]} {t[2]})"
            ),
            st.tuples(children, st.sampled_from(["2", "3"])).map(lambda t: f"({t[0]} ** {t[1]})"),
            children.map(lambda c: f"abs({c})"),
        ),
        max_leaves=6,
    )


def _formula_strategy():
    atom = st.tuples(_arith_strategy(), _CMP, _arith_strategy()).map(lambda t: f"{t[0]} {t[1]} {t[2]}")
    return st.one_of(
        atom,
        st.tuples(atom, st.sampled_from(["and", "or"]), atom).map(lambda t: f"({t[0]}) {t[1]} ({t[2]})"),
        atom.map(lambda a: f"not ({a})"),
    )


def _meaning_signature(expr: str):
    tree = m._parse(expr)
    nums = sorted(
        Fraction(str(n.value))
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float)
    )
    cmp_ops = [type(op).__name__ for n in ast.walk(tree) if isinstance(n, ast.Compare) for op in n.ops]
    bool_ops = [type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, ast.BoolOp)]
    syms = sorted(m._variables(tree))
    return (nums, cmp_ops, bool_ops, syms)


@settings(max_examples=60, deadline=None)
@given(expr=_formula_strategy())
def test_hypothesis_idempotence_and_surface_invariance(expr: str):
    # 1. Valid expression passes through byte-identical
    r0 = normalize_expression(expr)
    assert r0.ok and r0.canonical == expr and r0.transforms == ()

    # 2. Surface variants produce identical AST and meaning signature
    variant = (
        expr.replace("**", "^")
        .replace("<=", "≤")
        .replace(">=", "≥")
        .replace("!=", "≠")
        .replace("==", "=")
        .replace(" and ", " && ")
        .replace(" or ", " || ")
        .replace("not ", "! ")
    )
    rv = normalize_expression(variant)
    assert rv.ok and rv.canonical is not None
    assert ast.dump(m._parse(rv.canonical)) == ast.dump(m._parse(expr))
    assert _meaning_signature(rv.canonical) == _meaning_signature(expr)

    # 3. Idempotence on canonical output
    r2 = normalize_expression(rv.canonical)
    assert r2.ok and r2.canonical == rv.canonical and r2.transforms == ()


def test_linear_time_under_caps():
    # Balanced sum of 64 terms stays well within AST depth cap (depth ~7) while exercising lexer/parser speed
    terms = ["x"] * 64
    while len(terms) > 1:
        terms = [f"({terms[i]} + {terms[i+1]})" for i in range(0, len(terms), 2)]
    expr = terms[0] + " <= 1000"
    variant = expr.replace("<=", "≤")
    t0 = time.monotonic()
    res = normalize_expression(variant)
    dt = time.monotonic() - t0
    assert res.ok and dt < 0.2


# ---------------------------------------------------------------------------
# 3. Security rejection & caps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    '__import__("os").system("echo pwn")',
    "x.__class__.__mro__",
    "(lambda x: x)(1)",
    'f"{x}"',
    "9**9**9",
    "9^9^9",
    "(" * 80 + "x" + ")" * 80,
    "x + " * 300_000 + "1",
    "x ≈ 1.0",
    "x ~ 1.0",
    "x is approximately 1.0",
])
def test_security_payloads_rejected_or_capped(payload: str):
    res = normalize_expression(payload)
    assert not res.ok and res.canonical is None
    assert res.diagnostics


# ---------------------------------------------------------------------------
# 4. Valid-input byte-identity over all fixture expressions
# ---------------------------------------------------------------------------

def test_valid_fixture_expressions_byte_identical():
    valid_exprs: set[str] = set()

    def _collect(obj):
        if isinstance(obj, str) and "{" in obj and ("obligations" in obj or "equations" in obj):
            try:
                parsed = _extract_json(obj)
                if isinstance(parsed, dict):
                    _collect(parsed)
            except Exception:
                pass
            return
        if isinstance(obj, dict):
            for ob in obj.get("obligations") or []:
                if isinstance(ob, dict) and isinstance(ob.get("expression"), str):
                    valid_exprs.add(ob["expression"])
            if isinstance(obj.get("equations"), dict):
                for eq in obj["equations"].values():
                    if isinstance(eq, dict) and isinstance(eq.get("expression"), str):
                        valid_exprs.add(eq["expression"])
            for ea in obj.get("environmental_assumptions") or []:
                if isinstance(ea, str):
                    valid_exprs.add(ea)
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v)

    for p in sorted(glob.glob(str(FIXTURES_DIR / "*.json"))):
        if Path(p).name == "formalization_failures.json":
            continue
        _collect(json.loads(Path(p).read_text(encoding="utf-8")))

    checked = 0
    for s in sorted(valid_exprs):
        try:
            m._parse(s)
        except Exception:
            continue
        checked += 1
        res = normalize_expression(s)
        assert res.ok and res.canonical == s and res.transforms == ()
    assert checked >= 100


# ---------------------------------------------------------------------------
# 5. Typed channel lowering tests
# ---------------------------------------------------------------------------

def _seir_bp():
    return {
        "blueprint_id": "seir_typed_test",
        "title": "SEIR typed channel",
        "falsifiable_hypothesis": "Reff < 1",
        "variable_map": {"R0": 0, "e": 1, "pv": 2, "Reff": 3},
        "typed_slots": [
            {"symbol": "R0", "slot_idx": 0, "si_dim": [0]*7, "role": "CONTROL_PARAM"},
            {"symbol": "e", "slot_idx": 1, "si_dim": [0]*7, "role": "CONTROL_PARAM"},
            {"symbol": "pv", "slot_idx": 2, "si_dim": [0]*7, "role": "CONTROL_PARAM"},
            {"symbol": "Reff", "slot_idx": 3, "si_dim": [0]*7, "role": "DERIVED_OBSERVABLE"},
        ],
    }


def test_typed_channel_lowering_preserves_strict_inequality_and_notes_unhashed():
    bp = _seir_bp()
    pack_base = {
        "evidence": {
            "equations": {
                "Reff": {"expression": "R0 * (1 - e * pv)", "si_dim": [0]*7, "kind": "ALGEBRAIC"}
            },
            "environmental_assumptions": [
                {"bound": {"symbol": "pv", "lo": 0.0, "hi": 1.0, "lo_open": False, "hi_open": False}},
                {"fixed": {"symbol": "e", "value": 0.9}},
            ],
            "quantifier_binders": [
                {"symbol": "R0", "lo": 2.0, "hi": 4.0, "lo_open": False, "hi_open": False}
            ],
            "obligations": [
                {
                    "id": "h0",
                    "predicate": {"rel": {"lhs": "Reff", "op": "<", "rhs": "1"}},
                    "obligation_class": "HYPOTHESIS_TARGET",
                }
            ],
            "quantifier_mode": "BOUNDARY",
            "mechanism_parameters": ["pv"],
            "control_values": {"R0": 3.0, "e": 0.9},
        },
        "domain": {"R0": [2.0, 4.0], "e": [0.9, 0.9], "pv": [0.0, 1.0], "Reff": [0.0, 5.0]},
        "notes": ["Time in days; 1 day = 86400 s. Calculation is not clinical evidence."],
        "archetype": "PARAM_SWEEP",
    }
    res1 = compile_and_preflight_blueprint(bp, pack_base, question="SEIR study", run_z3_preflight=True)
    assert res1.ok, res1.error_summary()
    assert res1.compiled_graph is not None
    # Strict `<` preserved
    ob_expr = res1.compiled_graph.obligations[0].expression
    assert ob_expr == "Reff < 1"
    # Notes lowered into blueprint assumptions_ledger as kind="NOTE", NOT in sealed IhcgGraph
    note_entries = [a for a in res1.assumptions_ledger if a.get("kind") == "NOTE"]
    assert len(note_entries) == 1
    assert all(a.kind != "NOTE" for a in res1.compiled_graph.assumptions_ledger)

    # Changing notes must NOT change sealed IHCG hash
    pack_other_notes = json.loads(json.dumps(pack_base))
    pack_other_notes["notes"] = ["Completely different prose note."]
    res2 = compile_and_preflight_blueprint(bp, pack_other_notes, question="SEIR study", run_z3_preflight=True)
    assert res2.ok and res2.compiled_graph is not None
    assert res1.compiled_graph.h_ihcg == res2.compiled_graph.h_ihcg


def test_typed_channel_rejects_binder_outside_domain():
    bp = _seir_bp()
    pack = {
        "evidence": {
            "equations": {"Reff": {"expression": "R0 * (1 - e * pv)", "si_dim": [0]*7, "kind": "ALGEBRAIC"}},
            "quantifier_binders": [{"symbol": "R0", "lo": 1.0, "hi": 5.0}],
            "obligations": [
                {"id": "h0", "predicate": {"rel": {"lhs": "Reff", "op": "<", "rhs": "1"}}, "obligation_class": "HYPOTHESIS_TARGET"}
            ],
            "quantifier_mode": "BOUNDARY",
            "mechanism_parameters": ["pv"],
        },
        "domain": {"R0": [2.0, 4.0], "e": [0.9, 0.9], "pv": [0.0, 1.0], "Reff": [0.0, 5.0]},
        "archetype": "PARAM_SWEEP",
    }
    res = compile_and_preflight_blueprint(bp, pack, question="SEIR study", run_z3_preflight=True)
    assert not res.ok
    assert any(d.code == "BINDER_OUTSIDE_DOMAIN" for d in res.diagnostics)


# ---------------------------------------------------------------------------
# 6. Meaning-lock guard, observability persistence, and HTTP 200 inconclusive
# ---------------------------------------------------------------------------

def test_meaning_lock_guard_rejects_weakening_and_number_drift():
    declared = {"Reff", "R0", "e", "pv"}
    ok, _ = _check_meaning_lock("Reff < 1", "Reff < 1", declared)
    assert ok
    # Weakening < to <= MUST be rejected
    ok_weak, reason = _check_meaning_lock("Reff < 1", "Reff <= 1", declared)
    assert not ok_weak and "strict" in reason
    # Dropping numeric literal MUST be rejected
    ok_num, reason = _check_meaning_lock("R0 * (1 - 0.9 * pv) < 1", "R0 * (1 - 0.85 * pv) < 1", declared)
    assert not ok_num and "numeric" in reason
    # Inventing undeclared symbol MUST be rejected
    ok_sym, reason = _check_meaning_lock("Reff < 1", "Reff < magic_const", declared)
    assert not ok_sym and "undeclared" in reason


def test_diagnostics_persistence_redacts_secrets(tmp_path, monkeypatch):
    fake_secret = "sk-test-secret-1234567890abcdef"
    monkeypatch.setenv("DEEPSEEK_API_KEY", fake_secret)
    out_dir = persist_formalization_diagnostics(
        tmp_path,
        "run-redact-test",
        [{"attempt": 1, "raw_response": f"leaked {fake_secret}", "error": "bad"}],
        extra_secrets=[fake_secret],
    )
    assert out_dir is not None
    saved = (out_dir / "attempt_1.json").read_text(encoding="utf-8")
    assert fake_secret not in saved
    assert "[REDACTED]" in saved


def test_server_run_returns_http_200_inconclusive_on_formalization_error(tmp_path, monkeypatch):
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")
    import invar_agent.server as server
    monkeypatch.setattr(server, "_STATE", None)

    async def _fail_plan(*args, **kwargs):
        raise FormalizationError([{
            "attempt": 1,
            "error_type": "ValueError",
            "error": "[UNEXPECTED_TOKEN] obligations[0].expression: prose",
            "diagnostics": [{"code": "UNEXPECTED_TOKEN", "field": "obligations[0].expression", "column": 5}],
            "encoding_diagnostics": [],
        }])

    monkeypatch.setattr(server, "generate_plan_with_feasibility", _fail_plan)
    from fastapi.testclient import TestClient
    app = server.create_app(tmp_path / "ws")
    server._STATE.api_key = "sk-dummy-key-for-test-only-0000"
    c = TestClient(app)
    r = c.post("/api/v1/research/run", json={"question": "test prompt"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "inconclusive"
    assert body["code"] == "PLANNING_VALIDATION_FAILED"
    assert body["started"] is False
    run_uid = body["run_uid"]
    # Verify persisted attempt log + vault inconclusive record
    assert (tmp_path / "ws" / ".invar" / "diagnostics" / "formalization" / run_uid / "attempt_1.json").exists()
    from invar_agent.research.vault import ResearchVaultManager
    vdir = ResearchVaultManager(tmp_path / "ws").run_dir(run_uid)
    assert (vdir / "meaningfulness_certificate.json").exists()
    assert (vdir / "metadata.json").exists()
