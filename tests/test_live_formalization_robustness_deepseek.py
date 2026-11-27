"""Live DeepSeek formalization robustness matrix via real HTTP routes.

Run with DEEPSEEK_API_KEY set:
    INVAR_LLM_PROVIDER=deepseek pytest -m live -s -v tests/test_live_formalization_robustness_deepseek.py

Artifacts are written outside the repo to ~/.invar-live-tests/formalization/<timestamp>/.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invar_agent.llm.telemetry import get_llm_telemetry_summary, reset_llm_telemetry
from invar_agent.research import meaningfulness as m
from invar_agent.research.vault import ResearchVaultManager


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"),
]


PROMPT_A = (
    "Carry out one bounded computational study of the herd-immunity boundary in an SEIR model "
    "with a leaky (imperfect) vaccine. Use population fractions with S+E+I+R=1 conserved, latent "
    "rate sigma=1/5.2 per day, recovery rate gamma=1/7 per day, vaccine efficacy e=0.90, "
    "vaccination coverage pv in [0,1], and basic reproduction number R0 in [2,4] with nominal R0=3 "
    "(beta=R0*gamma). The effective reproduction number is Reff=R0*(1-e*pv). Integrate the SEIR "
    "dynamics and use the simulated early-time growth of the infected fraction, not only the "
    "closed-form Reff, to find and bracket the coverage boundary for the strict criterion Reff<1 "
    "at R0=3, and certify a safe coverage interval across the whole R0 range [2,4]. Keep the "
    "criterion strict (Reff<1, never Reff<=1): the boundary point itself must not be reported as "
    "safe. Report both sides of the boundary. Express time in days, state the conversion to SI "
    "seconds (1 day = 86400 s) explicitly, and use explicit finite domains for every quantity. "
    "Do not clamp compartments, add defensive guards or hard-code the threshold. Log every "
    "assumption you add (initial conditions, seed infection, domains). Distinguish the threshold "
    "calculation from clinical or policy evidence."
)

PROMPT_R1 = (
    "Carry out one bounded computational study of the herd-immunity boundary "
    "in a standard SEIR model with imperfect vaccination. Use population "
    "fractions, a basic reproduction number R0 between 2.5 and 4, vaccine "
    "efficacy e=0.85, and vaccination coverage pv between 0 and 1. The "
    "effective reproduction number is Reff=R0*(1-e*pv). Find and bracket "
    "the coverage boundary for the strict criterion Reff < 1 at R0=3, "
    "and certify a safe coverage interval across the stated R0 uncertainty. "
    "Keep the strict criterion unchanged, report both sides of the boundary, "
    "and distinguish the threshold calculation from clinical evidence. "
    "Use dimensionless quantities and explicit finite parameter domains."
)

PROMPT_D_CASUAL = (
    "We have an imperfect vaccine that is ninety percent effective against an infection "
    "whose basic reproduction number varies between two and four around a typical value of three. "
    "Please model the SEIR outbreak dynamics to find the strict vaccination coverage threshold "
    "needed to stop early epidemic growth and identify the safe coverage range across that uncertainty."
)

_PROTECTED_WORDS = frozenset({
    "seir", "sigma", "gamma", "beta", "reff", "pv", "r0", "si", "s", "e", "i", "r",
    "in", "at", "per", "day", "days", "not", "never", "strict",
})


def perturb_prompt_words_only(text: str, seed: int) -> str:
    """Seeded perturbation touching ONLY prose words, punctuation, casing, smart dashes, and spaces.
    Never modifies digits, math operators, or protected mathematical symbols."""
    rng = random.Random(seed)
    tokens = re.split(r"(\s+)", text)
    out = []
    for tok in tokens:
        if not tok or tok.isspace():
            out.append("  " if rng.random() < 0.18 else tok)
            continue
        # Never touch tokens with digits or math operators
        if re.search(r"[0-9=<>+\-*/\[\]()]", tok):
            out.append(tok)
            continue
        m_word = re.fullmatch(r"([A-Za-z]+)([.,;:]?)", tok)
        if not m_word:
            out.append(tok)
            continue
        word, punct = m_word.group(1), m_word.group(2)
        if word.lower() in _PROTECTED_WORDS or len(word) < 5:
            out.append(word + ("" if punct and rng.random() < 0.25 else punct))
            continue
        w = word
        # Occasional typo (adjacent swap inside word)
        if rng.random() < 0.12 and len(w) >= 6:
            idx = rng.randint(1, len(w) - 3)
            w = w[:idx] + w[idx + 1] + w[idx] + w[idx + 2:]
        # Occasional odd casing
        if rng.random() < 0.15:
            w = w.upper() if rng.random() < 0.5 else w.capitalize()
        # Occasional dropped punctuation
        if punct and rng.random() < 0.3:
            punct = ""
        out.append(w + punct)
    res = "".join(out)
    # Replace hyphen in prose phrases with smart dash (never touching minus signs next to digits/vars)
    res = res.replace("herd-immunity", "herd\u2013immunity").replace("early-time", "early\u2014time")
    return f"\u201c{res}\u201d"


_RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
_ARTIFACT_ROOT = Path.home() / ".invar-live-tests" / "formalization" / _RUN_TS
_SUMMARY_ROWS: list[dict[str, Any]] = []


def _scrub(obj: Any) -> Any:
    secret = os.environ.get("DEEPSEEK_API_KEY", "")
    if isinstance(obj, str):
        return obj.replace(secret, "[REDACTED]") if secret else obj
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _run_http_case(case_id: str, prompt: str, *, seed: int | None = None, timeout_s: float = 420.0) -> dict[str, Any]:
    import invar_agent.server as server

    os.environ["INVAR_LLM_PROVIDER"] = "deepseek"
    os.environ["INVAR_RESEARCH_SEARCH_LIVE"] = "0"
    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    ws_dir = _ARTIFACT_ROOT / f"ws_{case_id}"
    ws_dir.mkdir(parents=True, exist_ok=True)

    server._STATE = None
    reset_llm_telemetry()
    app = server.create_app(ws_dir, provider="deepseek")
    client = TestClient(app)

    t0 = time.monotonic()
    resp = client.post("/api/v1/research/run", json={"question": prompt})
    http_status = resp.status_code
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}

    final_status_payload: dict[str, Any] = {}
    verdict = "UNKNOWN"
    reason_codes: list[str] = []
    run_uid = body.get("run_uid")
    vault_dir: Path | None = None

    if http_status == 200 and body.get("status") == "running":
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            st_resp = client.get("/api/v1/research/status")
            final_status_payload = st_resp.json()
            state_str = str(final_status_payload.get("state") or final_status_payload.get("status") or "").lower()
            if not final_status_payload.get("running", False) and state_str not in ("running", "planning", "starting"):
                break
            time.sleep(1.5)
        run_uid = final_status_payload.get("run_uid") or run_uid

    elapsed_s = round(time.monotonic() - t0, 2)
    tel = get_llm_telemetry_summary() or {}
    calls = int(tel.get("total_calls", 0) or 0)
    prompt_tokens = int(tel.get("prompt_tokens", 0) or 0)
    completion_tokens = int(tel.get("completion_tokens", 0) or 0)

    cert_data: dict[str, Any] = {}
    meta_data: dict[str, Any] = {}
    bp_data: dict[str, Any] = {}

    if body.get("status") == "refused":
        verdict = str((body.get("feasibility") or {}).get("verdict") or "REFUSED")
        reason_codes = [verdict]
    elif body.get("status") == "inconclusive":
        verdict = "INCONCLUSIVE_UNVERIFIED"
        reason_codes = [str(body.get("code") or "PLANNING_VALIDATION_FAILED")]
        pfd = ((body.get("diagnostics") or {}).get("per_field_diagnostics") or [])
        for d in pfd:
            c = d.get("code")
            if c and c not in reason_codes:
                reason_codes.append(str(c))
    else:
        vm = ResearchVaultManager(ws_dir)
        # Find the persisted vault directory
        if run_uid and vm.run_dir(run_uid).exists():
            vault_dir = vm.run_dir(run_uid)
        else:
            runs_root = ws_dir / ".invar" / "research_vault" / "runs"
            if runs_root.exists():
                dirs = sorted([d for d in runs_root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
                if dirs:
                    vault_dir = dirs[0]
        if vault_dir is not None:
            for fname, target in (
                ("meaningfulness_certificate.json", cert_data),
                ("metadata.json", meta_data),
                ("blueprint.json", bp_data),
            ):
                p = vault_dir / fname
                if p.exists():
                    target.update(json.loads(p.read_text(encoding="utf-8")))
        verdict = str(
            cert_data.get("epistemic_verdict")
            or meta_data.get("verdict")
            or meta_data.get("status")
            or final_status_payload.get("epistemic_verdict")
            or "UNKNOWN"
        )
        if cert_data.get("failure_reason"):
            reason_codes.append(str(cert_data["failure_reason"]))
        if meta_data.get("failure_reason") and str(meta_data["failure_reason"]) not in reason_codes:
            reason_codes.append(str(meta_data["failure_reason"]))

    row = _scrub({
        "case_id": case_id,
        "seed": seed,
        "http_status": http_status,
        "run_status": body.get("status"),
        "run_uid": run_uid,
        "verdict": verdict,
        "reason_codes": reason_codes,
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "elapsed_s": elapsed_s,
        "vault_dir": str(vault_dir) if vault_dir else None,
        "cert_verdict": cert_data.get("epistemic_verdict"),
        "ihcg_obligations": ((bp_data.get("ihcg_evidence") or {}).get("evidence") or {}).get("obligations"),
        "ihcg_domain": (bp_data.get("ihcg_evidence") or {}).get("domain"),
        "normalization_source": ((bp_data.get("ihcg_evidence") or {}).get("execution") or {}).get("normalization_source"),
    })
    _SUMMARY_ROWS.append(row)
    (_ARTIFACT_ROOT / f"case_{case_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (_ARTIFACT_ROOT / "matrix_summary.json").write_text(json.dumps(_SUMMARY_ROWS, indent=2), encoding="utf-8")
    return row


_SYNTAX_FORMAT_CODES = frozenset({
    "INVALID_EXPRESSION",
    "INVALID_DERIVED_EXPRESSION",
    "UNEXPECTED_TOKEN",
    "UNSUPPORTED_TOKEN",
    "APPROXIMATE_OPERATOR",
    "STRICT_GATE_REJECTED",
    "INVALID_MEMBERSHIP",
    "ASSIGNMENT_IN_CALL",
    "UNCLOSED_CALL",
    "UNCLOSED_PAREN",
    "UNEXPECTED_KEYWORD",
    "INVALID_PREDICATE",
})


def _is_syntax_planning_failure(row: dict[str, Any]) -> bool:
    codes = set(row.get("reason_codes") or [])
    if "PLANNING_VALIDATION_FAILED" not in codes:
        return False
    return bool(codes & _SYNTAX_FORMAT_CODES)


@pytest.mark.parametrize("case_id,prompt,seed,group", [
    ("A_prompt_a", PROMPT_A, None, "A-C"),
    ("B_r1_verbatim", PROMPT_R1, None, "A-C"),
    ("C1_seed_42", perturb_prompt_words_only(PROMPT_A, 42), 42, "A-C"),
    ("C2_seed_137", perturb_prompt_words_only(PROMPT_A, 137), 137, "A-C"),
    ("C3_seed_2026", perturb_prompt_words_only(PROMPT_A, 2026), 2026, "A-C"),
    ("D_casual", PROMPT_D_CASUAL, None, "D"),
    ("E1_contradiction", "Find a real temperature T in [200, 600] K that simultaneously satisfies T >= 500 and T <= 300.", None, "E"),
    ("E2_carnot_false", "Test whether a reversible cyclic heat engine operating between a 300 K cold reservoir and a 600 K hot reservoir can reach thermal efficiency eta >= 0.65 while obeying both thermodynamic laws (Qin in [1, 200] J, Qin = W + Qout, eta = W / Qin, Qout / 300 >= Qin / 600).", None, "E"),
    ("E3_gibberish", "zxqv plmokn bhuytf flibberty gibbet nonsense words with no math whatsoever", None, "E"),
])
def test_live_formalization_matrix(case_id: str, prompt: str, seed: int | None, group: str):
    row = _run_http_case(case_id, prompt, seed=seed)
    print(
        f"\n[LIVE {case_id}] http={row['http_status']} status={row['run_status']} "
        f"verdict={row['verdict']} codes={row['reason_codes']} "
        f"calls={row['calls']} tokens={row['total_tokens']} time={row['elapsed_s']}s"
    )
    # Never HTTP 5xx for any case A-E
    assert row["http_status"] < 500, f"Unexpected 5xx on {case_id}: {row}"

    if group in ("A-C", "D"):
        assert not _is_syntax_planning_failure(row), f"Syntax/format PLANNING_VALIDATION_FAILED on {case_id}: {row}"
        assert row["verdict"] != "UNKNOWN", f"Did not reach terminal verdict on {case_id}: {row}"
    if group == "A-C":
        # Verify sealed Reff criterion keeps strict `<` when blueprint persisted
        obs = row.get("ihcg_obligations") or []
        if obs:
            hyp_exprs = [o.get("expression", "") for o in obs if isinstance(o, dict) and o.get("obligation_class") == "HYPOTHESIS_TARGET"]
            assert any("<" in e and "<=" not in e for e in hyp_exprs), f"Strict `<` not preserved in {hyp_exprs}"
    if group == "E":
        assert row["verdict"] != "VERIFIED_DISCOVERY", f"Negative control {case_id} falsely verified!"
        assert row["verdict"] != "VERIFIED_BOUNDARY_DISCOVERY", f"Negative control {case_id} falsely boundary-verified!"
