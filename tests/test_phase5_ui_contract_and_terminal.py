"""Phase 5 offline tests (ARCH-011, ARCH-015, ARCH-016 -> BUG-015..017, BUG-023..026, BUG-031).

Covers:
- Test 18: test_windows_terminal_cr_executes (xterm.js CR triggers execution on Windows/POSIX)
- Test 20: test_plan_clarification_returns_200 (HTTP 200 structured NEEDS_CLARIFICATION response)
- Vendored KaTeX offline static assets check (BUG-031)
- Workbench /api/sweep/stream pre-stream validation (BUG-024) & /api/sweep violation reporting (BUG-025)
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from invar_agent.schemas.research import FeasibilityAssessment, FeasibilityVerdict
from invar_agent.server import create_app


def test_windows_terminal_cr_executes(tmp_path: Path) -> None:
    """Test 18: sending 'echo <marker>\\r' over /api/v1/terminal/ws executes and outputs the marker."""
    app = create_app(workspace_root=tmp_path)
    marker = "INVAR_TERM_CR_OK_7391"
    collected = ""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/terminal/ws") as ws:
            ws.send_text(f"echo {marker}\r")
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                try:
                    chunk = ws.receive_text()
                except Exception:
                    break
                collected += chunk
                # Require marker on a line after echo (or twice: echoed command + output)
                lines = [ln.strip() for ln in collected.replace("\r", "\n").splitlines() if ln.strip()]
                if any(ln == marker for ln in lines):
                    break
    assert marker in collected, f"Expected {marker!r} in terminal output, got: {collected!r}"
    lines = [ln.strip() for ln in collected.replace("\r", "\n").splitlines() if ln.strip()]
    assert any(ln == marker for ln in lines), f"Command did not execute (only echoed?): {lines!r}"


def test_plan_clarification_returns_200(tmp_path: Path) -> None:
    """Test 20: /api/v1/research/plan returns HTTP 200 with structured questions on NEEDS_CLARIFICATION."""
    app = create_app(workspace_root=tmp_path)
    clarification = FeasibilityAssessment(
        verdict=FeasibilityVerdict.NEEDS_CLARIFICATION,
        rationale="Initial angle and damping coefficient were not specified.",
        questions=[
            "What is the initial angle theta(0) in radians?",
            "What is the damping coefficient gamma in 1/s?",
        ],
        missing_fields=["theta(0)", "gamma"],
    )
    fake_outcome = {"feasibility": clarification, "blueprint": None, "source": "feasibility_gate"}

    with patch("invar_agent.server.generate_plan_with_feasibility", new=AsyncMock(return_value=fake_outcome)):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/research/plan",
                json={"question": "Does the damped pendulum stay below 1 rad?"},
            )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body.get("state") == "NEEDS_CLARIFICATION"
    assert body.get("status") == "needs_clarification"
    assert len(body.get("questions") or []) == 2
    assert "theta(0)" in (body.get("questions") or [])[0]


def test_vendored_katex_assets_served_locally(tmp_path: Path) -> None:
    """BUG-031: KaTeX JS/CSS and fonts are served from /static/vendor/katex/ without external CDN."""
    app = create_app(workspace_root=tmp_path)
    with TestClient(app) as client:
        html = client.get("/").text
        assert "cdn.jsdelivr.net/npm/katex" not in html
        assert "/static/vendor/katex/katex.min.css" in html
        assert "/static/vendor/katex/katex.min.js" in html
        assert client.get("/static/vendor/katex/katex.min.css").status_code == 200
        assert client.get("/static/vendor/katex/katex.min.js").status_code == 200


def test_workbench_sweep_validation_and_violations() -> None:
    """BUG-024 & BUG-025: /api/sweep/stream validates grid keys before streaming; /api/sweep returns violations."""
    from invar_agent.builder.emitters.workbench import APP

    assert "Unknown sweep parameter" in APP
    assert "400" in APP or "422" in APP
    assert "'ok':" in APP or '"ok":' in APP
    assert "'violations':" in APP or '"violations":' in APP
