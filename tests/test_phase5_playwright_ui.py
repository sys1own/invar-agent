"""Phase 5 Playwright browser acceptance tests (all 6 required UI gates).

Tests:
1. Boot with all external network requests blocked: KaTeX math still renders locally.
2. Mocked NEEDS_CLARIFICATION: all questions visible, stepper reset, composer enabled.
3. Live progress event visible within 2 s of pressing Send.
4. INCONCLUSIVE_UNVERIFIED run: zero 'Verified' labels anywhere on the page.
5. Builder UI E2E: ingest -> override -> plan -> execute -> launch -> sweep -> stop -> download .zip.
6. Terminal: `echo x` prints `x`.
"""
from __future__ import annotations

import io
import json
import re
import socket
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

import pytest
import uvicorn

from invar_agent.server import create_app

pytestmark = [pytest.mark.serial, pytest.mark.timeout(360)]

playwright_sync = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync.sync_playwright


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture()
def live_ui_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """Start a real unmodified FastAPI/uvicorn server on a loopback port for Playwright."""
    app = create_app(workspace_root=tmp_path)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/status", timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        raise RuntimeError("Live UI server failed to start within 15s")
    try:
        yield base_url, tmp_path
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def test_playwright_1_boot_offline_katex_renders(live_ui_server: tuple[str, Path]) -> None:
    """Gate 1: All external requests blocked (only localhost allowed); KaTeX math still renders."""
    base_url, _ = live_ui_server
    external_blocked: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()

        def _route_filter(route):
            url = route.request.url
            if url.startswith("http://127.0.0.1") or url.startswith("http://localhost") or url.startswith("ws://127.0.0.1") or url.startswith("data:") or url.startswith("blob:"):
                route.continue_()
            else:
                external_blocked.append(url)
                route.abort()

        context.route("**/*", _route_filter)
        page = context.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        page.wait_for_function("() => typeof window.katex !== 'undefined'", timeout=10000)

        rendered = page.evaluate("""() => {
            const html = renderMarkdown('Energy is $E = m c^2$ and $$\\\\frac{dy}{dt} = -k y$$');
            const div = document.createElement('div');
            div.id = 'pw-katex-probe';
            div.innerHTML = html;
            document.body.appendChild(div);
            return div.querySelectorAll('.katex').length;
        }""")
        assert rendered >= 2, f"Expected >=2 rendered .katex nodes offline, got {rendered}"
        assert not any("katex" in u.lower() for u in external_blocked), f"KaTeX hit external CDN: {external_blocked}"
        browser.close()


def test_playwright_2_needs_clarification_ui(live_ui_server: tuple[str, Path]) -> None:
    """Gate 2: Mocked NEEDS_CLARIFICATION shows all questions, resets stepper, keeps composer enabled."""
    base_url, _ = live_ui_server
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        page.click("#mode-research")

        def _mock_run(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "state": "NEEDS_CLARIFICATION",
                    "status": "needs_clarification",
                    "started": False,
                    "run_uid": "clar-001",
                    "questions": [
                        "What is the initial angle theta(0) in radians?",
                        "What is the damping coefficient gamma in 1/s?",
                    ],
                    "feasibility": {
                        "verdict": "NEEDS_CLARIFICATION",
                        "rationale": "Missing initial condition and damping parameter.",
                    },
                }),
            )

        page.route("**/api/v1/research/run", _mock_run)
        page.fill("#input", "Does the pendulum decay below 0.5 rad?")
        page.click("#send")

        card = page.locator(".clarification-card")
        card.wait_for(state="visible", timeout=5000)
        card_text = card.inner_text()
        assert "What is the initial angle theta(0) in radians?" in card_text
        assert "What is the damping coefficient gamma in 1/s?" in card_text

        # Stepper must be reset (no active or completed step)
        active_steps = page.locator("#research-stepper .step.active, #research-stepper .step.completed").count()
        assert active_steps == 0, f"Expected stepper reset, found {active_steps} active/completed steps"
        # Composer must remain enabled
        assert not page.locator("#input").is_disabled()
        assert not page.locator("#send").is_disabled()
        browser.close()


def test_playwright_3_live_progress_within_2s(live_ui_server: tuple[str, Path]) -> None:
    """Gate 3: Live progress event visible within 2 s of pressing Send."""
    base_url, _ = live_ui_server
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        page.click("#mode-research")

        def _mock_run_202(route):
            route.fulfill(
                status=202,
                content_type="application/json",
                body=json.dumps({
                    "state": "PLANNING",
                    "status": "planning",
                    "started": True,
                    "run_uid": "pw-live-prog-1",
                }),
            )

        page.route("**/api/v1/research/run", _mock_run_202)
        page.fill("#input", "Verify exponential decay y'(t) = -k y with y(0)=1, k=0.5")
        t0 = time.monotonic()
        page.click("#send")
        page.locator(".process-log-entries .activity-line").first.wait_for(state="visible", timeout=2000)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"Live progress took {elapsed:.2f}s (> 2.0s)"
        browser.close()


def test_playwright_4_inconclusive_no_verified_labels(live_ui_server: tuple[str, Path]) -> None:
    """Gate 4: INCONCLUSIVE_UNVERIFIED run displays zero standalone 'Verified' labels anywhere."""
    base_url, ws_root = live_ui_server
    run_id = "inc_unverified_pw_001"
    run_dir = ws_root / ".invar" / "research_vault" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "title": "Damped Oscillator Diagnostic Run",
        "status": "INCONCLUSIVE_UNVERIFIED",
        "epistemic_verdict": "INCONCLUSIVE_UNVERIFIED",
        "verdict": "OPTIMAL_DIRECT",
        "key_metrics": {
            "x_final": {"value": 0.42, "unit": "m", "evidence_status": "OBSERVED_UNVERIFIED"},
        },
        "governing_equations": ["dx/dt = v"],
        "verified_functions": ["simulation_step(buf, dt, steps)"],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "report.md").write_text(
        "# Quantitative Research Report (INCONCLUSIVE_UNVERIFIED)\n\n"
        "## Observed Numerical Findings (Unverified)\n\n"
        "x_final = 0.42 m (OBSERVED_UNVERIFIED).\n",
        encoding="utf-8",
    )
    (run_dir / "unified_module.py").write_text("def simulation_step(buf, dt, steps): pass\n", encoding="utf-8")

    # Also register in vault index so /api/v1/research/vault lists it
    index_path = ws_root / ".invar" / "research_vault" / "index.json"
    index_path.write_text(json.dumps({"runs": [metadata]}), encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        page.evaluate(f"openResearchViewer({json.dumps(run_id)})")
        page.wait_for_function(
            "() => document.getElementById('rv-title').textContent.includes('Damped Oscillator')",
            timeout=5000,
        )

        # Inspect visible text across all 3 Research Viewer tabs
        for tab in ("report", "equations", "kernel"):
            page.evaluate(f"switchResearchTab({json.dumps(tab)})")
            viewer_text = page.locator("#view-research").inner_text()
            # Match standalone "Verified" (not preceded by "Un" or inside "UNVERIFIED")
            bad_hits = re.findall(r"(?<![Uu]n)\b[Vv]erified\b", viewer_text)
            assert not bad_hits, f"Found forbidden 'Verified' label on tab {tab!r}: {viewer_text!r}"
        browser.close()


def test_playwright_5_builder_ui_e2e(live_ui_server: tuple[str, Path]) -> None:
    """Gate 5: Builder UI E2E: ingest -> override -> plan -> execute -> launch -> sweep -> stop -> download .zip."""
    base_url, _ = live_ui_server
    spec_md = (
        "# First-Order Decay\n\n"
        "Governing equation:\n"
        "$$\\frac{dy}{dt} = -k y$$\n\n"
        "Parameters:\n"
        "- k = 0.5 1/s\n"
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        page.click("#view-tab-builder")
        page.fill("#builder-spec-text", spec_md)
        page.click("#builder-ingest")

        # Wait for IR verification table to populate
        page.locator("#builder-verification tr").first.wait_for(state="visible", timeout=20000)

        # Override constant k to 0.75
        const_input = page.locator("#builder-verification input[data-constant-id]").first
        const_input.fill("0.75")

        # Enable interactive_workbench profile so we can launch & sweep
        wb_cb = page.locator("input[name='builder-profile'][value='interactive_workbench']")
        if not wb_cb.is_checked():
            wb_cb.check()

        page.click("#builder-plan")
        page.wait_for_function("() => !document.getElementById('builder-execute').disabled", timeout=20000)

        page.click("#builder-execute")
        page.wait_for_function(
            "() => (document.getElementById('builder-status').textContent || '').startsWith('COMPLETED')",
            timeout=60000,
        )

        # Launch workbench
        page.wait_for_function("() => !document.getElementById('builder-launch').disabled", timeout=10000)
        page.click("#builder-launch")
        page.wait_for_function(
            "() => (document.getElementById('builder-instrument').getAttribute('href') || '').startsWith('http')",
            timeout=20000,
        )
        wb_url = page.locator("#builder-instrument").get_attribute("href")
        assert wb_url and wb_url.startswith("http")

        # Exercise workbench /api/sweep
        req = urllib.request.Request(
            f"{wb_url}/api/sweep",
            data=json.dumps({"grid": {"k": [0.5, 0.75]}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            assert resp.status == 200
            sweep_data = json.loads(resp.read().decode("utf-8"))
            assert len(sweep_data) == 2
            assert all(item.get("ok") is True for item in sweep_data)

        # Stop / teardown workbench
        page.click("#builder-stop")
        page.wait_for_function(
            "() => (document.getElementById('builder-status').textContent || '').includes('Workbench stopped')",
            timeout=10000,
        )

        # Verify download .zip link returns a valid zip
        dl_href = page.locator("#builder-download").get_attribute("href")
        assert dl_href and dl_href.endswith(".zip")
        with urllib.request.urlopen(f"{base_url}{dl_href}", timeout=15) as resp:
            assert resp.status == 200
            zbytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            names = zf.namelist()
            assert any(n.endswith("build_receipt.json") for n in names), f"Missing build_receipt.json in {names}"
        browser.close()


def test_playwright_6_terminal_echo(live_ui_server: tuple[str, Path]) -> None:
    """Gate 6: Terminal executes `echo x` and prints `x`."""
    base_url, _ = live_ui_server
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        # Open terminal drawer tab and verify WebSocket executes echo with CR
        page.evaluate("switchDrawerTab('terminal')")
        output = page.evaluate("""async () => {
            const wsUrl = (location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/api/v1/terminal/ws';
            return await new Promise((resolve, reject) => {
                const ws = new WebSocket(wsUrl);
                let buf = '';
                const timer = setTimeout(() => { ws.close(); resolve(buf); }, 6000);
                ws.onopen = () => {
                    ws.send('echo PW_TERM_MARKER_99\\r');
                };
                ws.onmessage = (ev) => {
                    buf += ev.data;
                    const lines = buf.replace(/\\r/g, '\\n').split('\\n').map(s => s.trim());
                    if (lines.includes('PW_TERM_MARKER_99')) {
                        clearTimeout(timer);
                        ws.close();
                        resolve(buf);
                    }
                };
                ws.onerror = (err) => { clearTimeout(timer); reject(String(err)); };
            });
        }""")
        lines = [ln.strip() for ln in output.replace("\r", "\n").splitlines() if ln.strip()]
        assert "PW_TERM_MARKER_99" in lines, f"Expected PW_TERM_MARKER_99 in terminal output lines: {lines!r}"
        browser.close()
