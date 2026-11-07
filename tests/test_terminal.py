"""Tests for the embedded workspace terminal (PTY manager + WebSocket)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invar_agent.terminal import TerminalSessionManager

pytestmark = pytest.mark.skipif(os.name != "posix", reason="PTY terminal requires POSIX")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_session_initialization_and_cwd(tmp_path):
    mgr = TerminalSessionManager(tmp_path)
    session = mgr.create_session(cols=90, rows=25)
    try:
        assert session.pid and session.pid > 0
        assert session.cols == 90 and session.rows == 25
        # The shell child really runs in the workspace directory. Poll briefly:
        # pty.fork() returns before the child has exec'd/chdir'ed.
        cwd_link = Path(f"/proc/{session.pid}/cwd")
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                if cwd_link.resolve() == tmp_path.resolve():
                    break
            except OSError:
                pass
            time.sleep(0.05)
        assert cwd_link.resolve() == tmp_path.resolve()
    finally:
        session.close()
        mgr.close_all()
    assert not _pid_alive(session.pid)


def test_session_resize_and_close_idempotent(tmp_path):
    mgr = TerminalSessionManager(tmp_path)
    session = mgr.create_session()
    session.resize(120, 40)
    assert session.cols == 120 and session.rows == 40
    session.close()
    session.close()  # idempotent: no double-reap errors
    assert session.closed
    assert not _pid_alive(session.pid)
    mgr.release(session)
    assert len(mgr) == 0


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INVAR_RESEARCH_SEARCH_LIVE", "0")
    import invar_agent.server as server
    server._STATE = None
    test_client = TestClient(server.create_app(workspace_root=str(tmp_path)))
    return test_client


def _collect_until(ws, needle: bytes, max_messages: int = 200) -> bytes:
    buf = b""
    for _ in range(max_messages):
        if needle in buf:
            break
        buf += ws.receive_text().encode("utf-8")
    return buf


def test_terminal_websocket_echo_resize_and_cleanup(client):
    mgr = client.app.state.terminal_manager
    pid = None
    with client.websocket_connect("/api/v1/terminal/ws") as ws:
        # Session is bound to the workspace and registered.
        assert len(mgr) == 1
        session = next(iter(mgr._sessions))
        pid = session.pid
        assert session.cwd == Path(client.app.state.terminal_manager.workspace_dir)

        # Send a command; the PTY echoes output back.
        ws.send_text('echo "TERMINAL_ALIVE"\n')
        out = _collect_until(ws, b"TERMINAL_ALIVE")
        assert b"TERMINAL_ALIVE" in out

        # Resize JSON is accepted and applied.
        ws.send_text(json.dumps({"type": "resize", "cols": 111, "rows": 33}))
        deadline = time.time() + 5
        while session.cols != 111 and time.time() < deadline:
            time.sleep(0.05)
        assert session.cols == 111 and session.rows == 33

        # Raw keystrokes are forwarded as UTF-8 bytes.
        ws.send_text("pwd\n")
        out = _collect_until(ws, b"/", max_messages=100)
        assert b"/" in out

    # Disconnect cleaned up the session and reaped the child.
    assert len(mgr) == 0
    assert pid is not None and not _pid_alive(pid)


def test_terminal_websocket_close_reaps_child_on_abort(client):
    """Abrupt client disconnect (no close frame) still terminates the shell."""
    mgr = client.app.state.terminal_manager
    ws = client.websocket_connect("/api/v1/terminal/ws")
    ws.__enter__()
    assert len(mgr) == 1
    session = next(iter(mgr._sessions))
    pid = session.pid
    # Simulate an abrupt drop by closing the portal socket directly.
    ws.__exit__(None, None, None)
    deadline = time.time() + 5
    while len(mgr) != 0 and time.time() < deadline:
        time.sleep(0.05)
    assert len(mgr) == 0
    assert not _pid_alive(pid)
