"""PTY-backed terminal sessions for the embedded workspace terminal.

POSIX (Linux/macOS): each session is a real pseudo-terminal (``pty.fork``) running
the user's shell in the workspace directory, with ANSI/truecolor support, raw
keystroke forwarding, ``TIOCSWINSZ`` resizing and clean process-group-free
shutdown (SIGTERM -> SIGKILL -> ``waitpid`` reap) so no zombies survive.

Windows: falls back to a subprocess pipe (``cmd.exe``/``powershell.exe`` or
``$COMSPEC``) with the same write/read/close interface; resize is a no-op.
"""

from __future__ import annotations

import asyncio
import os
import queue
import selectors
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Set

IS_POSIX = os.name == "posix"

if IS_POSIX:  # pragma: no cover - exercised on POSIX CI/dev machines
    import fcntl
    import pty
    import termios

_CLOSE_GRACE_SECONDS = 1.0
_READ_CHUNK = 65536


def _default_shell() -> str:
    if IS_POSIX:
        for candidate in (os.environ.get("SHELL"), "/bin/bash", "/bin/sh"):
            if candidate and Path(candidate).exists():
                return candidate
        return "/bin/sh"
    return os.environ.get("COMSPEC") or "cmd.exe"


def _winsize(rows: int, cols: int) -> bytes:
    # struct winsize { unsigned short ws_row, ws_col, ws_xpixel, ws_ypixel }
    return struct.pack("HHHH", max(1, rows), max(1, cols), 0, 0)


class TerminalSession:
    """One interactive shell session.

    Output is produced by a reader thread and exposed to asyncio consumers via
    :meth:`read_chunks`. All mutating methods are safe to call from the event
    loop thread; :meth:`close` is idempotent and reaps the child process.
    """

    def __init__(self, cwd: Path, cols: int = 80, rows: int = 24):
        self.cwd = Path(cwd)
        self.cols = max(1, int(cols))
        self.rows = max(1, int(rows))
        self.pid: Optional[int] = None
        self.closed = False
        self._out: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        if IS_POSIX:
            self._start_posix()
        else:
            self._start_windows()

    # -- spawn ---------------------------------------------------------------

    def _child_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env.setdefault("PATH", os.defpath)
        return env

    def _start_posix(self) -> None:  # pragma: no cover - POSIX path
        pid, master_fd = pty.fork()
        if pid == 0:  # child: becomes the session leader on the slave pty
            try:
                os.chdir(str(self.cwd))
            except OSError:
                os._exit(127)
            os.environ.update(self._child_env())
            shell = _default_shell()
            os.execvp(shell, [shell])
            os._exit(127)
        self.pid = pid
        self._master_fd: int = master_fd
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, _winsize(self.rows, self.cols))
        os.set_blocking(master_fd, False)
        self._reader = threading.Thread(
            target=self._posix_reader, name=f"terminal-reader-{pid}", daemon=True)
        self._reader.start()

    def _start_windows(self) -> None:  # pragma: no cover - Windows path
        shell = _default_shell()
        kwargs: Dict[str, Any] = {
            "cwd": str(self.cwd),
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "bufsize": 0,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._proc = subprocess.Popen([shell], **kwargs)
        self.pid = self._proc.pid
        self._reader = threading.Thread(
            target=self._windows_reader, name=f"terminal-reader-{self.pid}", daemon=True)
        self._reader.start()

    # -- reader threads ------------------------------------------------------

    def _posix_reader(self) -> None:  # pragma: no cover - POSIX path
        sel = selectors.DefaultSelector()
        try:
            sel.register(self._master_fd, selectors.EVENT_READ)
            while not self._stop.is_set():
                events = sel.select(timeout=0.2)
                if not events:
                    continue
                try:
                    chunk = os.read(self._master_fd, _READ_CHUNK)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:  # EIO: child closed the pty -> EOF
                    break
                if not chunk:
                    break
                self._out.put(chunk)
        finally:
            sel.close()
            self._out.put(None)

    def _windows_reader(self) -> None:  # pragma: no cover - Windows path
        try:
            while not self._stop.is_set():
                chunk = self._proc.stdout.read(4096)  # type: ignore[union-attr]
                if not chunk:
                    break
                self._out.put(chunk)
        except (OSError, ValueError):
            pass
        finally:
            self._out.put(None)

    # -- public API ----------------------------------------------------------

    async def read_chunks(self) -> AsyncIterator[bytes]:
        """Yield output chunks until the child exits or the session closes."""
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self._out.get)
            if chunk is None:
                return
            yield chunk

    def write(self, data: bytes) -> None:
        """Send keystrokes to the shell (raw bytes, ANSI input supported)."""
        if self.closed or not data:
            return
        if IS_POSIX:
            fd = getattr(self, "_master_fd", None)
            if fd is None:
                return
            view = memoryview(data)
            while view:
                try:
                    n = os.write(fd, view)
                except (BlockingIOError, InterruptedError):
                    time.sleep(0.001)
                    continue
                except OSError:
                    return
                view = view[n:]
        else:
            try:
                self._proc.stdin.write(data)  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
            except (OSError, ValueError):
                pass

    def resize(self, cols: int, rows: int) -> None:
        """Resize the terminal (PTY window size); no-op on Windows pipes."""
        self.cols, self.rows = max(1, int(cols)), max(1, int(rows))
        if not IS_POSIX or self.closed:
            return
        fd = getattr(self, "_master_fd", None)
        if fd is None:
            return
        try:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, _winsize(self.rows, self.cols))
        except OSError:
            pass

    def close(self) -> None:
        """Terminate the child, reap it (no zombies), close fds. Idempotent."""
        with self._lock:
            if self.closed:
                return
            self.closed = True
        self._stop.set()

        if IS_POSIX:
            pid = self.pid
            if pid:
                self._terminate_posix(pid)
            fd = getattr(self, "_master_fd", None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                self._master_fd = -1  # type: ignore[attr-defined]
        else:
            proc = getattr(self, "_proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=_CLOSE_GRACE_SECONDS)
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    try:
                        proc.kill()
                        proc.wait(timeout=_CLOSE_GRACE_SECONDS)
                    except Exception:  # noqa: BLE001
                        pass

        if self._reader is not None:
            self._reader.join(timeout=_CLOSE_GRACE_SECONDS)
        # Drain one sentinel so a pending read_chunks() consumer unblocks.
        try:
            self._out.get_nowait()
        except queue.Empty:
            pass

    def _terminate_posix(self, pid: int) -> None:  # pragma: no cover - POSIX path
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            except PermissionError:
                break
            deadline = time.monotonic() + _CLOSE_GRACE_SECONDS
            while time.monotonic() < deadline:
                try:
                    done, _ = os.waitpid(pid, os.WNOHANG)
                    if done == pid:
                        return
                except ChildProcessError:
                    return
                time.sleep(0.01)
        # Final blocking reap; the reader thread may have hit EOF already.
        try:
            os.waitpid(pid, 0)
        except (ChildProcessError, OSError):
            pass


class TerminalSessionManager:
    """Tracks live terminal sessions for a workspace and shuts them all down."""

    def __init__(self, workspace_dir: str | Path):
        self.workspace_dir = Path(workspace_dir).resolve()
        self._sessions: Set[TerminalSession] = set()
        self._lock = threading.Lock()

    def create_session(self, cols: int = 80, rows: int = 24) -> TerminalSession:
        session = TerminalSession(self.workspace_dir, cols=cols, rows=rows)
        with self._lock:
            self._sessions.add(session)
        return session

    def release(self, session: TerminalSession) -> None:
        with self._lock:
            self._sessions.discard(session)

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions)
            self._sessions.clear()
        for session in sessions:
            session.close()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
