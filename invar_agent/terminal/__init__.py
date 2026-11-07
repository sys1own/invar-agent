"""PTY-backed interactive terminal sessions for the embedded workspace terminal."""

from .manager import TerminalSession, TerminalSessionManager

__all__ = ["TerminalSession", "TerminalSessionManager"]
