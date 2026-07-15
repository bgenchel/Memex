"""Public exceptions raised by :mod:`memex`."""

from __future__ import annotations


class MemexError(Exception):
    """Base class for Memex-specific errors."""


class BackendUnavailableError(MemexError):
    """Raised when an explicitly requested memory backend is unavailable."""


class ProcessExitedError(MemexError):
    """Raised when a task process exits without returning a result."""

    def __init__(self, pid: int | None, exitcode: int | None) -> None:
        self.pid = pid
        self.exitcode = exitcode
        super().__init__(
            f"task process {pid if pid is not None else '<unknown>'} "
            f"exited unexpectedly with code {exitcode}"
        )


class RemoteTaskError(MemexError):
    """Fallback error used when a child exception cannot be serialized."""

    def __init__(self, exception_type: str, message: str, remote_traceback: str) -> None:
        self.exception_type = exception_type
        self.remote_traceback = remote_traceback
        super().__init__(f"{exception_type}: {message}\n\nRemote traceback:\n{remote_traceback}")

