"""Spawned child process entry point."""

from __future__ import annotations

import math
import pickle
import sys
import traceback
from multiprocessing.connection import Connection
from typing import Any, Callable


def _self_peak_rss_mib() -> int:
    """Return this process's OS high-water RSS where the platform exposes it."""

    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return 0
    # getrusage uses bytes on macOS and KiB on Linux/BSD.
    bytes_used = peak if sys.platform == "darwin" else peak * 1024
    return math.ceil(bytes_used / (1024 * 1024))


def is_likely_oom(exception: BaseException) -> bool:
    if isinstance(exception, MemoryError):
        return True
    type_name = type(exception).__name__.lower()
    if "outofmemory" in type_name or "out_of_memory" in type_name:
        return True
    message = str(exception).lower()
    indicators = (
        "out of memory",
        "outofmemory",
        "cannot allocate memory",
        "cuda error: memory allocation",
        "cublas_status_alloc_failed",
        "hip error out of memory",
        "mps backend out of memory",
        "std::bad_alloc",
    )
    return any(indicator in message for indicator in indicators)


def run_task(
    connection: Connection,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Execute one callable and send a serialization-safe outcome to the parent."""

    try:
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            remote_traceback = traceback.format_exc()
            try:
                pickle.dumps(exc)
            except Exception:
                payload = (
                    "remote_error",
                    type(exc).__qualname__,
                    str(exc),
                    remote_traceback,
                    is_likely_oom(exc),
                )
            else:
                payload = (
                    "error",
                    exc,
                    remote_traceback,
                    is_likely_oom(exc),
                    _self_peak_rss_mib(),
                )
            if payload[0] == "remote_error":
                payload = (*payload, _self_peak_rss_mib())
            connection.send(payload)
        else:
            try:
                connection.send(("result", result, _self_peak_rss_mib()))
            except BaseException as exc:
                connection.send(
                    (
                        "remote_error",
                        type(exc).__qualname__,
                        f"task result could not be serialized: {exc}",
                        traceback.format_exc(),
                        False,
                        _self_peak_rss_mib(),
                    )
                )
    finally:
        connection.close()
