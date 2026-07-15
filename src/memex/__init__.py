"""Memory-aware process execution."""

import logging

from ._exceptions import (
    BackendUnavailableError,
    MemexError,
    ProcessExitedError,
    RemoteTaskError,
)
from ._executor import MemoryExecutor
from ._stats import DeviceStats, ExecutorStats

__all__ = [
    "BackendUnavailableError",
    "DeviceStats",
    "ExecutorStats",
    "MemexError",
    "MemoryExecutor",
    "ProcessExitedError",
    "RemoteTaskError",
]

__version__ = "0.1.0"

# Library logging remains silent unless the application configures it.
logging.getLogger("memex").addHandler(logging.NullHandler())
