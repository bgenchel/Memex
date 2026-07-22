"""Memory discovery and process-memory sampling backends."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import psutil

from ._exceptions import BackendUnavailableError

_LOG = logging.getLogger("memex")
_MIB = 1024 * 1024


@dataclass(frozen=True)
class MemoryDevice:
    name: str
    total_mib: int
    available_mib: int


def _process_tree_pids(pid: int) -> set[int]:
    pids = {pid}
    try:
        pids.update(child.pid for child in psutil.Process(pid).children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return pids


def _rss_mib(pid: int) -> int:
    total = 0
    try:
        processes = [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    seen: set[int] = set()
    for process in processes:
        if process.pid in seen:
            continue
        seen.add(process.pid)
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return math.ceil(total / _MIB)


class MemoryBackend(ABC):
    """Interface used by the scheduler. Third parties may implement it for testing."""

    name: str

    @abstractmethod
    def devices(self) -> tuple[MemoryDevice, ...]:
        """Return a current memory snapshot for all schedulable devices."""

    @abstractmethod
    def process_memory_mib(self, pid: int, device: str) -> int:
        """Return memory owned by a process and its descendants."""

    def close(self) -> None:
        """Release backend resources."""


class CPUBackend(MemoryBackend):
    name = "cpu"

    def devices(self) -> tuple[MemoryDevice, ...]:
        memory = psutil.virtual_memory()
        return (MemoryDevice("cpu", memory.total // _MIB, memory.available // _MIB),)

    def process_memory_mib(self, pid: int, device: str) -> int:
        return _rss_mib(pid)


class AppleBackend(CPUBackend):
    """Apple Silicon uses system RAM as a unified CPU/GPU memory pool."""

    name = "apple"

    def __init__(self) -> None:
        if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            raise BackendUnavailableError("the Apple backend requires Apple Silicon")

    def devices(self) -> tuple[MemoryDevice, ...]:
        memory = psutil.virtual_memory()
        return (MemoryDevice("mps", memory.total // _MIB, memory.available // _MIB),)


class CUDABackend(MemoryBackend):
    name = "cuda"

    def __init__(self) -> None:
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailableError(
                "the CUDA backend requires the 'nvidia-ml-py' package"
            ) from exc
        self._nvml = pynvml
        initialized = False
        try:
            pynvml.nvmlInit()
            initialized = True
            count = pynvml.nvmlDeviceGetCount()
        except Exception as exc:
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            raise BackendUnavailableError(f"NVML initialization failed: {exc}") from exc
        if count < 1:
            pynvml.nvmlShutdown()
            raise BackendUnavailableError("NVML found no NVIDIA GPUs")
        self._handles = self._visible_handles(count)
        if not self._handles:
            pynvml.nvmlShutdown()
            raise BackendUnavailableError(
                "CUDA_VISIBLE_DEVICES exposes no NVIDIA GPUs"
            )

    def _visible_handles(self, count: int) -> tuple[Any, ...]:
        """Map framework-logical ordinals to their physical NVML handles."""

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            return tuple(
                self._nvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(count)
            )
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if not tokens or tokens == ["-1"]:
            return ()
        handles = []
        for token in tokens:
            try:
                physical_index = int(token)
            except ValueError:
                handle = self._handle_for_uuid(token, count)
                if handle is None:
                    break
            else:
                if physical_index < 0 or physical_index >= count:
                    break
                handle = self._nvml.nvmlDeviceGetHandleByIndex(physical_index)
            handles.append(handle)
        return tuple(handles)

    def _handle_for_uuid(self, token: str, count: int) -> Any | None:
        getter = getattr(self._nvml, "nvmlDeviceGetHandleByUUID", None)
        if getter is not None:
            for candidate in (token, token.encode()):
                try:
                    return getter(candidate)
                except Exception:
                    pass
        # CUDA permits unambiguous abbreviated GPU UUIDs. Resolve those against
        # NVML's full UUID list when direct lookup does not accept the token.
        matches = []
        for index in range(count):
            handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
            try:
                uuid = self._nvml.nvmlDeviceGetUUID(handle)
            except Exception:
                continue
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            if str(uuid).startswith(token):
                matches.append(handle)
        return matches[0] if len(matches) == 1 else None

    def devices(self) -> tuple[MemoryDevice, ...]:
        result = []
        for index, handle in enumerate(self._handles):
            info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            result.append(
                MemoryDevice(
                    f"cuda:{index}", info.total // _MIB, info.free // _MIB
                )
            )
        return tuple(result)

    def process_memory_mib(self, pid: int, device: str) -> int:
        index = int(device.rsplit(":", 1)[1])
        handle = self._handles[index]
        total_bytes = int(self._nvml.nvmlDeviceGetMemoryInfo(handle).total)
        pids = _process_tree_pids(pid)
        functions = (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses",
        )
        memory_by_pid: dict[int, int] = {}
        for function_name in functions:
            function = getattr(self._nvml, function_name, None)
            if function is None:
                continue
            try:
                records = function(handle)
            except Exception:
                continue
            for record in records:
                record_pid = int(record.pid)
                memory = int(getattr(record, "usedGpuMemory", 0))
                if (
                    record_pid in pids
                    and 0 < memory <= total_bytes
                ):
                    # NVML exposes versioned compute and graphics queries as
                    # alternatives. The same PID may appear in several of them
                    # with slightly different readings because the calls are not
                    # atomic. Count the process once, retaining its largest
                    # reading, while still summing distinct descendant PIDs.
                    memory_by_pid[record_pid] = max(
                        memory_by_pid.get(record_pid, 0), memory
                    )
        return math.ceil(sum(memory_by_pid.values()) / _MIB)

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


class AMDBackend(MemoryBackend):
    """ROCm backend using the widely available ``rocm-smi`` command."""

    name = "amd"

    def __init__(self) -> None:
        executable = shutil.which("rocm-smi")
        if executable is None:
            raise BackendUnavailableError("the AMD backend requires 'rocm-smi'")
        self._executable = executable
        if not self.devices():
            raise BackendUnavailableError("rocm-smi found no AMD GPUs")

    def _json(self, *arguments: str) -> dict[str, Any]:
        completed = subprocess.run(
            [self._executable, *arguments, "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return json.loads(completed.stdout)

    @staticmethod
    def _number(record: dict[str, Any], contains: tuple[str, ...]) -> int | None:
        for key, value in record.items():
            lowered = key.lower()
            if all(part in lowered for part in contains):
                try:
                    return int(str(value).split()[0])
                except (TypeError, ValueError):
                    continue
        return None

    def devices(self) -> tuple[MemoryDevice, ...]:
        try:
            data = self._json("--showmeminfo", "vram")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise BackendUnavailableError(f"rocm-smi memory query failed: {exc}") from exc
        result = []
        for index, (_, record) in enumerate(sorted(data.items())):
            if not isinstance(record, dict):
                continue
            total = self._number(record, ("vram", "total"))
            used = self._number(record, ("vram", "used"))
            if total is not None and used is not None:
                # rocm-smi reports these fields in bytes.
                result.append(
                    MemoryDevice(
                        f"cuda:{index}", total // _MIB, max(0, total - used) // _MIB
                    )
                )
        return tuple(result)

    def process_memory_mib(self, pid: int, device: str) -> int:
        # ROCm installations expose different process schemas. Recursively pair
        # matching PIDs with byte-valued VRAM fields where the tooling permits it.
        try:
            data = self._json("--showpids")
        except Exception:
            return 0
        target_pids = _process_tree_pids(pid)
        used = 0

        def visit(value: Any, matched: bool = False) -> None:
            nonlocal used
            if isinstance(value, dict):
                local_match = matched
                for key, item in value.items():
                    if "pid" in key.lower():
                        try:
                            local_match = int(item) in target_pids
                        except (TypeError, ValueError):
                            pass
                for key, item in value.items():
                    if local_match and "vram" in key.lower() and "memory" in key.lower():
                        try:
                            used += int(str(item).split()[0])
                        except (TypeError, ValueError):
                            pass
                    elif isinstance(item, (dict, list)):
                        visit(item, local_match)
            elif isinstance(value, list):
                for item in value:
                    visit(item, matched)

        visit(data)
        return math.ceil(used / _MIB)


def select_backend(name: str) -> MemoryBackend:
    """Construct an explicit backend or choose the best available one."""

    normalized = name.lower()
    factories = {
        "cpu": CPUBackend,
        "cuda": CUDABackend,
        "amd": AMDBackend,
        "apple": AppleBackend,
    }
    if normalized != "auto":
        try:
            factory = factories[normalized]
        except KeyError as exc:
            choices = ", ".join(["auto", *factories])
            raise ValueError(f"unknown backend {name!r}; expected one of: {choices}") from exc
        return factory()

    for factory in (CUDABackend, AMDBackend, AppleBackend):
        try:
            backend = factory()
        except BackendUnavailableError:
            continue
        _LOG.debug("selected %s memory backend", backend.name)
        return backend
    return CPUBackend()
