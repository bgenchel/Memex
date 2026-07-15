from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from memex._backends import CPUBackend, CUDABackend, MemoryDevice, select_backend
from memex import BackendUnavailableError, MemoryExecutor


def echo_device(*, device: str) -> str:
    return device


class FakeMultiGPUBackend:
    name = "cuda"

    def devices(self) -> tuple[MemoryDevice, ...]:
        return (
            MemoryDevice("cuda:0", 100, 40),
            MemoryDevice("cuda:1", 200, 120),
        )

    def process_memory_mib(self, pid: int, device: str) -> int:
        return 10

    def close(self) -> None:
        pass


class FakeNVML:
    def __init__(self) -> None:
        self.shutdown_called = False

    def nvmlInit(self) -> None:
        pass

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True

    def nvmlDeviceGetCount(self) -> int:
        return 3

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> SimpleNamespace:
        return SimpleNamespace(
            total=(handle + 1) * 100 * 1024 * 1024,
            free=(handle + 1) * 50 * 1024 * 1024,
        )


def test_cpu_backend_reports_positive_memory() -> None:
    backend = CPUBackend()
    (device,) = backend.devices()
    assert device.name == "cpu"
    assert device.total_mib > 0
    assert 0 <= device.available_mib <= device.total_mib


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        select_backend("quantum")


def test_executor_selects_device_with_most_effective_free_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        "memex._executor.select_backend", lambda requested: FakeMultiGPUBackend()
    )
    with MemoryExecutor(backend="auto", headroom=5, poll_interval=0.02) as executor:
        assert executor.submit(echo_device).result(timeout=10) == "cuda:1"


def test_cuda_visible_devices_are_mapped_to_logical_ordinals(monkeypatch) -> None:
    nvml = FakeNVML()
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")

    backend = CUDABackend()
    try:
        devices = backend.devices()
    finally:
        backend.close()

    assert [device.name for device in devices] == ["cuda:0", "cuda:1"]
    assert [device.total_mib for device in devices] == [300, 100]
    assert nvml.shutdown_called


def test_cuda_backend_rejects_empty_visible_device_set(monkeypatch) -> None:
    nvml = FakeNVML()
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    with pytest.raises(BackendUnavailableError, match="exposes no"):
        CUDABackend()
    assert nvml.shutdown_called
