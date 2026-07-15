"""Opt-in integration tests for real NVIDIA CUDA machines.

Enable with ``MEMEX_RUN_CUDA_TESTS=1``. The functions submitted to Memex live at
module scope because the executor intentionally uses multiprocessing ``spawn``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from memex import MemoryExecutor
from memex._backends import CUDABackend


CUDA_ENABLED = os.environ.get("MEMEX_RUN_CUDA_TESTS") == "1"
CONCURRENCY_ENABLED = os.environ.get("MEMEX_RUN_CUDA_CONCURRENCY") == "1"
ALLOCATION_MIB = int(os.environ.get("MEMEX_CUDA_TEST_MIB", "32"))

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not CUDA_ENABLED,
        reason="set MEMEX_RUN_CUDA_TESTS=1 to run NVIDIA hardware tests",
    ),
]


def _allocate_cuda(
    allocation_mib: int, hold_seconds: float, *, device: str
) -> tuple[str, int, int, float, float]:
    import torch

    started = time.monotonic()
    torch.cuda.set_device(device)
    tensor = torch.empty(
        allocation_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    tensor.fill_(1)
    torch.cuda.synchronize(device)
    allocated_mib = int(torch.cuda.memory_allocated(device) / (1024 * 1024))
    time.sleep(hold_seconds)
    finished = time.monotonic()
    # The process exit—not empty_cache—is what Memex relies on for full cleanup.
    return device, allocated_mib, os.getpid(), started, finished


def _raise_cuda_oom_once(
    marker: str, allocation_mib: int, *, device: str
) -> tuple[str, int, int, float, float]:
    import torch

    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return _allocate_cuda(allocation_mib, 0.1, device=device)
    else:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        raise torch.cuda.OutOfMemoryError("synthetic CUDA OOM for Memex retry test")


@pytest.fixture(scope="module", autouse=True)
def require_cuda_runtime() -> None:
    if not CUDA_ENABLED:
        return
    try:
        import pynvml  # noqa: F401
        import torch
    except ImportError as exc:
        pytest.fail(
            f"CUDA hardware tests require PyTorch and nvidia-ml-py: {exc}",
            pytrace=False,
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.fail(
            "PyTorch reports no visible CUDA devices; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
            pytrace=False,
        )
    if ALLOCATION_MIB < 1:
        pytest.fail("MEMEX_CUDA_TEST_MIB must be at least 1", pytrace=False)


@pytest.fixture
def require_allocation_headroom() -> None:
    backend = CUDABackend()
    try:
        most_free = max(device.available_mib for device in backend.devices())
    finally:
        backend.close()
    minimum_free = ALLOCATION_MIB + 512
    if most_free < minimum_free:
        pytest.skip(
            f"safety check requires at least {minimum_free} MiB free on one GPU; "
            f"most available is {most_free} MiB"
        )


def test_cuda_backend_discovery_matches_visible_torch_devices() -> None:
    import torch

    backend = CUDABackend()
    try:
        devices = backend.devices()
    finally:
        backend.close()

    assert devices
    assert all(device.name == f"cuda:{index}" for index, device in enumerate(devices))
    assert all(0 <= device.available_mib <= device.total_mib for device in devices)
    assert len(devices) == torch.cuda.device_count(), (
        f"NVML/PyTorch visibility mismatch; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    )


def test_real_cuda_task_is_injected_measured_and_process_isolated(
    require_allocation_headroom: None,
) -> None:
    parent_pid = os.getpid()
    with MemoryExecutor(
        backend="cuda",
        headroom=0,
        poll_interval=0.05,
        max_oom_retries=0,
    ) as executor:
        future = executor.submit(_allocate_cuda, ALLOCATION_MIB, 0.75)
        deadline = time.monotonic() + 30
        peak_seen = 0
        running_seen = False
        reserved_seen = False
        while not future.done() and time.monotonic() < deadline:
            stats = executor.stats()
            peak_seen = max(peak_seen, stats.observed_peak_mib or 0)
            running_seen = running_seen or stats.running == 1
            reserved_seen = reserved_seen or any(
                device.running == 1 and device.reserved_mib > 0
                for device in stats.devices
            )
            time.sleep(0.05)
        device, allocated_mib, child_pid, _, _ = future.result(timeout=30)
        final_stats = executor.stats()

    assert device.startswith("cuda:")
    assert allocated_mib >= ALLOCATION_MIB
    assert child_pid != parent_pid
    assert running_seen
    assert reserved_seen
    assert max(peak_seen, final_stats.observed_peak_mib or 0) >= ALLOCATION_MIB
    assert final_stats.completed == 1


def test_torch_cuda_oom_is_retried_in_a_fresh_process(
    tmp_path: Path, require_allocation_headroom: None
) -> None:
    marker = tmp_path / "cuda-oom-attempted"
    with MemoryExecutor(
        backend="cuda",
        headroom=0,
        poll_interval=0.05,
        max_oom_retries=1,
    ) as executor:
        result = executor.submit(
            _raise_cuda_oom_once, str(marker), ALLOCATION_MIB
        ).result(timeout=60)
        stats = executor.stats()

    assert marker.exists()
    assert result[0].startswith("cuda:")
    assert result[1] >= ALLOCATION_MIB
    assert result[2] != int(marker.read_text(encoding="utf-8"))
    assert stats.completed == 1
    assert stats.failed == 0


@pytest.mark.cuda_concurrency
@pytest.mark.skipif(
    not CONCURRENCY_ENABLED,
    reason="set MEMEX_RUN_CUDA_CONCURRENCY=1 to create concurrent CUDA contexts",
)
def test_cuda_tasks_run_concurrently_after_calibration(
    require_allocation_headroom: None,
) -> None:
    import torch

    task_count = max(2, min(4, torch.cuda.device_count() * 2))
    with MemoryExecutor(
        backend="cuda",
        headroom=0,
        poll_interval=0.05,
        max_oom_retries=0,
    ) as executor:
        calibration = executor.submit(_allocate_cuda, ALLOCATION_MIB, 0.2)
        calibration.result(timeout=30)
        futures = [
            executor.submit(_allocate_cuda, ALLOCATION_MIB, 0.75)
            for _ in range(task_count)
        ]
        results = [future.result(timeout=60) for future in futures]

    intervals = [(result[3], result[4]) for result in results]
    overlaps = any(
        left_start < right_end and right_start < left_end
        for index, (left_start, left_end) in enumerate(intervals)
        for right_start, right_end in intervals[index + 1 :]
    )
    assert overlaps, "tasks remained serial after CUDA memory calibration"
    assert all(result[0].startswith("cuda:") for result in results)
    assert len({result[2] for result in results}) == task_count
