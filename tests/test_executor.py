from __future__ import annotations

import math
import os
import time
from collections import deque
from concurrent.futures import CancelledError, Future
from pathlib import Path

import pytest

from memex import MemoryExecutor, ProcessExitedError
from memex._executor import _RunningTask, _Task, _future_commitment_mib


def return_call_details(value: int, *, device: str) -> tuple[int, str, int]:
    return value * 2, device, os.getpid()


def return_without_device(value: int) -> int:
    return value


def raise_value_error(*, device: str) -> None:
    raise ValueError(f"bad task on {device}")


def sleep_then_return(delay: float, *, device: str) -> str:
    time.sleep(delay)
    return device


def timed_task(delay: float, *, device: str) -> tuple[float, float]:
    started = time.monotonic()
    time.sleep(delay)
    return started, time.monotonic()


def wait_for_peer_tasks(
    marker_dir: str, participants: int, *, device: str
) -> int:
    directory = Path(marker_dir)
    (directory / str(os.getpid())).touch()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        count = len(tuple(directory.iterdir()))
        if count >= participants:
            return count
        time.sleep(0.01)
    return len(tuple(directory.iterdir()))


def fail_once_with_oom(marker: str, *, device: str) -> str:
    path = Path(marker)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return device
    else:
        os.close(descriptor)
        raise MemoryError("synthetic out of memory")


def always_oom(attempts_file: str, *, device: str) -> None:
    with Path(attempts_file).open("a", encoding="utf-8") as stream:
        stream.write("attempt\n")
    raise MemoryError("persistent out of memory")


def exit_without_result(*, device: str) -> None:
    os._exit(3)


def custom_device_keyword(*, torch_device: str) -> str:
    return torch_device


def test_submit_returns_normal_future_and_injects_cpu_device() -> None:
    parent_pid = os.getpid()
    with MemoryExecutor(backend="cpu", headroom=0, poll_interval=0.02) as executor:
        future = executor.submit(return_call_details, 21)
        assert isinstance(future, Future)
        value, device, child_pid = future.result(timeout=10)

    assert value == 42
    assert device == "cpu"
    assert child_pid != parent_pid


def test_signature_validation_is_eager() -> None:
    with MemoryExecutor(backend="cpu", headroom=0) as executor:
        with pytest.raises(TypeError, match="device"):
            executor.submit(return_without_device, 1)


def test_device_injection_can_be_disabled() -> None:
    with MemoryExecutor(
        backend="cpu", headroom=0, inject_device=False, poll_interval=0.02
    ) as executor:
        assert executor.submit(return_without_device, 7).result(timeout=10) == 7


def test_device_parameter_is_configurable() -> None:
    with MemoryExecutor(
        backend="cpu",
        headroom=0,
        device_parameter="torch_device",
        poll_interval=0.02,
    ) as executor:
        assert executor.submit(custom_device_keyword).result(timeout=10) == "cpu"


def test_remote_exception_keeps_its_type_and_traceback_note() -> None:
    with MemoryExecutor(backend="cpu", headroom=0, poll_interval=0.02) as executor:
        future = executor.submit(raise_value_error)
        with pytest.raises(ValueError, match="bad task on cpu") as caught:
            future.result(timeout=10)

    notes = getattr(caught.value, "__notes__", [])
    assert any("Remote traceback" in note for note in notes)


def test_queued_future_can_be_cancelled() -> None:
    with MemoryExecutor(backend="cpu", headroom=0, poll_interval=0.02) as executor:
        first = executor.submit(sleep_then_return, 0.3)
        second = executor.submit(sleep_then_return, 0.01)
        assert second.cancel()
        with pytest.raises(CancelledError):
            second.result()
        assert first.result(timeout=10) == "cpu"
        deadline = time.monotonic() + 2
        while executor.stats().cancelled != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert executor.stats().cancelled == 1


def test_likely_oom_is_retried_in_a_fresh_process(tmp_path: Path) -> None:
    marker = tmp_path / "attempted"
    with MemoryExecutor(
        backend="cpu",
        headroom=0,
        poll_interval=0.02,
        max_oom_retries=2,
    ) as executor:
        future = executor.submit(fail_once_with_oom, str(marker))
        assert future.result(timeout=10) == "cpu"
        stats = executor.stats()

    assert marker.exists()
    assert stats.completed == 1
    assert stats.failed == 0
    assert stats.observed_peak_mib is not None


def test_shutdown_does_not_cancel_internal_retry_of_running_task(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "shutdown-attempted"
    executor = MemoryExecutor(
        backend="cpu", headroom=0, poll_interval=0.01, max_oom_retries=1
    )
    future = executor.submit(fail_once_with_oom, str(marker))
    deadline = time.monotonic() + 2
    while not future.running() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert future.running()
    executor.shutdown(wait=True, cancel_futures=True)

    assert future.result() == "cpu"


def test_persistent_oom_stops_at_retry_limit(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    with MemoryExecutor(
        backend="cpu", headroom=0, poll_interval=0.01, max_oom_retries=2
    ) as executor:
        future = executor.submit(always_oom, str(attempts))
        with pytest.raises(MemoryError, match="persistent"):
            future.result(timeout=10)
        stats = executor.stats()

    assert attempts.read_text(encoding="utf-8").splitlines() == [
        "attempt",
        "attempt",
        "attempt",
    ]
    assert stats.failed == 1


def test_unexpected_child_exit_raises_process_exited_error() -> None:
    with MemoryExecutor(
        backend="cpu", headroom=0, poll_interval=0.01, max_oom_retries=0
    ) as executor:
        future = executor.submit(exit_without_result)
        with pytest.raises(ProcessExitedError) as caught:
            future.result(timeout=10)

    assert caught.value.exitcode == 3


def test_stats_are_structured_and_stringifiable() -> None:
    with MemoryExecutor(backend="cpu", headroom=12, poll_interval=0.02) as executor:
        assert executor.submit(sleep_then_return, 0.05).result(timeout=10) == "cpu"
        stats = executor.stats()

    assert stats.backend == "cpu"
    assert stats.completed == 1
    assert stats.headroom_mib == 12
    assert stats.mean_task_mib is not None
    assert stats.stddev_task_mib is None
    assert stats.devices[0].device == "cpu"
    assert "MemoryExecutor" in str(stats)
    assert "Completed: 1" in str(stats)
    assert "Window mean:" in str(stats)
    assert "Window stddev:" in str(stats)


def test_default_headroom_is_2048_mib() -> None:
    with MemoryExecutor(backend="cpu") as executor:
        assert executor.stats().headroom_mib == 2048


def test_estimate_is_online_mean_plus_two_sample_stddevs() -> None:
    with MemoryExecutor(backend="cpu", headroom=0) as executor:
        with executor._lock:
            executor._record_successful_peak(100)
            executor._record_successful_peak(200)
            assert executor._memory_sample_count == 2
            assert executor._memory_mean_mib == 150
            assert executor._memory_stddev_mib() == pytest.approx(math.sqrt(5000))
            assert executor._estimated_task_mib() == math.ceil(
                150 + 2 * math.sqrt(5000)
            )


def test_memory_estimate_forgets_samples_outside_rolling_window() -> None:
    with MemoryExecutor(backend="cpu", headroom=0) as executor:
        with executor._lock:
            executor._memory_peaks_mib = deque(maxlen=3)
            executor._record_successful_peak(1000)
            executor._record_successful_peak(100)
            executor._record_successful_peak(100)
            assert executor._estimated_task_mib() > 100

            executor._record_successful_peak(100)
            assert tuple(executor._memory_peaks_mib) == (100, 100, 100)
            assert executor._memory_sample_count == 3
            assert executor._estimated_task_mib() == 100


def test_available_memory_only_charges_unconsumed_reservations() -> None:
    future: Future[object] = Future()
    task = _Task(0, future, sleep_then_return, (0.01,), {})
    running = _RunningTask(
        task=task,
        device="cuda:0",
        process=None,  # type: ignore[arg-type]
        connection=None,  # type: ignore[arg-type]
        reserved_mib=4592,
        current_mib=4000,
    )

    assert _future_commitment_mib((running,), "cuda:0") == 592
    assert _future_commitment_mib((running,), "cuda:1") == 0


def test_shutdown_cancel_futures_cancels_queued_work() -> None:
    executor = MemoryExecutor(backend="cpu", headroom=0, poll_interval=0.02)
    first = executor.submit(sleep_then_return, 0.25)
    queued = executor.submit(sleep_then_return, 0.01)
    deadline = time.monotonic() + 2
    while not first.running() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert first.running()
    executor.shutdown(wait=True, cancel_futures=True)

    assert first.result() == "cpu"
    assert queued.cancelled()


def test_calibration_runs_alone_then_scheduler_allows_concurrency() -> None:
    with MemoryExecutor(backend="cpu", headroom=0, poll_interval=0.01) as executor:
        first = executor.submit(timed_task, 0.15)
        second = executor.submit(timed_task, 0.2)
        third = executor.submit(timed_task, 0.2)
        first_interval = first.result(timeout=10)
        second_interval = second.result(timeout=10)
        third_interval = third.result(timeout=10)

    assert second_interval[0] >= first_interval[1] - 0.02
    assert third_interval[0] < second_interval[1]


def test_scheduler_has_no_job_count_ceiling_after_calibration(
    tmp_path: Path,
) -> None:
    marker_dir = tmp_path / "peers"
    marker_dir.mkdir()
    participants = 4
    with MemoryExecutor(backend="cpu", headroom=0, poll_interval=0.01) as executor:
        executor.submit(sleep_then_return, 0.02).result(timeout=10)
        futures = [
            executor.submit(wait_for_peer_tasks, str(marker_dir), participants)
            for _ in range(participants)
        ]
        counts = [future.result(timeout=10) for future in futures]

    assert counts == [participants] * participants


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"headroom": -1}, "headroom"),
        ({"poll_interval": 0}, "poll_interval"),
        ({"max_oom_retries": -1}, "max_oom_retries"),
        ({"device_parameter": ""}, "device_parameter"),
    ],
)
def test_invalid_constructor_arguments(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MemoryExecutor(backend="cpu", **arguments)  # type: ignore[arg-type]
