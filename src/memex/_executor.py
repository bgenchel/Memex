"""Memory-aware, process-per-task Executor implementation."""

from __future__ import annotations

import inspect
import logging
import math
import multiprocessing
import threading
from collections import deque
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Callable, TypeVar

from ._backends import MemoryBackend, MemoryDevice, select_backend
from ._exceptions import ProcessExitedError, RemoteTaskError
from ._stats import DeviceStats, ExecutorStats
from ._worker import run_task

_LOG = logging.getLogger("memex")
_T = TypeVar("_T")


@dataclass
class _Task:
    identifier: int
    future: Future[Any]
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    oom_retries: int = 0
    started: bool = False
    peak_mib: int = 0


@dataclass
class _RunningTask:
    task: _Task
    device: str
    process: BaseProcess
    connection: Connection
    reserved_mib: int
    peak_mib: int = 0


class MemoryExecutor(Executor):
    """An Executor whose process concurrency is controlled by available memory.

    Every task attempt runs in a fresh process created with the ``spawn`` start
    method. The first task calibrates the executor; subsequent launches reserve
    the largest per-task memory peak observed during this executor session.
    """

    def __init__(
        self,
        *,
        headroom: int = 1024,
        poll_interval: float = 0.25,
        inject_device: bool = True,
        device_parameter: str = "device",
        backend: str = "auto",
        max_oom_retries: int = 2,
    ) -> None:
        if isinstance(headroom, bool) or not isinstance(headroom, int) or headroom < 0:
            raise ValueError("headroom must be a non-negative integer number of MiB")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")
        if not device_parameter or not isinstance(device_parameter, str):
            raise ValueError("device_parameter must be a non-empty string")
        if (
            isinstance(max_oom_retries, bool)
            or not isinstance(max_oom_retries, int)
            or max_oom_retries < 0
        ):
            raise ValueError("max_oom_retries must be a non-negative integer")

        self._headroom = headroom
        self._poll_interval = float(poll_interval)
        self._inject_device = inject_device
        self._device_parameter = device_parameter
        self._max_oom_retries = max_oom_retries
        self._backend: MemoryBackend = select_backend(backend)
        self._backend_query_lock = threading.Lock()
        self._context = multiprocessing.get_context("spawn")

        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._pending: deque[_Task] = deque()
        self._running: dict[int, _RunningTask] = {}
        self._next_identifier = 0
        self._observed_peak_mib: int | None = None
        self._calibration_task: int | None = None
        self._oom_draining = False
        self._growth_pause_at_running: int | None = None
        self._shutdown_requested = False
        self._backend_closed = False

        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._last_devices = self._query_devices()

        self._scheduler = threading.Thread(
            target=self._scheduler_loop,
            name="memex-scheduler",
            daemon=False,
        )
        self._scheduler.start()

    def submit(
        self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any
    ) -> Future[_T]:
        """Schedule ``fn(*args, **kwargs)`` and return an ordinary Future."""

        if not callable(fn):
            raise TypeError("fn must be callable")
        if self._inject_device:
            self._validate_device_parameter(fn)

        with self._lock:
            if self._shutdown_requested:
                raise RuntimeError("cannot schedule new futures after shutdown")
            future: Future[_T] = Future()
            task = _Task(
                identifier=self._next_identifier,
                future=future,
                fn=fn,
                args=tuple(args),
                kwargs=dict(kwargs),
            )
            self._next_identifier += 1
            self._pending.append(task)
            future.add_done_callback(lambda _: self._wake.set())
            _LOG.debug("task %d queued", task.identifier)
        self._wake.set()
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Stop accepting work and optionally cancel tasks that have not started."""

        with self._lock:
            self._shutdown_requested = True
            if cancel_futures:
                for task in self._pending:
                    task.future.cancel()
        self._wake.set()
        if wait and threading.current_thread() is not self._scheduler:
            self._scheduler.join()

    def stats(self) -> ExecutorStats:
        """Return an immutable snapshot of scheduler and device statistics."""

        devices = self._safe_devices()
        by_name = {device.name: device for device in devices}
        with self._lock:
            reservations: dict[str, int] = {}
            running_counts: dict[str, int] = {}
            for running in self._running.values():
                reservations[running.device] = (
                    reservations.get(running.device, 0) + running.reserved_mib
                )
                running_counts[running.device] = running_counts.get(running.device, 0) + 1
            queued = 0
            retrying = 0
            for task in self._pending:
                if task.future.cancelled():
                    continue
                if task.oom_retries:
                    retrying += 1
                else:
                    queued += 1
            names = list(by_name)
            for name in reservations:
                if name not in by_name:
                    names.append(name)
            device_stats = tuple(
                DeviceStats(
                    device=name,
                    total_mib=by_name.get(name, MemoryDevice(name, 0, 0)).total_mib,
                    available_mib=by_name.get(name, MemoryDevice(name, 0, 0)).available_mib,
                    reserved_mib=reservations.get(name, 0),
                    running=running_counts.get(name, 0),
                )
                for name in names
            )
            return ExecutorStats(
                backend=self._backend.name,
                running=len(self._running),
                queued=queued,
                retrying=retrying,
                completed=self._completed,
                failed=self._failed,
                cancelled=self._cancelled,
                observed_peak_mib=self._observed_peak_mib,
                headroom_mib=self._headroom,
                devices=device_stats,
            )

    def _validate_device_parameter(self, fn: Callable[..., Any]) -> None:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"cannot inspect submitted callable {fn!r}") from exc
        parameter = signature.parameters.get(self._device_parameter)
        accepts_kwargs = any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        )
        accepts_named_parameter = parameter is not None and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        if not accepts_named_parameter and not accepts_kwargs:
            raise TypeError(
                f"submitted callable must accept the {self._device_parameter!r} "
                "keyword argument or **kwargs"
            )

    def _query_devices(self) -> tuple[MemoryDevice, ...]:
        with self._backend_query_lock:
            devices = self._backend.devices()
        if not devices:
            raise RuntimeError(f"{self._backend.name} backend reported no devices")
        return devices

    def _safe_devices(self) -> tuple[MemoryDevice, ...]:
        if self._backend_closed:
            return self._last_devices
        try:
            devices = self._query_devices()
        except Exception:
            _LOG.warning("memory backend query failed", exc_info=True)
            return self._last_devices
        self._last_devices = devices
        return devices

    def _scheduler_loop(self) -> None:
        try:
            while True:
                self._wake.clear()
                self._sample_running_tasks()
                self._collect_finished_tasks()
                self._discard_cancelled_tasks()
                self._launch_tasks()
                with self._lock:
                    done = (
                        self._shutdown_requested
                        and not self._pending
                        and not self._running
                    )
                if done:
                    break
                self._wake.wait(self._poll_interval)
        except BaseException:
            _LOG.exception("scheduler failed unexpectedly")
            with self._lock:
                self._shutdown_requested = True
            self._fail_all_tasks(RuntimeError("Memex scheduler failed unexpectedly"))
        finally:
            with self._backend_query_lock:
                self._backend.close()
            self._backend_closed = True

    def _sample_running_tasks(self) -> None:
        with self._lock:
            running_tasks = tuple(self._running.values())
        for running in running_tasks:
            if running.process.pid is None:
                continue
            try:
                with self._backend_query_lock:
                    current = self._backend.process_memory_mib(
                        running.process.pid, running.device
                    )
            except Exception:
                _LOG.debug(
                    "could not sample task %d memory",
                    running.task.identifier,
                    exc_info=True,
                )
                continue
            if current <= running.peak_mib:
                continue
            with self._lock:
                running.peak_mib = current
                running.task.peak_mib = max(running.task.peak_mib, current)
                if current > running.reserved_mib:
                    running.reserved_mib = current
                if self._observed_peak_mib is None or current > self._observed_peak_mib:
                    previous = self._observed_peak_mib
                    self._observed_peak_mib = max(1, current)
                    # Every active attempt is now conservatively assumed capable
                    # of reaching the new session maximum. Also require at least
                    # one completion before adding more work after live growth.
                    for active in self._running.values():
                        active.reserved_mib = max(
                            active.reserved_mib, self._observed_peak_mib
                        )
                    if self._calibration_task is None:
                        self._growth_pause_at_running = len(self._running)
                    _LOG.info(
                        "memory estimate increased from %s to %d MiB",
                        previous,
                        self._observed_peak_mib,
                    )

    def _collect_finished_tasks(self) -> None:
        with self._lock:
            running_tasks = tuple(self._running.values())
        for running in running_tasks:
            payload: tuple[Any, ...] | None = None
            try:
                if running.connection.poll():
                    payload = running.connection.recv()
                elif running.process.is_alive():
                    continue
            except (EOFError, OSError):
                payload = None

            if payload is None and running.process.is_alive():
                continue
            # A result may become readable in the tiny interval between poll and exit.
            if payload is None:
                try:
                    if running.connection.poll():
                        payload = running.connection.recv()
                except (EOFError, OSError):
                    pass
            # Once a payload is readable, the worker has finished user code and
            # only closes its pipe before exiting. Reap it fully to avoid zombies.
            running.process.join()
            self._finish_attempt(running, payload)

    def _finish_attempt(
        self, running: _RunningTask, payload: tuple[Any, ...] | None
    ) -> None:
        task = running.task
        with self._lock:
            if self._running.pop(task.identifier, None) is None:
                return
        running.connection.close()

        if payload is not None and self._backend.name in {"cpu", "apple"}:
            final_peak = int(payload[-1])
            with self._lock:
                task.peak_mib = max(task.peak_mib, final_peak)
                if (
                    final_peak > 0
                    and (
                        self._observed_peak_mib is None
                        or final_peak > self._observed_peak_mib
                    )
                ):
                    self._observed_peak_mib = final_peak

        if payload is None:
            exception: BaseException = ProcessExitedError(
                running.process.pid, running.process.exitcode
            )
            likely_oom = running.process.exitcode in {-9, 9}
            self._handle_failure(task, exception, likely_oom)
        elif payload[0] == "result":
            task.future.set_result(payload[1])
            with self._lock:
                self._completed += 1
            _LOG.debug("task %d completed", task.identifier)
        elif payload[0] == "error":
            exception = payload[1]
            try:
                exception.add_note(f"Remote traceback:\n{payload[2]}")
            except (AttributeError, TypeError):
                pass
            self._handle_failure(task, exception, bool(payload[3]))
        else:
            exception = RemoteTaskError(payload[1], payload[2], payload[3])
            self._handle_failure(task, exception, bool(payload[4]))

        with self._lock:
            if self._calibration_task == task.identifier:
                self._calibration_task = None
        self._wake.set()

    def _handle_failure(
        self, task: _Task, exception: BaseException, likely_oom: bool
    ) -> None:
        if likely_oom and task.oom_retries < self._max_oom_retries:
            task.oom_retries += 1
            with self._lock:
                baseline = max(self._observed_peak_mib or 1, task.peak_mib, 1)
                increased = max(baseline + 1, math.ceil(baseline * 1.25))
                self._observed_peak_mib = increased
                self._pending.appendleft(task)
                self._oom_draining = True
            _LOG.warning(
                "task %d likely ran out of memory; retry %d/%d with %d MiB estimate",
                task.identifier,
                task.oom_retries,
                self._max_oom_retries,
                increased,
            )
            return

        task.future.set_exception(exception)
        with self._lock:
            self._failed += 1
        _LOG.debug("task %d failed: %r", task.identifier, exception)

    def _discard_cancelled_tasks(self) -> None:
        with self._lock:
            retained: deque[_Task] = deque()
            while self._pending:
                task = self._pending.popleft()
                if task.future.cancelled():
                    self._cancelled += 1
                    _LOG.debug("task %d cancelled", task.identifier)
                else:
                    retained.append(task)
            self._pending = retained

    def _launch_tasks(self) -> None:
        with self._lock:
            if not self._pending:
                return
            if self._calibration_task is not None:
                return
            if self._oom_draining:
                if self._running:
                    return
                self._oom_draining = False
            if self._growth_pause_at_running is not None:
                if len(self._running) >= self._growth_pause_at_running:
                    return
                self._growth_pause_at_running = None

        devices = self._safe_devices()
        while True:
            with self._lock:
                if not self._pending:
                    return
                task = self._pending[0]
                reservations: dict[str, int] = {}
                for running in self._running.values():
                    reservations[running.device] = (
                        reservations.get(running.device, 0) + running.reserved_mib
                    )
                estimate = self._observed_peak_mib
                unknown = estimate is None
                if unknown and self._running:
                    return

            choices: list[tuple[int, MemoryDevice]] = []
            for device in devices:
                effective = (
                    device.available_mib
                    - reservations.get(device.name, 0)
                    - self._headroom
                )
                if unknown:
                    if effective > 0:
                        choices.append((effective, device))
                elif effective >= estimate:
                    choices.append((effective, device))
            if not choices:
                return
            effective, selected = max(choices, key=lambda item: item[0])
            reservation = effective if unknown else max(1, int(estimate))

            with self._lock:
                if not self._pending or self._pending[0] is not task:
                    continue
                self._pending.popleft()
                if task.future.cancelled():
                    self._cancelled += 1
                    continue
                if not task.started:
                    if not task.future.set_running_or_notify_cancel():
                        self._cancelled += 1
                        continue
                    task.started = True
                if unknown:
                    self._calibration_task = task.identifier
            self._start_attempt(task, selected.name, reservation)
            if unknown:
                return
            # The device snapshot stays valid for this pass because every launch
            # is subtracted through reservations before the next choice.

    def _start_attempt(self, task: _Task, device: str, reservation: int) -> None:
        receive_connection, send_connection = self._context.Pipe(duplex=False)
        kwargs = dict(task.kwargs)
        if self._inject_device:
            kwargs[self._device_parameter] = device
        process = self._context.Process(
            target=run_task,
            args=(send_connection, task.fn, task.args, kwargs),
            name=f"memex-task-{task.identifier}-attempt-{task.oom_retries + 1}",
        )
        try:
            process.start()
        except BaseException as exc:
            receive_connection.close()
            send_connection.close()
            task.future.set_exception(exc)
            with self._lock:
                self._failed += 1
                if self._calibration_task == task.identifier:
                    self._calibration_task = None
            return
        send_connection.close()
        running = _RunningTask(
            task=task,
            device=device,
            process=process,
            connection=receive_connection,
            reserved_mib=reservation,
        )
        with self._lock:
            self._running[task.identifier] = running
        # Sample once immediately. Very short tasks may finish before the first
        # polling interval, but their spawned interpreter still has a meaningful
        # CPU/unified-memory footprint at this point.
        if process.pid is not None:
            try:
                with self._backend_query_lock:
                    initial = self._backend.process_memory_mib(process.pid, device)
            except Exception:
                initial = 0
            if initial > 0:
                with self._lock:
                    running.peak_mib = initial
                    task.peak_mib = max(task.peak_mib, initial)
                    running.reserved_mib = max(running.reserved_mib, initial)
                    if (
                        self._observed_peak_mib is None
                        or initial > self._observed_peak_mib
                    ):
                        self._observed_peak_mib = initial
        _LOG.debug(
            "task %d launched on %s with %d MiB reserved",
            task.identifier,
            device,
            reservation,
        )

    def _fail_all_tasks(self, exception: BaseException) -> None:
        with self._lock:
            pending = tuple(self._pending)
            running = tuple(self._running.values())
            self._pending.clear()
            self._running.clear()
        for task in pending:
            if not task.future.done():
                task.future.set_exception(exception)
                self._failed += 1
        for item in running:
            if item.process.is_alive():
                item.process.terminate()
            item.process.join(timeout=1)
            item.connection.close()
            if not item.task.future.done():
                item.task.future.set_exception(exception)
                self._failed += 1
