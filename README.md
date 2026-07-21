# Memex

Memex is a `concurrent.futures.Executor` implementation that schedules each task
in a fresh spawned process and learns how much memory tasks need. It selects the
CPU, Apple MPS, AMD ROCm, or least-loaded NVIDIA CUDA device automatically,
preserves configured memory headroom, and retries likely out-of-memory failures.

```python
from concurrent.futures import as_completed
from memex import MemoryExecutor


def job(*args, **kwargs, device):
    # Load and run the model on device, then write output_path deterministically.
    ...


if __name__ == "__main__": 
    tasks = [(*args1), (*args2)]
    with MemoryExecutor(headroom=2048) as executor:
        futures = [executor.submit(job, *task) for task in tasks]
        for future in as_completed(futures):
            future.result()
```

`headroom` and all reported memory values are in MiB. Its default is 2048 MiB,
and that amount is always excluded from schedulable memory on every device.
Memex starts one task, doubles its concurrency limit after each successful
warm-up stage, and learns task memory from all successfully completed peaks
using the mean plus two sample standard deviations. Likely out-of-memory tasks
are retried with a task-local estimate increased by 1024 MiB; a retry does not
inflate the estimate used for unrelated work.

Explicit backends are
selected with `backend="cpu"`, `"cuda"`, `"amd"`, or `"apple"`; `"auto"`
prefers an available accelerator. NVIDIA support requires the optional
`nvidia-ml-py` dependency.

Tasks must be picklable under Python's `spawn` rules. With device injection
enabled (the default), the callable must accept the configured keyword or
`**kwargs`. Use `executor.stats()` for an immutable `ExecutorStats` snapshot.

## NVIDIA hardware verification

An opt-in test suite for real NVIDIA machines is included under
`tests/hardware`. It exercises NVML discovery, real PyTorch CUDA allocation,
per-process peak learning, injected device names, process isolation, OOM retry,
and—behind a second opt-in flag—concurrent CUDA scheduling.

See [`tests/hardware/README.md`](tests/hardware/README.md) for installation,
safety notes, and commands. The normal test suite never allocates CUDA memory.
