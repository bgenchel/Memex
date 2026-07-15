# NVIDIA CUDA hardware tests

These tests are opt-in. They use NVML in the parent and create real PyTorch CUDA
contexts in fresh spawned child processes.

## Setup

Use the same Python environment and CUDA-enabled PyTorch installation as the
workload you intend to run. Then install Memex and its NVIDIA/test dependencies:

```bash
python -m pip install -e '.[nvidia,test]'
```

Confirm that PyTorch can see CUDA before running Memex tests:

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

Memex does not modify `CUDA_VISIBLE_DEVICES`. If it is set to physical ordinals
or GPU UUIDs, Memex maps framework-logical names such as `cuda:0` to the matching
physical NVML devices. The suite checks only devices visible to PyTorch.

## Safe smoke suite

```bash
MEMEX_RUN_CUDA_TESTS=1 python -m pytest tests/hardware/test_nvidia_cuda.py -v -s
```

The allocation size defaults to 32 MiB and can be changed:

```bash
MEMEX_RUN_CUDA_TESTS=1 MEMEX_CUDA_TEST_MIB=128 \
  python -m pytest tests/hardware/test_nvidia_cuda.py -v -s
```

Although the tensor allocation is small, every spawned process also creates a
PyTorch CUDA context, which may consume several hundred MiB depending on the
driver and framework. Check `nvidia-smi` first when other training jobs are
using the machine. The suite skips allocation tests unless at least 512 MiB plus
the requested tensor size is available on one GPU.

## Concurrent scheduling test

The concurrency test is separately gated because it creates multiple CUDA
contexts at once:

```bash
MEMEX_RUN_CUDA_TESTS=1 MEMEX_RUN_CUDA_CONCURRENCY=1 \
  python -m pytest tests/hardware/test_nvidia_cuda.py -v -s
```

To run only discovery without allocating CUDA memory:

```bash
MEMEX_RUN_CUDA_TESTS=1 \
  python -m pytest tests/hardware/test_nvidia_cuda.py -v -k discovery
```

The suite deliberately does not force a genuine hardware OOM, since doing so
could disrupt colocated training. OOM classification/retry is verified with a
real `torch.cuda.OutOfMemoryError` raised once from a CUDA child process.
