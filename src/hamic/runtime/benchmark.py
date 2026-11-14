"""Micro-benchmarks for runtime readiness: GPU matmul and object-store throughput.

These numbers bound what a training run can achieve; they are not training throughput.
Capacity planning must use end-to-end training benchmarks.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from hamic.checkpoint.storage import ObjectStore
from hamic.provenance import file_sha256

__all__ = ["matmul_tflops", "storage_throughput"]

_MIB = float(2**20)


def matmul_tflops(
    size: int = 8192,
    dtype: str = "bfloat16",
    warmup: int = 5,
    iterations: int = 50,
    device: str | None = None,
) -> dict[str, Any]:
    """Measure dense square matmul throughput after warmup.

    Warmup iterations are excluded so cuBLAS heuristics, kernel loading and clock ramp-up
    do not dominate the result.

    Args:
        size: Matrix side length.
        dtype: Torch dtype name.
        warmup: Untimed iterations.
        iterations: Timed iterations.
        device: ``"cuda"`` or ``"cpu"``; defaults to CUDA when available.
    """
    import torch

    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = getattr(torch, dtype)
    a = torch.randn(size, size, device=target, dtype=torch_dtype)
    b = torch.randn(size, size, device=target, dtype=torch_dtype)
    for _ in range(warmup):
        torch.matmul(a, b)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
        start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start.record()
        for _ in range(iterations):
            torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize(target)
        seconds = start.elapsed_time(end) / 1000.0
    else:
        started = time.perf_counter()
        for _ in range(iterations):
            torch.matmul(a, b)
        seconds = time.perf_counter() - started
    flops = 2.0 * size**3 * iterations
    return {
        "device": str(target),
        "dtype": dtype,
        "size": size,
        "iterations": iterations,
        "seconds": round(seconds, 4),
        "tflops": round(flops / seconds / 1e12, 2),
    }


def storage_throughput(
    store: ObjectStore, size_mib: int = 256, prefix: str = "bench"
) -> dict[str, Any]:
    """Upload then download one incompressible file and verify it round-trips.

    The file is left at ``<prefix>/<uuid>.bin`` so the measurement is auditable; delete
    the prefix when it is no longer needed.

    Args:
        store: Object store to measure.
        size_mib: File size in MiB. Use at least a few hundred MiB: small files are
            dominated by per-request overhead and understate bulk throughput.
        prefix: Remote folder for benchmark objects.
    """
    remote = f"{prefix}/{uuid.uuid4().hex}.bin"
    with tempfile.TemporaryDirectory(prefix="hamic-bench-") as tmp:
        source = Path(tmp) / "upload.bin"
        with source.open("wb") as stream:
            for _ in range(size_mib):
                stream.write(os.urandom(2**20))
        expected = file_sha256(source)

        started = time.perf_counter()
        store.put_files({remote: source}, message=f"storage benchmark {size_mib} MiB")
        upload_seconds = time.perf_counter() - started

        target = Path(tmp) / "download.bin"
        started = time.perf_counter()
        store.download(remote, target)
        download_seconds = time.perf_counter() - started
        intact = file_sha256(target) == expected

    return {
        "remote_path": remote,
        "size_mib": size_mib,
        "upload_seconds": round(upload_seconds, 2),
        "download_seconds": round(download_seconds, 2),
        "upload_mib_s": round(size_mib / upload_seconds, 2),
        "download_mib_s": round(size_mib / download_seconds, 2),
        "intact": intact,
    }
