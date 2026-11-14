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
    store: ObjectStore, size_mib: int = 256, files: int = 1, prefix: str = "bench"
) -> dict[str, Any]:
    """Upload then download incompressible data and verify it round-trips.

    With ``files > 1`` the data is split into equal files uploaded in one commit and
    downloaded with :meth:`ObjectStore.download_prefix`, which is how sharded checkpoints
    are restored; compare it with ``files=1`` to see whether parallel transfer helps.
    Objects are left under ``<prefix>/<uuid>/`` so the measurement is auditable.

    Args:
        store: Object store to measure.
        size_mib: Total size in MiB. Use at least a few hundred MiB: small transfers are
            dominated by per-request overhead and understate bulk throughput.
        files: Number of files the data is split into.
        prefix: Remote folder for benchmark objects.
    """
    if files < 1 or size_mib < files:
        raise ValueError("Require 1 <= files <= size_mib")
    remote_dir = f"{prefix}/{uuid.uuid4().hex}"
    per_file = size_mib // files
    with tempfile.TemporaryDirectory(prefix="hamic-bench-") as tmp:
        upload_dir = Path(tmp) / "upload"
        upload_dir.mkdir()
        expected: dict[str, str] = {}
        for index in range(files):
            source = upload_dir / f"part-{index:03d}.bin"
            with source.open("wb") as stream:
                for _ in range(per_file):
                    stream.write(os.urandom(2**20))
            expected[source.name] = file_sha256(source)
        mapping = {f"{remote_dir}/{name}": upload_dir / name for name in expected}

        started = time.perf_counter()
        store.put_files(mapping, message=f"storage benchmark {files}x{per_file} MiB")
        upload_seconds = time.perf_counter() - started

        target = Path(tmp) / "download"
        started = time.perf_counter()
        if files == 1:
            name = next(iter(expected))
            target.mkdir()
            store.download(f"{remote_dir}/{name}", target / name)
        else:
            store.download_prefix(remote_dir, target)
        download_seconds = time.perf_counter() - started
        intact = all(file_sha256(target / name) == sha for name, sha in expected.items())

    total = per_file * files
    return {
        "remote_path": remote_dir,
        "files": files,
        "size_mib": total,
        "upload_seconds": round(upload_seconds, 2),
        "download_seconds": round(download_seconds, 2),
        "upload_mib_s": round(total / upload_seconds, 2),
        "download_mib_s": round(total / download_seconds, 2),
        "intact": intact,
    }
