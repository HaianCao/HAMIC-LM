"""Secret-safe runtime probe (GPU, CUDA, memory, disk, package versions)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hamic.provenance import environment_fingerprint, utc_now

__all__ = ["probe_runtime"]

_GIB = float(2**30)


def _host_memory_gib() -> float | None:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / (1024**2), 2)
        sysconf = getattr(os, "sysconf", None)
        if sysconf is None:  # Windows
            return None
        return round(float(sysconf("SC_PAGE_SIZE")) * float(sysconf("SC_PHYS_PAGES")) / _GIB, 2)
    except (OSError, ValueError, AttributeError):
        return None


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False}
    query = "--query-gpu=index,name,driver_version,memory.total,memory.free"
    try:
        done = subprocess.run(
            [executable, query, "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": True, "error": type(exc).__name__}
    return {"available": True, "exit_code": done.returncode, "gpus": done.stdout.splitlines()}


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / _GIB, 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        info["devices"] = devices
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["flash_sdp_enabled"] = bool(torch.backends.cuda.flash_sdp_enabled())  # type: ignore[no-untyped-call]
    return info


def probe_runtime(scratch: str | Path = ".") -> dict[str, Any]:
    """Return a JSON-friendly description of the runtime.

    Only names, versions and sizes are reported: no environment variables, tokens or
    absolute user paths.
    """
    disk = shutil.disk_usage(scratch)
    return {
        "probed_at_utc": utc_now(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "host_memory_gib": _host_memory_gib(),
            "scratch_free_gib": round(disk.free / _GIB, 2),
        },
        "nvidia_smi": _nvidia_smi(),
        "torch": _torch_info(),
        "environment": environment_fingerprint(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    }
