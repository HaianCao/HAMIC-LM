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
# Sandboxes such as gVisor can report absurd filesystem sizes (exabytes).
_IMPLAUSIBLE_DISK_GIB = 1_000_000.0


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cgroup_limits() -> dict[str, float | None]:
    """Container CPU/memory limits (cgroup v2, then v1); ``None`` when unlimited/unknown.

    Inside a container ``os.cpu_count()`` and ``/proc/meminfo`` may describe the host,
    so these limits are what a workload can actually use.
    """
    cpus: float | None = None
    memory: float | None = None
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        quota, _, period = cpu_max.partition(" ")
        if quota != "max" and period:
            cpus = round(int(quota) / int(period), 2)
    else:
        quota_v1 = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_v1 = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_v1 and period_v1 and int(quota_v1) > 0:
            cpus = round(int(quota_v1) / int(period_v1), 2)
    mem_raw = _read("/sys/fs/cgroup/memory.max") or _read(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    if mem_raw and mem_raw != "max" and int(mem_raw) < 2**60:
        memory = round(int(mem_raw) / _GIB, 2)
    return {"cpu_limit": cpus, "memory_limit_gib": memory}


def _usable_cpus() -> int | None:
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity is not None else os.cpu_count()


def _disk_free_gib(path: str | Path) -> float | None:
    free = shutil.disk_usage(path).free / _GIB
    return round(free, 2) if free < _IMPLAUSIBLE_DISK_GIB else None


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
        "version": str(torch.__version__),
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
    return {
        "probed_at_utc": utc_now(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "usable_cpus": _usable_cpus(),
            "host_memory_gib": _host_memory_gib(),
            **_cgroup_limits(),
            "scratch_free_gib": _disk_free_gib(scratch),
        },
        "nvidia_smi": _nvidia_smi(),
        "torch": _torch_info(),
        "environment": environment_fingerprint(),
        "hf_token_in_environment": bool(os.environ.get("HF_TOKEN")),
    }
