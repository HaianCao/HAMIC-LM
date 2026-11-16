from pathlib import Path

import hamic.runtime.probe as probe
from hamic.checkpoint import LocalObjectStore
from hamic.runtime import matmul_tflops, probe_runtime, storage_throughput


def test_probe_reports_plain_strings_and_no_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_should_never_appear")
    report = probe_runtime()
    assert isinstance(report["torch"]["version"], str)
    assert "hf_should_never_appear" not in repr(report)
    assert report["hf_token_in_environment"] is True


def test_cgroup_v2_limits(monkeypatch):
    files = {
        "/sys/fs/cgroup/cpu.max": "400000 100000",
        "/sys/fs/cgroup/memory.max": str(32 * 2**30),
    }
    monkeypatch.setattr(probe, "_read", files.get)
    assert probe._cgroup_limits() == {"cpu_limit": 4.0, "memory_limit_gib": 32.0}


def test_cgroup_unlimited(monkeypatch):
    files = {"/sys/fs/cgroup/cpu.max": "max 100000", "/sys/fs/cgroup/memory.max": "max"}
    monkeypatch.setattr(probe, "_read", files.get)
    assert probe._cgroup_limits() == {"cpu_limit": None, "memory_limit_gib": None}


def test_implausible_disk_size_is_dropped(monkeypatch):
    class Usage:
        free = 2**63

    monkeypatch.setattr(probe.shutil, "disk_usage", lambda _: Usage())
    assert probe._disk_free_gib(".") is None


def test_matmul_benchmark_runs_on_cpu():
    result = matmul_tflops(size=64, dtype="float32", warmup=1, iterations=3, device="cpu")
    # A 64x64 matmul is ~1.6 MFLOP: on a busy CPU the rounded TFLOP/s can be 0.0.
    assert result["seconds"] >= 0
    assert result["tflops"] >= 0
    assert result["device"] == "cpu"


def test_storage_throughput_roundtrip(tmp_path: Path):
    store = LocalObjectStore(tmp_path / "store")
    result = storage_throughput(store, size_mib=2)
    assert result["intact"] is True
    assert result["upload_mib_s"] > 0
    assert store.list_files(result["remote_path"])


def test_storage_throughput_parallel_files(tmp_path: Path):
    store = LocalObjectStore(tmp_path / "store")
    result = storage_throughput(store, size_mib=4, files=4)
    assert result["intact"] is True
    assert result["files"] == 4
    assert len(store.list_files(result["remote_path"])) == 4
