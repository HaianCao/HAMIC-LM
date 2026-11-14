import json
from pathlib import Path

import pytest

from hamic.checkpoint import (
    CheckpointError,
    CheckpointPublisher,
    CheckpointResolver,
    LocalObjectStore,
    build_manifest,
    verify_local,
)
from hamic.runtime import SessionClock
from hamic.training import SmokeConfig, run_smoke


def _payload(directory: Path, content: bytes) -> Path:
    directory.mkdir(parents=True)
    (directory / "weights.bin").write_bytes(content)
    (directory / "sub").mkdir()
    (directory / "sub" / "state.json").write_text("{}", encoding="utf-8")
    return directory


def test_manifest_detects_tampering(tmp_path: Path):
    ckpt = _payload(tmp_path / "c", b"abc")
    build_manifest(ckpt, {"global_step": 1})
    with pytest.raises(CheckpointError, match="incomplete"):
        verify_local(ckpt)  # no COMMITTED.json yet


def test_publish_resolve_and_fallback(tmp_path: Path):
    store = LocalObjectStore(tmp_path / "store")
    publisher = CheckpointPublisher(store, "runs/r1")
    publisher.publish(_payload(tmp_path / "s1", b"one"), {"global_step": 1})
    publisher.publish(_payload(tmp_path / "s2", b"two"), {"global_step": 2})

    resolver = CheckpointResolver(store, "runs/r1")
    assert [c.step for c in resolver.committed()] == [1, 2]
    latest = json.loads(store.read_bytes("runs/r1/LATEST.json") or b"{}")
    assert latest["global_step"] == 2

    # Corrupt the newest generation: resolution must fall back to step 1.
    (tmp_path / "store/runs/r1/checkpoints/step-000000002/weights.bin").write_bytes(b"bad")
    restored = resolver.restore_latest(tmp_path / "restore")
    assert restored is not None
    assert restored.step == 1
    assert (tmp_path / "restore" / "weights.bin").read_bytes() == b"one"


def test_uncommitted_prefix_is_ignored(tmp_path: Path):
    store = LocalObjectStore(tmp_path / "store")
    store.put_bytes("runs/r/checkpoints/step-000000005/manifest.json", b"{}", "partial")
    assert CheckpointResolver(store, "runs/r").committed() == []


def test_committed_checkpoint_is_immutable(tmp_path: Path):
    store = LocalObjectStore(tmp_path / "store")
    publisher = CheckpointPublisher(store, "runs/r")
    publisher.publish(_payload(tmp_path / "a", b"x"), {"global_step": 3})
    with pytest.raises(CheckpointError, match="immutable"):
        publisher.publish(_payload(tmp_path / "b", b"y"), {"global_step": 3})


@pytest.mark.parametrize("max_shard_mib", [1024.0, 0.05], ids=["one-shard", "many-shards"])
def test_strict_resume_reproduces_uninterrupted_run(tmp_path: Path, max_shard_mib: float):
    config = SmokeConfig(steps=8, checkpoint_every=3, max_shard_mib=max_shard_mib)
    reference = run_smoke(LocalObjectStore(tmp_path / "a"), tmp_path / "wa", config)

    store = LocalObjectStore(tmp_path / "b")
    first = run_smoke(store, tmp_path / "wb1", config, stop_after_step=5)
    assert first.published_steps == [3]
    resumed = run_smoke(store, tmp_path / "wb2", config)
    assert resumed.resumed_from_step == 3
    assert resumed.losses == {k: reference.losses[k] for k in resumed.losses}
    assert resumed.final_step == 8


def test_session_clock_reserves_checkpoint_margin():
    now = [0.0]
    clock = SessionClock(clock=lambda: now[0])
    now[0] = 10 * 3600
    assert not clock.should_stop()
    clock.record_checkpoint(1.5 * 3600)  # slow uploads shrink the window to 9h
    assert clock.should_stop()
    assert clock.remaining == 0.0


def test_hub_downloads_land_directly_without_extra_copy(tmp_path: Path, monkeypatch):
    from unittest import mock

    from hamic.checkpoint import HuggingFaceObjectStore

    def fake_file(repo_id, filename, *, repo_type, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"payload")
        return str(target)

    def fake_snapshot(repo_id, *, repo_type, allow_patterns, local_dir, max_workers):
        assert allow_patterns == ["runs/r/checkpoints/step-000000001/*"]
        for name in ("weights.bin", "manifest.json"):
            fake_file(
                repo_id,
                f"runs/r/checkpoints/step-000000001/{name}",
                repo_type=repo_type,
                local_dir=local_dir,
            )
        return str(local_dir)

    with mock.patch("huggingface_hub.HfApi") as api:
        api.return_value.repo_exists.return_value = True
        api.return_value.hf_hub_download.side_effect = fake_file
        api.return_value.snapshot_download.side_effect = fake_snapshot
        store = HuggingFaceObjectStore("ns/runs", token="t")

        single = store.download("a/b.bin", tmp_path / "out" / "b.bin")
        assert single.read_bytes() == b"payload"
        tree = store.download_prefix("runs/r/checkpoints/step-000000001", tmp_path / "ckpt")
        assert sorted(p.name for p in tree.iterdir()) == ["manifest.json", "weights.bin"]

    leftovers = [p.name for p in tmp_path.rglob(".*")]
    assert leftovers == []  # staging directories are removed


def test_sharded_state_roundtrip_and_tied_weights(tmp_path: Path):
    import json

    import torch

    from hamic.modeling import Architecture, build_model
    from hamic.training import DataCursor, load_training_state, save_training_state
    from tests.conftest import tiny_preset

    preset = tiny_preset(Architecture.QWEN3, vocab_size=300)
    model = build_model(preset, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(5, 300, (2, 8))
    model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    root = save_training_state(
        tmp_path / "ckpt", model, optimizer, None, DataCursor(offset=7), max_shard_mib=0.02
    )
    index = json.loads((root / "checkpoint_index.json").read_text(encoding="utf-8"))
    assert len(index["weights"]["files"]) > 1
    assert len(index["optimizer"]["files"]) > 1
    assert index["weights"]["aliases"] == {"lm_head.weight": "model.embed_tokens.weight"}

    fresh = build_model(preset, seed=1)
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    cursor = load_training_state(root, fresh, fresh_opt, None)
    assert cursor.offset == 7
    for (name, a), b in zip(model.state_dict().items(), fresh.state_dict().values(), strict=True):
        assert torch.equal(a, b), name
    assert fresh.get_input_embeddings().weight is fresh.get_output_embeddings().weight
    saved, loaded = optimizer.state_dict()["state"], fresh_opt.state_dict()["state"]
    assert all(torch.equal(saved[k]["exp_avg"], loaded[k]["exp_avg"]) for k in saved)


def test_legacy_single_file_checkpoint_still_loads(tmp_path: Path):
    import json

    import torch
    from safetensors.torch import save_model

    from hamic.modeling import Architecture, build_model
    from hamic.training import capture_rng, load_training_state
    from tests.conftest import tiny_preset

    preset = tiny_preset(Architecture.LLAMA, vocab_size=300)
    model = build_model(preset, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    root = tmp_path / "legacy"
    root.mkdir()
    save_model(model, str(root / "model.safetensors"))
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": None, "rng": capture_rng()},
        root / "trainer_state.pt",
    )
    (root / "data_cursor.json").write_text(json.dumps({"offset": 3}), encoding="utf-8")
    fresh = build_model(preset, seed=1)
    cursor = load_training_state(root, fresh, torch.optim.AdamW(fresh.parameters()), None)
    assert cursor.offset == 3
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), fresh.parameters(), strict=True)
    )
