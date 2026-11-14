import json
from pathlib import Path

import pytest

import hamic.provenance as provenance
import hamic.training.smoke as smoke
from hamic.checkpoint import LocalObjectStore
from hamic.training import (
    ResumeCompatibilityError,
    SmokeConfig,
    check_strict_resume,
    run_smoke,
)


class _FakeDistribution:
    def __init__(self, payload: dict):
        self._payload = payload

    def read_text(self, name: str):
        return json.dumps(self._payload) if name == "direct_url.json" else None


def test_installed_revision_reads_pip_record(monkeypatch):
    payload = {
        "url": "https://github.com/HaianCao/HAMIC-LM",
        "vcs_info": {"vcs": "git", "commit_id": "abc123" * 6, "requested_revision": "main"},
    }
    monkeypatch.setattr(provenance.metadata, "distribution", lambda _: _FakeDistribution(payload))
    revision = provenance.installed_revision()
    assert revision["commit"] == "abc123" * 6
    assert revision["requested_revision"] == "main"


def test_code_revision_falls_back_to_pip_without_git(monkeypatch):
    monkeypatch.setattr(provenance, "git_revision", lambda _cwd: {"commit": None, "dirty": None})
    monkeypatch.setattr(
        provenance,
        "installed_revision",
        lambda: {"commit": "f" * 40, "requested_revision": "main", "url": "x"},
    )
    assert provenance.code_revision() == {
        "commit": "f" * 40,
        "dirty": False,
        "source": "pip",
        "requested_revision": "main",
    }


def test_code_revision_in_this_checkout_comes_from_git():
    revision = provenance.code_revision()
    assert revision["source"] in {"git", "pip", "unknown"}


def _meta(commit, config="cfg"):
    return {"config_sha256": config, "code_revision": {"commit": commit}}


def test_strict_resume_accepts_same_code_and_config():
    assert check_strict_resume(_meta("a" * 40), _meta("a" * 40)) == []


def test_strict_resume_warns_when_commit_unknown():
    assert check_strict_resume(_meta(None), _meta("a" * 40))


@pytest.mark.parametrize(
    ("saved", "running"),
    [(_meta("a" * 40), _meta("b" * 40)), (_meta("a" * 40, "x"), _meta("a" * 40, "y"))],
)
def test_strict_resume_rejects_changes(saved, running):
    with pytest.raises(ResumeCompatibilityError):
        check_strict_resume(saved, running)


def test_drill_refuses_resume_after_code_change(tmp_path: Path, monkeypatch):
    store = LocalObjectStore(tmp_path / "store")
    config = SmokeConfig(steps=6, checkpoint_every=2)
    monkeypatch.setattr(smoke, "code_revision", lambda: {"commit": "a" * 40})
    run_smoke(store, tmp_path / "w1", config, stop_after_step=3)
    monkeypatch.setattr(smoke, "code_revision", lambda: {"commit": "b" * 40})
    with pytest.raises(ResumeCompatibilityError, match="code changed"):
        run_smoke(store, tmp_path / "w2", config)
