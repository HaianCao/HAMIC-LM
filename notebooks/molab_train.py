"""Molab pretraining: run (or resume) one HAMIC run configuration on the GPU.

One session trains until the run's token budget is reached or the session's soft deadline
(10.5 h), publishing checkpoints to a private Hugging Face model repository. Opening the
notebook again with the same configuration resumes from the newest committed checkpoint.

Before the first session, upload the tokenized shards from a workstation::

    hamic corpus push --tokenized data/tokenized/pilot-v0.1/t1-bbpe-20544 \\
        --store hf-dataset:<namespace>/hamic-data

Paste a fine-grained Hugging Face token (write access to ``HUB_REPO``, read access to
``DATA_REPO``) into the password field. It stays in kernel memory and is never written to
the notebook source; an ``HF_TOKEN`` environment variable is used when the field is empty.
Never type the token into a code cell: Molab notebooks are viewable by anyone with the link.
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # HAMIC · Molab pretraining
        Check the settings in the next cell, run the cells in order, pick a run
        configuration and press **Train**. After pushing new code, restart the kernel.
        """
    )
    return (mo,)


@app.cell
def _():
    # Any Git ref works after "@": a branch (main), a tag or a commit SHA. Pin a SHA for
    # a multi-session run so every session uses the same code (resume checks it).
    HAMIC_SOURCE = "git+https://github.com/HaianCao/HAMIC-LM@main"
    HUB_REPO = "HaianCao/hamic-runs"  # private model repo: checkpoints under runs/<run_id>/
    DATA_REPO = "HaianCao/hamic-data"  # private dataset repo: tokenized shards
    XET_HIGH_PERFORMANCE = True  # read at first huggingface_hub import; restart to change
    return DATA_REPO, HAMIC_SOURCE, HUB_REPO, XET_HIGH_PERFORMANCE


@app.cell
def _(HAMIC_SOURCE, XET_HIGH_PERFORMANCE, mo):
    import json
    import os as _os
    import subprocess
    import sys
    from importlib import metadata

    _os.environ["HF_XET_HIGH_PERFORMANCE"] = "1" if XET_HIGH_PERFORMANCE else "0"
    requested_ref = HAMIC_SOURCE.rsplit("@", 1)[-1]
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run([*pip, HAMIC_SOURCE], check=True)  # dependencies
    # Every dev build has the same version number, so pip would otherwise keep an
    # older commit that is already installed.
    subprocess.run([*pip, "--force-reinstall", "--no-deps", HAMIC_SOURCE], check=True)
    direct_url = json.loads(metadata.distribution("hamic").read_text("direct_url.json") or "{}")
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id", "")
    if not installed_commit:
        raise RuntimeError("pip did not record the installed commit (direct_url.json)")
    is_hex = all(c in "0123456789abcdef" for c in requested_ref)
    if len(requested_ref) >= 7 and is_hex and not installed_commit.startswith(requested_ref):
        raise RuntimeError(f"Installed commit {installed_commit!r}, expected {requested_ref!r}")
    mo.stop(
        "hamic" in sys.modules,
        mo.md(
            f"Installed `hamic` at `{installed_commit[:12]}`, but an older copy is already "
            "imported in this kernel. **Restart the kernel**, then run the notebook again."
        ),
    )
    hamic_ready = installed_commit
    mo.md(f"`hamic` installed from `@{requested_ref}` at commit `{installed_commit[:12]}`")
    return (hamic_ready,)


@app.cell
def _(HAMIC_SOURCE, hamic_ready, mo):
    import os
    import shutil
    import tarfile
    import tempfile
    import urllib.request
    from pathlib import Path

    # Run configurations reference presets by repository-relative paths, so fetch the
    # configs/ tree of exactly the installed commit and work from that directory.
    work_root = Path(tempfile.gettempdir()) / "hamic-work"
    configs_dir = work_root / "configs"
    stamp = work_root / "configs.commit"
    if not stamp.is_file() or stamp.read_text().strip() != hamic_ready:
        repo_url = HAMIC_SOURCE.removeprefix("git+").rsplit("@", 1)[0].removesuffix(".git")
        archive = work_root / "source.tar.gz"
        work_root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{repo_url}/archive/{hamic_ready}.tar.gz", archive)
        shutil.rmtree(configs_dir, ignore_errors=True)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if "/configs/" in f"/{m.name}"]
            tar.extractall(work_root / "src", members=members, filter="data")
        (top,) = (work_root / "src").iterdir()
        shutil.move(str(top / "configs"), configs_dir)
        shutil.rmtree(work_root / "src")
        archive.unlink()
        stamp.write_text(hamic_ready)
    os.chdir(work_root)
    run_configs = sorted(
        p.relative_to(work_root).as_posix() for p in (configs_dir / "train").rglob("*.yaml")
    )
    mo.md(f"Working directory `{work_root}` · {len(run_configs)} run configurations")
    return Path, os, run_configs, work_root


@app.cell
def _(hamic_ready, mo):
    from hamic.runtime import probe_runtime

    del hamic_ready  # dependency only: run after the install cell
    probe = probe_runtime("/tmp")
    if not probe["torch"].get("cuda_available"):
        mo.stop(True, mo.md("**No GPU attached.** Attach it from the notebook specs panel."))
    gpu = probe["torch"]["devices"][0]
    mo.md(f"GPU **{gpu['name']}** · CUDA `{probe['torch'].get('cuda_version')}`")
    return (probe,)


@app.cell
def _(mo):
    token_input = mo.ui.text(
        kind="password",
        label="Hugging Face token (write: HUB_REPO, read: DATA_REPO)",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("The token stays in this session's memory and is not saved in the notebook."),
            token_input,
        ]
    )
    return (token_input,)


@app.cell
def _(DATA_REPO, HUB_REPO, mo, os, token_input):
    from hamic.checkpoint import HuggingFaceObjectStore

    hub_token = token_input.value.strip() or os.environ.get("HF_TOKEN", "")
    mo.stop(not hub_token, mo.md("**Enter a Hugging Face token above** to continue."))
    run_store = HuggingFaceObjectStore(HUB_REPO, private=True, token=hub_token)
    data_store = HuggingFaceObjectStore(
        DATA_REPO, repo_type="dataset", create=False, token=hub_token
    )
    account = run_store.whoami()
    mo.md(f"Hub account **{account}** · runs `{HUB_REPO}` · data `{DATA_REPO}`")
    return data_store, run_store


@app.cell
def _(mo, run_configs):
    default = next((c for c in run_configs if "proxy" in c), run_configs[0])
    config_choice = mo.ui.dropdown(options=run_configs, value=default, label="Run configuration")
    mode = mo.ui.radio(
        options={
            "Throughput test (40 steps, nothing saved)": "bench",
            "Train / resume": "train",
        },
        value="Throughput test (40 steps, nothing saved)",
        label="Mode",
    )
    mo.vstack([config_choice, mode])
    return config_choice, mode


@app.cell
def _(config_choice, mo):
    from hamic.modeling import ModelPreset, parameter_report
    from hamic.training import PretrainConfig

    config = PretrainConfig.from_yaml(config_choice.value)
    preset = ModelPreset.from_yaml(config.model_preset)
    params = parameter_report(preset)
    summary = {
        "run_id": config.run_id,
        "model": f"{preset.name} ({preset.architecture.value})",
        "parameters": f"{params.total / 1e6:.1f}M",
        "objective": config.objective
        + (f" (mask {config.mask_rate:.0%})" if config.objective == "mlm" else ""),
        "tokens": f"{config.total_tokens / 1e9:.3f}B",
        "steps": config.total_steps,
        "tokens_per_step": config.tokens_per_step,
        "tokenized": config.tokenized,
    }
    mo.ui.table([summary], selection=None)
    return config, params


@app.cell
def _(Path, config, data_store, mo):
    from hamic.data import fetch_tokenized

    local_tokens = Path(config.tokenized)
    prefix = f"tokenized/{local_tokens.parent.name}/{local_tokens.name}"
    with mo.status.spinner(f"Fetching and verifying {prefix} ..."):
        corpus = fetch_tokenized(data_store, prefix, local_tokens)
    split_tokens = {k: v["tokens"] for k, v in corpus.manifest["splits"].items()}
    epochs = config.total_tokens / split_tokens["train"]
    mo.md(f"Shards ready: `{split_tokens}` · the budget is **{epochs:.2f} epochs** of train")
    return (corpus,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train")
    train_button
    return (train_button,)


@app.cell
def _(config, corpus, mode, mo, params, run_store, train_button, work_root):
    import dataclasses
    import json as _json

    from hamic.checkpoint import LocalObjectStore
    from hamic.training import Pretrainer

    del corpus  # dependency only: shards are local
    mo.stop(not train_button.value, mo.md("Press **Train** to start."))
    records = []

    def log(record):
        records.append(record)
        print(_json.dumps(record), flush=True)

    if mode.value == "bench":
        run_config = dataclasses.replace(
            config,
            run_id=f"{config.run_id}-bench",
            eval_every_steps=0,
            checkpoint_every_steps=0,
            checkpoint_every_minutes=0,
        )
        store = LocalObjectStore(work_root / "bench-store")
        result = Pretrainer(run_config, store, work_root / "bench", log=log).run(
            stop_after_steps=40
        )
    else:
        result = Pretrainer(config, run_store, work_root / "run", log=log).run()

    steady = [r["tokens_per_second"] for r in records[len(records) // 2 :]]
    tokens_per_second = sum(steady) / max(1, len(steady))
    flops = 6 * params.total * tokens_per_second
    hours_left = (config.total_steps - result.final_step) * config.tokens_per_step
    hours_left /= max(tokens_per_second, 1.0) * 3600
    mo.md(
        f"Status **{result.status}** · step `{result.final_step}/{config.total_steps}` · "
        f"resumed from `{result.resumed_from_step}` · published `{result.published_steps}`\n\n"
        f"Throughput **{tokens_per_second:,.0f} tokens/s** (~{flops / 1e12:.0f} TFLOP/s "
        f"by 6·N·D) · about **{hours_left:.1f} h** of training left"
        + "".join(f"\n\n> Warning: {w}" for w in result.warnings)
    )
    return (result,)


@app.cell
def _(mo, result):
    rows = [
        {
            "step": step,
            "eval_loss": loss,
            "eval_bits_per_byte": result.eval_bits_per_byte.get(step),
        }
        for step, loss in sorted(result.evaluations.items())
    ]
    mo.ui.table(rows, selection=None) if rows else mo.md("No evaluations in this session.")
    return


if __name__ == "__main__":
    app.run()
