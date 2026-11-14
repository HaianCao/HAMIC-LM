"""Molab readiness drill: GPU probe and cross-session strict resume via the Hub.

Run in two Molab sessions:

1. Session 1, phase "train": probe the GPU, then train the smoke model and stop at step 6
   (checkpoints are published to a private Hugging Face repository).
2. Restart the kernel or open a new session, choose phase "resume": the run restores from
   the Hub only and must continue with losses identical to an uninterrupted reference.

Paste a fine-grained Hugging Face token (write access to ``HUB_REPO`` only) into the
password field. It stays in kernel memory and is never written to the notebook source;
an ``HF_TOKEN`` environment variable, if present, is used when the field is empty.
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
        # HAMIC · Molab readiness drill
        Edit `HAMIC_SOURCE` and `HUB_REPO` in the next cell, then run the cells in order.
        """
    )
    return (mo,)


@app.cell
def _():
    # Pin an exact commit; an unpinned branch makes the run irreproducible.
    HAMIC_SOURCE = "git+https://github.com/HaianCao/HAMIC-LM@<commit-sha>"
    HUB_REPO = "<hf-namespace>/hamic-runs"
    RUN_ID = "drill-001"
    return HAMIC_SOURCE, HUB_REPO, RUN_ID


@app.cell
def _(HAMIC_SOURCE):
    import subprocess
    import sys

    if "<" in HAMIC_SOURCE:
        raise ValueError("Set HAMIC_SOURCE to a pinned commit before installing")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", HAMIC_SOURCE], check=True)
    return


@app.cell
def _(mo):
    from hamic.runtime import probe_runtime

    probe = probe_runtime("/tmp")
    if not probe["torch"].get("cuda_available"):
        mo.stop(True, mo.md("**No GPU attached.** Attach it from the notebook specs panel."))
    mo.json(probe)
    return (probe,)


@app.cell
def _(mo, probe):
    from hamic.runtime import matmul_tflops

    matmul = [matmul_tflops(size=n, dtype="bfloat16") for n in (4096, 8192)]
    mo.vstack(
        [
            mo.md(
                f"BF16 matmul on **{probe['torch']['devices'][0]['name']}** after warmup "
                "(micro-benchmark, not training throughput):"
            ),
            mo.ui.table(matmul, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    token_input = mo.ui.text(
        kind="password",
        label="Hugging Face token (fine-grained, write access to HUB_REPO only)",
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
def _(HUB_REPO, mo, token_input):
    import os

    from hamic.checkpoint import HuggingFaceObjectStore

    hub_token = token_input.value.strip() or os.environ.get("HF_TOKEN", "")
    mo.stop(not hub_token, mo.md("**Enter a Hugging Face token above** to continue."))
    if "<" in HUB_REPO:
        raise ValueError("Set HUB_REPO to your private Hugging Face repository first")
    store = HuggingFaceObjectStore(HUB_REPO, private=True, token=hub_token)
    account = store.whoami()
    phase = mo.ui.radio(
        options=["train", "resume"], value="train", label="Drill phase (session 1 / session 2)"
    )
    mo.vstack([mo.md(f"Hub account **{account}** · run store `{HUB_REPO}`"), phase])
    return phase, store


@app.cell
def _(mo):
    bench_size = mo.ui.dropdown(
        options={"256 MiB": 256, "1 GiB": 1024, "2 GiB": 2048},
        value="1 GiB",
        label="Test file size",
    )
    bench_button = mo.ui.run_button(label="Measure Hub upload/download throughput")
    mo.vstack(
        [
            mo.md(
                "Checkpoint upload speed decides how often a long run can checkpoint. "
                "The test file stays under `bench/` in the run repository."
            ),
            mo.hstack([bench_size, bench_button], justify="start"),
        ]
    )
    return bench_button, bench_size


@app.cell
def _(bench_button, bench_size, mo, store):
    from hamic.runtime import storage_throughput

    mo.stop(not bench_button.value)
    throughput = storage_throughput(store, size_mib=bench_size.value)
    mo.json(throughput)
    return


@app.cell
def _(RUN_ID, mo, phase, store):
    import tempfile
    from pathlib import Path

    from hamic.training import SmokeConfig, run_smoke

    workdir = Path(tempfile.mkdtemp(prefix="hamic-drill-"))
    config = SmokeConfig(run_id=RUN_ID, steps=12, checkpoint_every=4)
    stop_after = 6 if phase.value == "train" else None
    result = run_smoke(store, workdir / "run", config, stop_after_step=stop_after)
    mo.md(
        f"Phase **{phase.value}** · resumed from step `{result.resumed_from_step}` · "
        f"published steps `{result.published_steps}` · final step `{result.final_step}`"
    )
    return config, result, run_smoke, workdir


@app.cell
def _(config, mo, phase, result, run_smoke, workdir):
    from hamic.checkpoint import LocalObjectStore

    mo.stop(phase.value != "resume")
    reference = run_smoke(LocalObjectStore(workdir / "reference-store"), workdir / "ref", config)
    identical = all(reference.losses[k] == v for k, v in result.losses.items())
    mo.md(
        "### Drill passed: strict resume reproduces the uninterrupted run"
        if identical and result.resumed_from_step == 4
        else "### Drill FAILED: compare the losses below"
    )
    return


if __name__ == "__main__":
    app.run()
