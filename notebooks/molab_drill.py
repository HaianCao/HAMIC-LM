"""Molab readiness drill: GPU probe and cross-session strict resume via the Hub.

Run in two Molab sessions:

1. Session 1, phase "train": probe the GPU, then train the smoke model and stop at step 6
   (checkpoints are published to a private Hugging Face repository).
2. Restart the kernel or open a new session, choose phase "resume": the run restores from
   the Hub only and must continue with losses identical to an uninterrupted reference.

Set ``HF_TOKEN`` through Molab's environment/secrets UI. Never paste it into a cell.
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
    HAMIC_SOURCE = "git+https://github.com/<owner>/HAMIC-LM@<commit-sha>"
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
    import torch

    device = torch.device("cuda")
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        a @ a
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1000
    tflops = 20 * 2 * 4096**3 / seconds / 1e12
    mo.md(
        f"BF16 matmul sanity check on **{probe['torch']['devices'][0]['name']}**: "
        f"~{tflops:.0f} TFLOP/s (micro-benchmark, not training throughput)"
    )
    return


@app.cell
def _(mo):
    import os

    mo.stop(
        not os.environ.get("HF_TOKEN"),
        mo.md("**`HF_TOKEN` is not set.** Add it in Molab's environment settings."),
    )
    phase = mo.ui.radio(
        options=["train", "resume"], value="train", label="Drill phase (session 1 / session 2)"
    )
    phase
    return (phase,)


@app.cell
def _(HUB_REPO, RUN_ID, mo, phase):
    import tempfile
    from pathlib import Path

    from hamic.checkpoint import HuggingFaceObjectStore
    from hamic.training import SmokeConfig, run_smoke

    store = HuggingFaceObjectStore(HUB_REPO, private=True)
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
