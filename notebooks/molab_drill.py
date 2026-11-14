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
        Check `HAMIC_SOURCE`, `HUB_REPO` and `RUN_ID` in the next cell, then run the cells
        in order. After pushing new code, restart the kernel and run again.
        """
    )
    return (mo,)


@app.cell
def _():
    # Any Git ref works after "@": a branch (main), a tag (v0.1.0) or a commit SHA.
    # The resolved commit is always recorded, so a branch stays reproducible; pin a tag or
    # SHA for long runs so every session of the run uses the same code.
    HAMIC_SOURCE = "git+https://github.com/HaianCao/HAMIC-LM@main"
    HUB_REPO = "HaianCao/hamic-runs"
    RUN_ID = "drill-003"  # use a new id per drill; a finished run cannot be resumed
    # Faster parallel Hub transfers. Read when huggingface_hub is first imported, so
    # changing it requires a kernel restart.
    XET_HIGH_PERFORMANCE = True
    return HAMIC_SOURCE, HUB_REPO, RUN_ID, XET_HIGH_PERFORMANCE


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
    looks_like_sha = len(requested_ref) >= 7 and is_hex
    if looks_like_sha and not installed_commit.startswith(requested_ref):
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
def _(hamic_ready, mo):
    from hamic.runtime import probe_runtime

    del hamic_ready  # dependency only: run after the install cell

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
def _(HUB_REPO, hamic_ready, mo, token_input):
    import os

    from hamic.checkpoint import HuggingFaceObjectStore

    hub_token = token_input.value.strip() or os.environ.get("HF_TOKEN", "")
    mo.stop(not hub_token, mo.md("**Enter a Hugging Face token above** to continue."))
    if "<" in HUB_REPO:
        raise ValueError("Set HUB_REPO to your private Hugging Face repository first")
    from huggingface_hub import constants as _hf_constants

    store = HuggingFaceObjectStore(HUB_REPO, private=True, token=hub_token)
    account = store.whoami()
    xet_mode = "high-performance" if _hf_constants.HF_XET_HIGH_PERFORMANCE else "default"
    phase = mo.ui.radio(
        options=["train", "resume"], value="train", label="Drill phase (session 1 / session 2)"
    )
    mo.vstack(
        [
            mo.md(
                f"Hub account **{account}** · run store `{HUB_REPO}` · "
                f"code `{hamic_ready[:12]}` · Xet transfers: **{xet_mode}**"
            ),
            phase,
        ]
    )
    return phase, store


@app.cell
def _(mo):
    bench_size = mo.ui.dropdown(
        options={"256 MiB": 256, "1 GiB": 1024, "2 GiB": 2048},
        value="1 GiB",
        label="Total size",
    )
    bench_files = mo.ui.dropdown(
        options={"1 file": 1, "4 files": 4, "8 files": 8, "16 files": 16},
        value="8 files",
        label="Split into",
    )
    bench_button = mo.ui.run_button(label="Measure Hub upload/download throughput")
    mo.vstack(
        [
            mo.md(
                "Upload speed decides how often a run can checkpoint; download speed decides "
                "how long a resume takes. Compare **1 file** with **8 files**: sharded "
                "checkpoints are restored with parallel downloads. Test files stay under "
                "`bench/` in the run repository."
            ),
            mo.hstack([bench_size, bench_files, bench_button], justify="start"),
        ]
    )
    return bench_button, bench_files, bench_size


@app.cell
def _(bench_button, bench_files, bench_size, mo, store):
    from hamic.runtime import storage_throughput

    mo.stop(not bench_button.value)
    throughput = storage_throughput(store, size_mib=bench_size.value, files=bench_files.value)
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
        f"published steps `{result.published_steps}` · final step `{result.final_step}` · "
        f"code `{(result.code_commit or 'unknown')[:12]}`"
        + "".join(f"\n\n> Warning: {w}" for w in result.warnings)
    )
    return config, result, run_smoke, workdir


@app.cell
def _(config, mo, phase, result, run_smoke, workdir):
    from hamic.checkpoint import LocalObjectStore

    mo.stop(phase.value != "resume")
    reference = run_smoke(LocalObjectStore(workdir / "reference-store"), workdir / "ref", config)
    identical = all(reference.losses[k] == v for k, v in result.losses.items())
    rows = [
        {"step": k, "resumed": v, "reference": reference.losses[k]}
        for k, v in sorted(result.losses.items())
    ]
    for row in rows:
        row["equal"] = row["resumed"] == row["reference"]
    mo.vstack(
        [
            mo.md(
                "### Drill passed: strict resume reproduces the uninterrupted run"
                if identical and result.resumed_from_step == 4
                else "### Drill FAILED: compare the losses below"
            ),
            mo.ui.table(rows, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
