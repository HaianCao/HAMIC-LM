"""Molab evaluation: score published MLM runs under token, span and PMI masking views.

Validation loss with token-level masking favours models trained with token-level masking
(PMI-Masking, Levine et al. 2021, Table 4). This notebook scores any set of finished runs on
the same validation rows with the same seeded masks under three views, so span and PMI
masking can be compared fairly. It only reads from the Hub and never writes.

Paste a fine-grained Hugging Face token with **read** access to ``RUN_REPO`` and
``DATA_REPO`` into the password field. It stays in kernel memory and is never written to
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
        # HAMIC · Molab MLM evaluation
        Run the cells in order, pick the runs and the masking views, and press **Evaluate**.
        Nothing is written to the Hub.
        """
    )
    return (mo,)


@app.cell
def _():
    # Pin the same commit as the runs when results go into a report.
    HAMIC_SOURCE = "git+https://github.com/HaianCao/HAMIC-LM@main"
    RUN_REPO = "HaianCao/hamic-runs"  # private model repo with runs/<run_id>/
    DATA_REPO = "HaianCao/hamic-data"  # private dataset repo with tokenized shards
    XET_HIGH_PERFORMANCE = True
    return DATA_REPO, HAMIC_SOURCE, RUN_REPO, XET_HIGH_PERFORMANCE


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
    subprocess.run([*pip, HAMIC_SOURCE], check=True)
    subprocess.run([*pip, "--force-reinstall", "--no-deps", HAMIC_SOURCE], check=True)
    direct_url = json.loads(metadata.distribution("hamic").read_text("direct_url.json") or "{}")
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id", "")
    if not installed_commit:
        raise RuntimeError("pip did not record the installed commit (direct_url.json)")
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
    import os
    import tempfile
    from pathlib import Path

    import torch

    del hamic_ready  # dependency only: run after the install cell
    work_root = Path(tempfile.gettempdir()) / "hamic-eval"
    work_root.mkdir(parents=True, exist_ok=True)
    os.chdir(work_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mo.md(
        f"Working directory `{work_root}` · device **{device}**"
        + ("" if device == "cuda" else " (no GPU attached: evaluation will be slow)")
    )
    return Path, device, os, work_root


@app.cell
def _(mo):
    token_input = mo.ui.text(
        kind="password", label="Hugging Face token (read: RUN_REPO, DATA_REPO)", full_width=True
    )
    mo.vstack(
        [
            mo.md("The token stays in this session's memory and is not saved in the notebook."),
            token_input,
        ]
    )
    return (token_input,)


@app.cell
def _(DATA_REPO, RUN_REPO, mo, os, token_input):
    from hamic.checkpoint import HuggingFaceObjectStore

    hub_token = token_input.value.strip() or os.environ.get("HF_TOKEN", "")
    mo.stop(not hub_token, mo.md("**Enter a Hugging Face token above** to continue."))
    run_store = HuggingFaceObjectStore(RUN_REPO, create=False, token=hub_token)
    data_store = HuggingFaceObjectStore(
        DATA_REPO, repo_type="dataset", create=False, token=hub_token
    )
    finished = sorted(
        path.split("/")[1]
        for path in run_store.list_files("runs/")
        if path.endswith("/COMPLETE.json")
    )
    mo.md(f"Account **{run_store.whoami()}** · {len(finished)} finished runs in `{RUN_REPO}`")
    return data_store, finished, run_store


@app.cell
def _(finished, mo):
    default_runs = [r for r in finished if r.startswith("e-v02-")] or finished[:1]
    runs = mo.ui.multiselect(options=finished, value=default_runs, label="Runs")
    views = mo.ui.multiselect(
        options=["token", "span", "pmi"], value=["token", "span", "pmi"], label="Masking views"
    )
    sequences = mo.ui.number(
        start=64, stop=8192, step=64, value=2048, label="Validation sequences"
    )
    rate = mo.ui.number(start=0.05, stop=0.5, step=0.05, value=0.15, label="Mask rate")
    evaluate_button = mo.ui.run_button(label="Evaluate")
    mo.vstack([runs, mo.hstack([views, sequences, rate]), evaluate_button])
    return evaluate_button, rate, runs, sequences, views


@app.cell
def _(
    Path,
    data_store,
    device,
    evaluate_button,
    mo,
    rate,
    run_store,
    runs,
    sequences,
    views,
    work_root,
):
    from hamic.data import TokenizedCorpus, fetch_tokenized
    from hamic.training.evaluate import evaluate_views, load_published_model, masking_views

    mo.stop(not evaluate_button.value, mo.md("Pick runs and press **Evaluate**."))
    mo.stop(not runs.value, mo.md("Select at least one run."))
    rows = []
    corpus = None
    objectives = None
    for run_id in runs.value:
        with mo.status.spinner(f"Loading {run_id} ..."):
            published = load_published_model(run_store, run_id, device=device)
        if corpus is None:
            local = Path(published.config.tokenized)
            prefix = f"tokenized/{local.parent.name}/{local.name}"
            with mo.status.spinner(f"Fetching and verifying {prefix} ..."):
                fetch_tokenized(data_store, prefix, local)
            corpus = TokenizedCorpus(local)
            with mo.status.spinner("Mining PMI pairs (first 100M train tokens) ..."):
                objectives = masking_views(
                    corpus,
                    published.preset,
                    rate=float(rate.value),
                    cache_dir=work_root / "cache",
                    views=views.value,
                )
        if published.config.tokenized != corpus.directory.as_posix():
            # Every run is scored on the validation split of the first run's corpus.
            note = f"skipped: trained on {published.config.tokenized}"
            rows.append({"run": run_id, "note": note})
            continue
        with mo.status.spinner(f"Evaluating {run_id} ..."):
            losses = evaluate_views(
                published, corpus, objectives, int(sequences.value), batch_size=64, device=device
            )
        rows.append({"run": run_id, "step": published.step, **losses})
        print(rows[-1], flush=True)
    mo.ui.table(rows, selection=None)
    return (rows,)


if __name__ == "__main__":
    app.run()
