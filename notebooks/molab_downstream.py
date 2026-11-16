"""Molab downstream evaluation: fine-tune published HAMIC encoders on the Vietnamese suite.

Nine tasks on eight datasets (PhoNER, VSMEC, ViHSD, ViNLI, ViQuAD2.0, UD-VTB POS,
VSFC sentiment/topic, VToC), pinned to Hub revisions; see ``hamic.downstream.tasks`` for
the deviations from HuITBERT's benchmark. Protocol: AdamW, 10% warmup, batch 32, length
512, up to 10 epochs with early stopping on dev; the learning rate is chosen on dev and test
is reported as mean and std over seeds. "Quick" mode (one rate, one seed, three epochs)
is for comparing proxy runs; reports use the full protocol.

It only reads from the Hub. Paste a fine-grained token with **read** access to
``RUN_REPO`` and ``DATA_REPO`` into the password field (never into a code cell).
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # HAMIC · Molab downstream evaluation
        Run the cells in order, pick a run, the tasks and the protocol, then press **Run**.
        Nothing is written to the Hub.
        """
    )
    return (mo,)


@app.cell
def _():
    HAMIC_SOURCE = "git+https://github.com/HaianCao/HAMIC-LM@main"
    RUN_REPO = "HaianCao/hamic-runs"
    DATA_REPO = "HaianCao/hamic-data"
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
    return hamic_ready, json


@app.cell
def _(hamic_ready, mo):
    import os
    import tempfile
    from pathlib import Path

    import torch

    del hamic_ready  # dependency only
    work_root = Path(tempfile.gettempdir()) / "hamic-downstream"
    work_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mo.md(f"Working directory `{work_root}` · device **{device}**")
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
    mo.md(f"Account **{run_store.whoami()}** · {len(finished)} finished runs")
    return data_store, finished, run_store


@app.cell
def _(finished, mo):
    from hamic.downstream import TASKS

    run = mo.ui.dropdown(options=finished, value=finished[-1] if finished else None, label="Run")
    tasks = mo.ui.multiselect(options=list(TASKS), value=list(TASKS), label="Tasks")
    protocol = mo.ui.radio(
        options={
            "Quick (lr 5e-5, 1 seed, 3 epochs) - compare proxy runs": "quick",
            "Full (lr 2e-5/5e-5 on dev, 3 seeds, 10 epochs) - reports": "full",
        },
        value="Quick (lr 5e-5, 1 seed, 3 epochs) - compare proxy runs",
        label="Protocol",
    )
    run_button = mo.ui.run_button(label="Run")
    mo.vstack([run, tasks, protocol, run_button])
    return TASKS, protocol, run, run_button, tasks


@app.cell
def _(
    Path,
    TASKS,
    data_store,
    device,
    json,
    mo,
    protocol,
    run,
    run_button,
    run_store,
    tasks,
    work_root,
):
    from hamic.downstream import (
        FinetuneConfig,
        build_tokenizer,
        load_task,
        run_task,
        save_for_finetuning,
    )
    from hamic.training.evaluate import load_published_model

    mo.stop(not run_button.value, mo.md("Pick a run and press **Run**."))
    mo.stop(not run.value or not tasks.value, mo.md("Select a run and at least one task."))
    with mo.status.spinner(f"Loading {run.value} ..."):
        published = load_published_model(run_store, run.value, device="cpu")
        model_dir = save_for_finetuning(published.model, work_root / "models" / run.value)
        local = Path(published.config.tokenized)
        prefix = f"tokenized/{local.parent.name}/{local.name}"
        tokenizer_path = data_store.download(
            f"{prefix}/tokenizer.json", work_root / "tokenizers" / local.name / "tokenizer.json"
        )
        tokenizer = build_tokenizer(tokenizer_path)
    if protocol.value == "quick":
        cfg = FinetuneConfig(learning_rates=(5e-5,), seeds=(42,), max_epochs=3, device=device)
    else:
        cfg = FinetuneConfig(device=device)
    rows = []
    results = {"run": run.value, "step": published.step, "protocol": protocol.value, "tasks": {}}
    for name in tasks.value:
        with mo.status.spinner(f"{name}: loading data ..."):
            data = load_task(TASKS[name], cache_dir=str(work_root / "datasets"))
        with mo.status.spinner(f"{name}: fine-tuning ..."):
            result = run_task(
                model_dir, data, tokenizer, cfg, log=lambda line: print(line, flush=True)
            )
        results["tasks"][name] = result.to_dict()
        rows.append(
            {
                "task": name,
                "metric": result.metric,
                "mean": result.mean,
                "std": result.std,
                "lr": result.learning_rate,
                "minutes": round(result.seconds / 60, 1),
            }
        )
    out = work_root / f"downstream-{run.value}-{protocol.value}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results), flush=True)
    mo.vstack([mo.ui.table(rows, selection=None), mo.md(f"Saved `{out}`")])
    return (results,)


if __name__ == "__main__":
    app.run()
