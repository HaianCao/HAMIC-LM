"""Command-line interface: thin wrappers around the library API."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

__all__ = ["main"]


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _expand(patterns: Sequence[str]) -> list[Path]:
    paths = sorted({Path(p) for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No files match: {' '.join(patterns)}")
    return paths


def _texts(paths: Sequence[Path], limit: int | None = None) -> Iterator[str]:
    from hamic.data import read_jsonl

    count = 0
    for path in paths:
        for document in read_jsonl(path):
            if limit is not None and count >= limit:
                return
            count += 1
            yield document.text


def _cmd_probe(args: argparse.Namespace) -> int:
    from hamic.runtime import probe_runtime

    _print_json(probe_runtime(args.scratch))
    return 0


def _cmd_params(args: argparse.Namespace) -> int:
    from hamic.modeling import ModelPreset, parameter_report

    reports = [parameter_report(ModelPreset.from_yaml(p)).to_dict() for p in args.presets]
    _print_json(reports)
    return 0 if all(r["within_label"] for r in reports) else 1


def _build_source(item: dict[str, Any]) -> Any:
    from hamic.data import HuggingFaceStreamSource, JsonlSource

    kind = item.pop("type")
    if kind == "huggingface":
        item["meta_fields"] = tuple(item.get("meta_fields", ("url", "title")))
        if item.get("data_files"):
            item["data_files"] = tuple(item["data_files"])
        return HuggingFaceStreamSource(**item)
    if kind == "jsonl":
        return JsonlSource(source_id=item["source_id"], paths=_expand(item["paths"]))
    raise SystemExit(f"Unknown source type: {kind}")


def _cmd_corpus_build(args: argparse.Namespace) -> int:
    from hamic.data import CorpusPipeline, DedupIndex, PipelineConfig, SourceRegistry
    from hamic.provenance import file_sha256

    with Path(args.sources).open(encoding="utf-8") as stream:
        source_items = yaml.safe_load(stream)["sources"]
    registry = SourceRegistry.from_yaml(args.registry)
    config = PipelineConfig.from_yaml(args.config) if args.config else None
    index = DedupIndex.load(args.dedup_index) if args.dedup_index else None
    base = None
    if args.base:
        if index is None:
            raise SystemExit("--base needs --dedup-index (the delta is deduplicated against it)")
        base_manifest = Path(args.base) / "manifest.json"
        base = {
            "corpus_version": json.loads(base_manifest.read_text(encoding="utf-8"))[
                "corpus_version"
            ],
            "manifest_sha256": file_sha256(base_manifest),
        }
        if base["corpus_version"] not in index.versions:
            raise SystemExit(f"The dedup index does not include {base['corpus_version']}")
    pipeline = CorpusPipeline(
        registry, workers=args.workers, config=config, dedup_index=index, base_corpus=base
    )
    result = pipeline.run(
        [_build_source(dict(item)) for item in source_items], args.out, args.version
    )
    if index is not None:
        index.save(args.dedup_index, version=args.version)
    _print_json({"output": str(result.output_dir), "documents": result.counts})
    return 0


def _cmd_corpus_index(args: argparse.Namespace) -> int:
    from hamic.data import DedupIndex, index_corpus

    target = Path(args.out)
    index = DedupIndex.load(target) if (target / "dedup_index.json").is_file() else DedupIndex()
    if args.version in index.versions:
        raise SystemExit(f"{args.version} is already in the index")
    added = index_corpus(args.corpus, index, args.version, workers=args.workers)
    manifest = index.save(target)
    _print_json({"added": added, **manifest})
    return 0


def _cmd_corpus_assemble(args: argparse.Namespace) -> int:
    from hamic.data import assemble_tokenized

    if bool(args.base) == bool(args.base_remote):
        raise SystemExit("Pass exactly one of --base (local) or --base-remote (published)")
    if args.base_remote:
        if not args.store:
            raise SystemExit("--base-remote needs --store")
        raw = _store(args.store).read_bytes(f"{args.base_remote}/tokenized_manifest.json")
        if raw is None:
            raise SystemExit(f"No tokenized manifest under {args.base_remote!r}")
        manifest = assemble_tokenized(
            None,
            args.delta,
            args.out,
            args.version,
            base_manifest=json.loads(raw.decode("utf-8")),
            base_prefix=args.base_remote,
        )
    else:
        manifest = assemble_tokenized(args.base, args.delta, args.out, args.version)
    splits = manifest["splits"]
    _print_json({"output": args.out, "tokens": {k: v["tokens"] for k, v in splits.items()}})
    return 0


def _cmd_corpus_tokenize(args: argparse.Namespace) -> int:
    from hamic.data import tokenize_corpus

    manifest = tokenize_corpus(
        args.corpus, args.tokenizer, args.out, shard_tokens=args.shard_tokens
    )
    _print_json(
        {
            "tokenizer": manifest["tokenizer"]["name"],
            "dtype": manifest["dtype"],
            "splits": {
                k: {"documents": v["documents"], "tokens": v["tokens"]}
                for k, v in manifest["splits"].items()
            },
        }
    )
    return 0


def _default_prefix(directory: str) -> str:
    path = Path(directory).resolve()
    return f"tokenized/{path.parent.name}/{path.name}"


def _cmd_corpus_push(args: argparse.Namespace) -> int:
    from hamic.data import publish_tokenized

    prefix = args.prefix or _default_prefix(args.tokenized)
    count = publish_tokenized(args.tokenized, _store(args.store), prefix, args.copy_from)
    _print_json({"prefix": prefix, "files": count})
    return 0


def _cmd_corpus_pull(args: argparse.Namespace) -> int:
    from hamic.data import fetch_tokenized

    prefix = args.prefix or _default_prefix(args.out)
    corpus = fetch_tokenized(_store(args.store), prefix, args.out)
    splits = corpus.manifest["splits"]
    _print_json({"prefix": prefix, "tokens": {k: v["tokens"] for k, v in splits.items()}})
    return 0


def _cmd_corpus_annotate(args: argparse.Namespace) -> int:
    from hamic.data import annotate_sources

    manifest = annotate_sources(args.corpus, args.tokenized)
    _print_json({"tokenized": args.tokenized, "source_ids": manifest["source_ids"]})
    return 0


def _source_weights(items: Sequence[str]) -> dict[str, float]:
    """Parse repeated ``SOURCE=WEIGHT`` CLI values."""
    weights: dict[str, float] = {}
    for item in items:
        source, separator, value = item.partition("=")
        if not separator or not source:
            raise SystemExit(f"Invalid source weight {item!r}; expected SOURCE=WEIGHT")
        weights[source] = float(value)
    return weights


def _cmd_corpus_sample(args: argparse.Namespace) -> int:
    from hamic.data import sample_corpora

    manifest = sample_corpora(
        args.corpus,
        args.out,
        args.documents,
        seed=args.seed,
        source_weights=_source_weights(args.source_weight),
        shard_documents=args.shard_documents,
    )
    _print_json(
        {
            "output": args.out,
            "documents": manifest["documents"],
            "characters": manifest["characters"],
            "files": len(manifest["files"]),
            "sources": manifest["sources"],
        }
    )
    return 0


def _cmd_corpus_sample_tokenized(args: argparse.Namespace) -> int:
    from hamic.data import sample_published_tokenized

    manifest = sample_published_tokenized(
        args.tokenized,
        _store(args.store),
        args.prefix,
        args.out,
        args.tokens,
        seed=args.seed,
        source_weights=_source_weights(args.source_weight),
        shard_documents=args.shard_documents,
        batch_size=args.batch_size,
        shard_indices=args.shard_index or None,
    )
    _print_json(
        {
            "output": args.out,
            "covered_tokens": manifest["selection"]["covered_tokens"],
            "documents": manifest["documents"],
            "files": len(manifest["files"]),
            "sources": manifest["sources"],
        }
    )
    return 0


def _cmd_tokenizer_train(args: argparse.Namespace) -> int:
    from hamic.tokenization import TokenizerKind, TokenizerSpec, export_tokenizer, train_tokenizer

    spec = TokenizerSpec.from_yaml(args.spec)
    corpus = _expand(args.corpus)
    segmenter = None
    if args.segmenter == "pyvi":
        from hamic.tokenization import PyviSegmenter

        segmenter = PyviSegmenter()
    tokenizer = train_tokenizer(
        spec, _texts(corpus, args.limit), segmenter=segmenter, show_progress=True
    )
    provenance = {"corpus_files": [p.as_posix() for p in corpus], "limit": args.limit}
    for kind in args.kind:
        export_tokenizer(tokenizer, spec, Path(args.out) / kind, TokenizerKind(kind), provenance)
    _print_json({"name": spec.name, "vocab_size": tokenizer.get_vocab_size(), "out": args.out})
    return 0


def _tokenizer_name(directory: Path) -> str:
    """Spec name from ``hamic_tokenizer.json``, else the directory name."""
    metadata = directory / "hamic_tokenizer.json"
    if metadata.is_file():
        return str(json.loads(metadata.read_text(encoding="utf-8"))["spec"]["name"])
    return directory.name


def _cmd_tokenizer_eval(args: argparse.Namespace) -> int:
    from tokenizers import Tokenizer

    from hamic.tokenization import TokenizerEvaluator

    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    report = TokenizerEvaluator(tokenizer, name=_tokenizer_name(Path(args.tokenizer))).run(
        _texts(_expand(args.corpus), args.limit)
    )
    data = report.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    _print_json({k: v for k, v in data.items() if k != "probes"})
    return 0 if report.passed_hard_gates else 1


def _cmd_tokenizer_compare(args: argparse.Namespace) -> int:
    from hamic.tokenization import comparison_table

    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    print(comparison_table(reports))
    return 0


def _store(spec: str) -> Any:
    from hamic.checkpoint import HuggingFaceObjectStore, LocalObjectStore

    kind, _, location = spec.partition(":")
    if kind == "local" and location:
        return LocalObjectStore(location)
    if kind in {"hf", "hf-dataset"} and location:
        repo_type = "dataset" if kind == "hf-dataset" else "model"
        return HuggingFaceObjectStore(location, repo_type=repo_type, private=True)
    raise SystemExit(
        "--store must be local:<dir>, hf:<namespace/repo> or hf-dataset:<namespace/repo>"
    )


def _cmd_drill(args: argparse.Namespace) -> int:
    from hamic.training import SmokeConfig, run_smoke

    result = run_smoke(
        _store(args.store),
        args.workdir,
        SmokeConfig(run_id=args.run_id, steps=args.steps),
        stop_after_step=args.stop_after,
    )
    _print_json(
        {
            "resumed_from_step": result.resumed_from_step,
            "final_step": result.final_step,
            "published_steps": result.published_steps,
            "losses": result.losses,
        }
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from hamic.training import PretrainConfig, Pretrainer

    config = PretrainConfig.from_yaml(args.config)
    result = Pretrainer(config, _store(args.store), args.workdir).run(
        stop_after_steps=args.stop_after_steps
    )
    _print_json(
        {
            "status": result.status,
            "final_step": result.final_step,
            "total_steps": config.total_steps,
            "resumed_from_step": result.resumed_from_step,
            "published_steps": result.published_steps,
            "last_eval_loss": list(result.evaluations.values())[-1:] or None,
            "warnings": result.warnings,
        }
    )
    return 0 if result.status in {"completed", "paused", "already_complete"} else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hamic", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="describe the runtime (GPU, CUDA, versions)")
    probe.add_argument("--scratch", default=".")
    probe.set_defaults(func=_cmd_probe)

    params = sub.add_parser("params", help="count runtime parameters of model presets")
    params.add_argument("presets", nargs="+")
    params.set_defaults(func=_cmd_params)

    corpus = sub.add_parser("corpus", help="corpus pipeline").add_subparsers(
        dest="corpus_command", required=True
    )
    build = corpus.add_parser("build", help="clean, deduplicate and split sources")
    build.add_argument("--registry", required=True)
    build.add_argument("--sources", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--config", help="pipeline configuration YAML (defaults if omitted)")
    build.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="worker processes for normalize/PII/quality/MinHash (1 = serial)",
    )
    build.add_argument(
        "--dedup-index",
        help="persistent dedup index of earlier versions; the build writes a delta and "
        "extends the index",
    )
    build.add_argument("--base", help="previous corpus version this delta extends")
    build.set_defaults(func=_cmd_corpus_build)

    tokenize = corpus.add_parser("tokenize", help="write memory-mappable token shards")
    tokenize.add_argument("--corpus", required=True, help="frozen corpus directory")
    tokenize.add_argument("--tokenizer", required=True, help="directory with tokenizer.json")
    tokenize.add_argument("--out", required=True)
    tokenize.add_argument("--shard-tokens", type=int, default=50_000_000)
    tokenize.set_defaults(func=_cmd_corpus_tokenize)

    store_help = "local:<dir>, hf:<namespace/repo> or hf-dataset:<namespace/repo>"
    prefix_help = "remote prefix (default: tokenized/<parent dir>/<dir>)"
    push = corpus.add_parser("push", help="upload tokenized shards to a store")
    push.add_argument("--tokenized", required=True, help="tokenized directory")
    push.add_argument("--store", required=True, help=store_help)
    push.add_argument("--prefix", help=prefix_help)
    push.add_argument(
        "--copy-from",
        help="published folder of the previous version; identical files are copied on the "
        "server instead of uploaded",
    )
    push.set_defaults(func=_cmd_corpus_push)

    index = corpus.add_parser("index", help="add a frozen corpus to a persistent dedup index")
    index.add_argument("--corpus", required=True, help="frozen corpus directory")
    index.add_argument("--version", required=True, help="corpus version recorded in the index")
    index.add_argument("--out", required=True, help="index directory (created or extended)")
    index.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    index.set_defaults(func=_cmd_corpus_index)

    assemble = corpus.add_parser(
        "assemble", help="hard-link a complete tokenized version from base + delta"
    )
    assemble.add_argument("--base", help="previous complete tokenized version (local)")
    assemble.add_argument(
        "--base-remote",
        help="published folder of the previous version, e.g. tokenized/v0.4/t1-bbpe-20544; "
        "only the delta is written locally",
    )
    assemble.add_argument("--store", help=f"store of --base-remote ({store_help})")
    assemble.add_argument("--delta", required=True, help="tokenized delta")
    assemble.add_argument("--out", required=True, help="complete version to write")
    assemble.add_argument("--version", required=True, help="corpus version, e.g. v0.3")
    assemble.set_defaults(func=_cmd_corpus_assemble)

    pull = corpus.add_parser("pull", help="download and verify tokenized shards")
    pull.add_argument("--store", required=True, help=store_help)
    pull.add_argument("--out", required=True, help="local tokenized directory")
    pull.add_argument("--prefix", help=prefix_help)
    pull.set_defaults(func=_cmd_corpus_pull)

    annotate = corpus.add_parser(
        "annotate-sources",
        help="add per-document source ids to shards tokenized without them",
    )
    annotate.add_argument("--corpus", required=True, help="frozen corpus the shards came from")
    annotate.add_argument("--tokenized", required=True, help="tokenized directory to update")
    annotate.set_defaults(func=_cmd_corpus_annotate)

    sample = corpus.add_parser(
        "sample", help="deterministically sample train documents from corpus deltas"
    )
    sample.add_argument("--corpus", nargs="+", required=True, help="frozen base/delta dirs")
    sample.add_argument("--out", required=True, help="new sample directory")
    sample.add_argument(
        "--documents", type=int, required=True, help="expected documents before weighting"
    )
    sample.add_argument("--seed", default="hamic-tokenizer-sample-v1")
    sample.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help="multiply the selection probability for one source (repeatable)",
    )
    sample.add_argument("--shard-documents", type=int, default=50_000)
    sample.set_defaults(func=_cmd_corpus_sample)

    sample_tokens = corpus.add_parser(
        "sample-tokenized",
        help="decode and round-trip-check distributed published token shards",
    )
    sample_tokens.add_argument("--tokenized", required=True, help="local lightweight manifest")
    sample_tokens.add_argument("--store", required=True, help=store_help)
    sample_tokens.add_argument("--prefix", required=True, help="current published folder")
    sample_tokens.add_argument("--out", required=True, help="new decoded sample directory")
    sample_tokens.add_argument("--tokens", type=int, required=True, help="approximate coverage")
    sample_tokens.add_argument("--seed", default="hamic-tokenizer-shards-v1")
    sample_tokens.add_argument(
        "--source-weight", action="append", default=[], metavar="SOURCE=WEIGHT"
    )
    sample_tokens.add_argument("--shard-documents", type=int, default=50_000)
    sample_tokens.add_argument("--batch-size", type=int, default=256)
    sample_tokens.add_argument(
        "--shard-index",
        action="append",
        type=int,
        default=[],
        help="decode this exact train-shard index instead of quantiles (repeatable)",
    )
    sample_tokens.set_defaults(func=_cmd_corpus_sample_tokenized)

    tok = sub.add_parser("tokenizer", help="tokenizer training and evaluation").add_subparsers(
        dest="tokenizer_command", required=True
    )
    train = tok.add_parser("train", help="train and export a tokenizer")
    train.add_argument("--spec", required=True)
    train.add_argument("--corpus", nargs="+", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--kind", nargs="+", default=["encoder", "decoder"])
    train.add_argument("--limit", type=int)
    train.add_argument(
        "--segmenter", choices=["pyvi"], help="word segmenter for input_mode: segmented"
    )
    train.set_defaults(func=_cmd_tokenizer_train)

    evaluate = tok.add_parser("eval", help="run the acceptance suite")
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--corpus", nargs="+", required=True)
    evaluate.add_argument("--out")
    evaluate.add_argument("--limit", type=int, default=5000)
    evaluate.set_defaults(func=_cmd_tokenizer_eval)

    compare = tok.add_parser("compare", help="print a Markdown comparison of eval reports")
    compare.add_argument("reports", nargs="+")
    compare.set_defaults(func=_cmd_tokenizer_compare)

    train = sub.add_parser("train", help="pretrain (or resume) a model from a run config")
    train.add_argument("--config", required=True, help="pretraining config YAML")
    train.add_argument("--store", required=True, help="local:<dir> or hf:<namespace/repo>")
    train.add_argument("--workdir", required=True, help="local scratch directory")
    train.add_argument("--stop-after-steps", type=int, help="drill: stop without a checkpoint")
    train.set_defaults(func=_cmd_train)

    drill = sub.add_parser("drill", help="checkpoint/strict-resume recovery drill")
    drill.add_argument("--store", required=True, help="local:<dir> or hf:<namespace/repo>")
    drill.add_argument("--workdir", required=True)
    drill.add_argument("--run-id", default="smoke")
    drill.add_argument("--steps", type=int, default=12)
    drill.add_argument("--stop-after", type=int)
    drill.set_defaults(func=_cmd_drill)

    evaluate = sub.add_parser(
        "eval-mlm",
        help="score published MLM checkpoints under token, span and PMI masking views",
    )
    evaluate.add_argument("--store", required=True, help="store holding runs/<run_id>/")
    evaluate.add_argument("--runs", nargs="+", required=True, help="run ids to score")
    evaluate.add_argument(
        "--tokenized", help="validation shards (default: the first run's tokenized directory)"
    )
    evaluate.add_argument("--views", nargs="+", default=["token", "span", "pmi"])
    evaluate.add_argument("--rate", type=float, default=0.15, help="mask rate of every view")
    evaluate.add_argument("--sequences", type=int, default=512, help="validation rows")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--workdir", default=".hamic-eval", help="PMI pair cache")
    evaluate.add_argument("--out", help="write results as JSON")
    evaluate.set_defaults(func=_cmd_eval_mlm)

    downstream = sub.add_parser(
        "downstream", help="fine-tune a published encoder on the Vietnamese task suite"
    )
    downstream.add_argument("--store", required=True, help="store holding runs/<run_id>/")
    downstream.add_argument("--run", required=True, help="run id of the encoder")
    downstream.add_argument(
        "--tokenizer", required=True, help="tokenizer.json of the run (e.g. its tokenized dir)"
    )
    downstream.add_argument("--tasks", nargs="+", help="task names (default: all nine)")
    downstream.add_argument("--learning-rates", nargs="+", type=float, default=[2e-5, 5e-5])
    downstream.add_argument("--seeds", nargs="+", type=int, default=[10, 42, 100])
    downstream.add_argument("--max-epochs", type=int, default=10)
    downstream.add_argument("--batch-size", type=int, default=32)
    downstream.add_argument("--device", default="auto")
    downstream.add_argument("--workdir", default=".hamic-downstream")
    downstream.add_argument("--out", help="write results as JSON")
    downstream.set_defaults(func=_cmd_downstream)
    return parser


def _cmd_eval_mlm(args: argparse.Namespace) -> int:
    import torch

    from hamic.data import TokenizedCorpus
    from hamic.training.evaluate import evaluate_views, load_published_model, masking_views

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    store = _store(args.store)
    results: dict[str, Any] = {}
    corpus = None
    views = None
    for run_id in args.runs:
        published = load_published_model(store, run_id, device=device)
        if corpus is None:
            corpus = TokenizedCorpus(args.tokenized or published.config.tokenized)
            views = masking_views(
                corpus, published.preset, rate=args.rate, cache_dir=args.workdir, views=args.views
            )
        assert views is not None
        losses = evaluate_views(
            published, corpus, views, args.sequences, args.batch_size, device=device
        )
        results[run_id] = {"step": published.step, **losses}
        _print_json({run_id: results[run_id]})
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def _cmd_downstream(args: argparse.Namespace) -> int:
    import torch

    from hamic.downstream import (
        TASKS,
        FinetuneConfig,
        build_tokenizer,
        load_task,
        run_task,
        save_for_finetuning,
    )
    from hamic.training.evaluate import load_published_model

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    names = args.tasks or list(TASKS)
    unknown = sorted(set(names) - set(TASKS))
    if unknown:
        raise SystemExit(f"Unknown tasks: {unknown}")
    workdir = Path(args.workdir)
    published = load_published_model(_store(args.store), args.run, device="cpu")
    model_dir = save_for_finetuning(published.model, workdir / "models" / args.run)
    tokenizer = build_tokenizer(args.tokenizer)
    cfg = FinetuneConfig(
        learning_rates=tuple(args.learning_rates),
        seeds=tuple(args.seeds),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=device,
    )
    results = {"run": args.run, "step": published.step, "tasks": {}}
    for name in names:
        data = load_task(TASKS[name], cache_dir=str(workdir / "datasets"))
        result = run_task(
            model_dir, data, tokenizer, cfg, log=lambda line: print(line, flush=True)
        )
        results["tasks"][name] = result.to_dict()
        _print_json({name: {"mean": result.mean, "std": result.std, "metric": result.metric}})
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``hamic`` command."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
