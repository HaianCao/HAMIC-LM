# Contributing to HAMIC

Every change must pass `ruff check .`, `ruff format --check .`, `mypy` and `pytest` before it
is committed. CI runs the same checks.

## Style

- **PEP 8**, enforced by Ruff (`line-length = 99`, the PEP 8 upper bound for teams).
- **PEP 257** docstrings in Google style for every public module, class and function.
- **Type hints everywhere**; `mypy --strict` must pass. Use `from __future__ import annotations`.
- Names: `snake_case` functions and variables, `PascalCase` classes, `UPPER_CASE` constants.
- No invisible or ambiguous characters in source. Write `"​"`, not the raw character.
- Vietnamese text in code and tests is always NFC.

## Design rules

The codebase follows SOLID, concretely:

| Principle | Where it shows up |
|---|---|
| Single responsibility | One concern per module: `text`, `data`, `tokenization`, `modeling`, `checkpoint`, `training`, `runtime`. The CLI only parses arguments and calls the library. |
| Open/closed | New model architectures are added to `modeling.factory.CONFIG_BUILDERS` without editing existing builders. Corpus stages subclass `data.stages.Stage`. |
| Liskov substitution | `LocalObjectStore` and `HuggingFaceObjectStore` are interchangeable wherever an `ObjectStore` is expected; tests run on the local one. |
| Interface segregation | Small protocols (`ObjectStore`, `Segmenter`, `DocumentSource`) instead of large base classes. |
| Dependency inversion | Checkpoint publishing and resolution depend on the `ObjectStore` protocol, not on a storage vendor. |

## Compatibility rules

- **Models are standard Transformers architectures.** A released checkpoint must load with
  `AutoModelForMaskedLM` / `AutoModelForCausalLM` / `pipeline(...)` and must not require
  `trust_remote_code`. New architecture ideas go into a preset, never into a custom class,
  unless no upstream class can express them.
- **Tokenizers are `tokenizers` JSON files** loadable with `AutoTokenizer`. NFC normalization
  lives inside the tokenizer, so callers never need HAMIC code to tokenize.
- **Special-token ids are a frozen contract** (`tokenization.spec.SpecialTokens`). Changing it
  creates a new tokenizer version.
- Parameter counts in documentation come from `hamic params` (runtime), never from formulas.

## Data, checkpoints and secrets

- Data sources go through the Source Registry, which fails closed.
- Checkpoints are immutable: publish payload and manifest, verify, then `COMMITTED.json`,
  then `LATEST.json`. Never overwrite a committed checkpoint.
- Credentials come from the environment (`HF_TOKEN`). Never put them in code, notebooks,
  command lines or logs.

## What may be committed

`.gitignore` is an allow-list: only the paths it re-includes can be committed (source,
tests, notebooks, configs, the model-architecture documents in `docs/architecture/`,
packaging and CI files). Plans, research notes, agent material, data, weights and run
outputs stay out of Git. Adding a new top-level path is a deliberate
change to `.gitignore` and must be reviewed.

## Tests

- Every behaviour change comes with a test. Tests must run on CPU in a few seconds and must
  not need network access or private configs (skip with a clear reason if they would).
- The strict-resume test (`tests/test_checkpoint.py`) must stay green: it is the recovery
  drill that gates long training runs.
