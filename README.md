# HAMIC

HAMIC is a research toolkit for building Vietnamese-first language models: Vietnamese text
normalization, corpus preparation with source governance, tokenizer training and evaluation,
and crash-safe checkpointing for single-GPU training sessions.

> **Status:** pre-release research code (`0.1.0.dev0`). APIs may change. No model weights
> are published yet. Research use only.

## Installation

```bash
pip install -e ".[dev]"
```

PyTorch is a dependency. For a CPU-only machine, install the CPU wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick start

Normalize Vietnamese text the same way the tokenizer does:

```python
from hamic.text import VietnameseNormalizer

normalizer = VietnameseNormalizer()
normalizer("Tiếng Việt")  # NFD input -> NFC output
```

Train a byte-level BPE tokenizer and load it back with the standard Transformers API:

```bash
hamic tokenizer train --spec my-tokenizer.yaml --corpus data/train/*.jsonl.gz --out out/tok
```

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("out/tok")
tokenizer("Hà Nội là thủ đô của Việt Nam.")
```

HAMIC model checkpoints are exported as standard Transformers architectures, so released
models load with `AutoModelForMaskedLM` / `AutoModelForCausalLM` and `trust_remote_code`
is not required.

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding conventions.

## License

Not yet chosen. Until a license is published, all rights are reserved by HAMIC.
