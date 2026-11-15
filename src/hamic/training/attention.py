"""Attention masks that keep packed documents apart.

Packed rows concatenate ``<s> document </s>`` pieces. With cross-document attention
disabled, a position may only attend to positions of its own document. Documents are
recovered from the ids alone: ``<s>`` opens every document and is never masked or
replaced by the MLM objective, so the running count of ``<s>`` along a row numbers the
documents (the leading continuation of a document cut by the previous row gets 0).
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["DocumentMask", "document_ids"]


def document_ids(input_ids: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Number the documents of each packed row: ``[batch, seq]`` int32."""
    return torch.cumsum(input_ids == bos_id, dim=1, dtype=torch.int32)


class DocumentMask:
    """Build block-diagonal attention masks in the format of an attention backend.

    The result is passed as ``attention_mask`` to Transformers models that accept a
    per-layer-type mapping (ModernBERT): ``{"full_attention": mask, ...}``.

    Args:
        backend: ``"sdpa"`` (boolean ``[batch, 1, seq, seq]`` tensor, ``True`` attends) or
            ``"flex_attention"`` (a ``BlockMask``; skips whole blocks across documents).
        layer_types: Attention layer types of the model; each gets the same mask.
    """

    def __init__(self, backend: str, layer_types: tuple[str, ...] = ("full_attention",)) -> None:
        if backend not in {"sdpa", "flex_attention"}:
            raise ValueError(f"Unsupported attention backend: {backend}")
        self.backend = backend
        self.layer_types = tuple(dict.fromkeys(layer_types))

    def __call__(self, documents: torch.Tensor) -> dict[str, Any]:
        """Masks for ``documents`` (output of :func:`document_ids`, on the model device)."""
        mask = self._sdpa(documents) if self.backend == "sdpa" else self._flex(documents)
        return dict.fromkeys(self.layer_types, mask)

    @staticmethod
    def _sdpa(documents: torch.Tensor) -> torch.Tensor:
        return (documents[:, :, None] == documents[:, None, :]).unsqueeze(1)

    @staticmethod
    def _flex(documents: torch.Tensor) -> Any:
        from torch.nn.attention.flex_attention import create_block_mask

        def same_document(b: Any, h: Any, q: Any, kv: Any) -> Any:
            return documents[b, q] == documents[b, kv]

        batch, length = documents.shape
        return create_block_mask(
            same_document, batch, None, length, length, device=str(documents.device)
        )
