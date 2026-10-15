"""Sentence-embedding wrapper returning L2-normalized numpy vectors."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray: ...


def _best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device or _best_device())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vectors.astype(np.float32)
