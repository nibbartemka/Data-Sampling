import numpy as np
from sentence_transformers import SentenceTransformer

from .interfaces import BaseEmbedder

__all__ = [
    'SentenceTransformerEmbedder',
]


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
