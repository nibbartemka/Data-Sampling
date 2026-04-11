from __future__ import annotations

import numpy as np

from app.services.preprocessing.interfaces import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    _MODEL_CACHE: dict[str, object] = {}

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # noqa: BLE001
            raise RuntimeError(
                'Для работы preprocessing этапа требуется установленный пакет sentence-transformers'
            ) from error

        if model_name not in self._MODEL_CACHE:
            self._MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        self.model = self._MODEL_CACHE[model_name]

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
