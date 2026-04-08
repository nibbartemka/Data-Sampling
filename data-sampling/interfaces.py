from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pandas as pd
    import numpy as np


__all__ = [
    'BaseDataManager',
    'BaseTextProcessor',
    'BaseStratifier',
    'BaseEmbedder',
]


class BaseDataManager(ABC):
    @abstractmethod
    def load_dataset(
        self,
        path: str,
        required_columns: list[str],
    ) -> 'pd.DataFrame':
        pass

    @abstractmethod
    def save_dataset(
        path: str,
        dataset: 'pd.DataFrame',
    ) -> None:
        pass


class BaseTextProcessor(ABC):
    @abstractmethod
    def process_text(self, text: str) -> str:
        pass


class BaseStratifier(ABC):
    @abstractmethod
    def build_stratum_key(
        self,
        dataset: 'pd.DataFrame',
        stratify_columns: list[str]
    ) -> 'pd.DataFrame':
        pass

    @abstractmethod
    def split(self, dataset: 'pd.DataFrame') -> list[tuple[str, 'pd.DataFrame']]:
        pass


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> 'np.ndarray':
        pass


class BaseClusterer(ABC):
    @abstractmethod
    def cluster(
        self,
        embeddings: 'np.ndarray',
        default_k: int
    ) -> 'np.ndarray':
        pass
