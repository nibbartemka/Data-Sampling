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
    'BaseClusterer',
    'BaseSemanticSelector',
    'BaseRuleEnricher',
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


class BaseSemanticSelector(ABC):
    @abstractmethod
    def select(
        self,
        dataset_stratum: "pd.DataFrame",
        embeddings: "np.ndarray",
        cluster_labels: "np.ndarray",
        id_column: str,
        center_per_cluster: int,
        outlier_per_cluster: int,
    ) -> dict:
        pass


class BaseRuleEnricher(ABC):
    @abstractmethod
    def enrich(
        self,
        dataset_stratum: 'pd.DataFrame',
        text_column: str,
        id_column: str,
        max_per_pattern: int,
    ) -> dict:
        pass
