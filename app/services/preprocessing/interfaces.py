from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTextProcessor(ABC):
    @abstractmethod
    def process_text(self, text: str) -> str:
        raise NotImplementedError


class BaseStratifier(ABC):
    @abstractmethod
    def build_stratum_key(self, dataset: pd.DataFrame, stratify_columns: list[str]) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def split(self, dataset: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        raise NotImplementedError


class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class BaseClusterer(ABC):
    @abstractmethod
    def cluster(self, embeddings: np.ndarray, default_k: int) -> np.ndarray:
        raise NotImplementedError


class BaseSemanticSelector(ABC):
    @abstractmethod
    def select(
        self,
        dataset_stratum: pd.DataFrame,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        record_key_column: str,
        center_per_cluster: int,
        outlier_per_cluster: int,
    ) -> dict[str, dict]:
        raise NotImplementedError


class BaseRuleEnricher(ABC):
    @abstractmethod
    def enrich(
        self,
        dataset_stratum: pd.DataFrame,
        text_column: str,
        record_key_column: str,
        max_per_pattern: int,
    ) -> dict[str, dict]:
        raise NotImplementedError


class BaseSampleMerger(ABC):
    @abstractmethod
    def merge_record_dicts(self, *dicts: dict[str, dict]) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def limit_per_group(self, dataset: pd.DataFrame, *group_cols: str, max_rows_per_group: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def deduplicate(
        self,
        dataset: pd.DataFrame,
        text_column: str,
        similarity_threshold: float,
        embedder: BaseEmbedder,
    ) -> pd.DataFrame:
        raise NotImplementedError
