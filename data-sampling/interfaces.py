from abc import ABC, abstractmethod

import pandas as pd


__all__ = [
    'BaseDataManager',
    'BaseTextProcessor',
    'BaseStratifier',
]


class BaseDataManager(ABC):
    @abstractmethod
    def load_dataset(
        self,
        path: str,
        required_columns: list[str],
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_dataset(
        path: str,
        dataset: pd.DataFrame,
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
        dataset: pd.DataFrame,
        stratify_columns: list[str]
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def split(self, dataset: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        pass
