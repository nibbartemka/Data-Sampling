from abc import ABC, abstractmethod

import pandas as pd


__all__ = [
    'BaseDataManager',
    'BaseTextProcessor',
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
