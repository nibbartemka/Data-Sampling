from abc import ABC, abstractmethod

import pandas as pd


__all__ = [
    'BaseDataManager'
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
