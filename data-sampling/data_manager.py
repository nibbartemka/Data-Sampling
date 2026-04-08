import pandas as pd

from .interfaces import BaseDataManager


class ExcelDataManager(BaseDataManager):
    def load_dataset(
        self,
        path: str,
        required_columns: list[str]
    ) -> pd.DataFrame:
        df = pd.read_excel(path)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"В файле нет колонок: {missing}")
        return df[required_columns].copy()

    def save_dataset(self, path: str, dataset: pd.DataFrame) -> None:
        dataset.to_excel(path, index=False)
