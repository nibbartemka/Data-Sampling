import pandas as pd

from .interfaces import BaseStratifier


class SimpleStratifier(BaseStratifier):
    def build_stratum_key(
        self,
        dataset: pd.DataFrame,
        stratify_columns: list[str]
    ) -> pd.DataFrame:
        dataset = dataset.copy()

        for col in stratify_columns:
            dataset[col] = (dataset[col]
                            .fillna("UNKNOWN")
                            .astype(str).str.strip())

        # Берем ключи стратификации и обхединяем через |
        dataset["stratum_key"] = dataset[stratify_columns].agg(" | ".join, axis=1)
        return dataset

    def split(self, dataset: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        return list(dataset.groupby("stratum_key", dropna=False))
