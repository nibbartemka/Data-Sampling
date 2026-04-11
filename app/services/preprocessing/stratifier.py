from __future__ import annotations

import pandas as pd

from app.services.preprocessing.interfaces import BaseStratifier


class SimpleStratifier(BaseStratifier):
    def build_stratum_key(self, dataset: pd.DataFrame, stratify_columns: list[str]) -> pd.DataFrame:
        work = dataset.copy()
        for col in stratify_columns:
            work[col] = work[col].fillna("UNKNOWN").astype(str).str.strip()
        work["stratum_key"] = work[stratify_columns].agg(" | ".join, axis=1)
        return work

    def split(self, dataset: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        return list(dataset.groupby("stratum_key", dropna=False))
