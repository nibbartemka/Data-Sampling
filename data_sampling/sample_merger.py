import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .interfaces import BaseSampleMerger


__all__ = [
    'SimpleSampleMerger',
]


class SimpleSampleMerger(BaseSampleMerger):
    def merge_record_dicts(self, *dicts: dict) -> pd.DataFrame:
        merged = {}

        for d in dicts:
            for rid, record in d.items():
                if rid not in merged:
                    merged[rid] = record.copy()
                    continue

                existing = merged[rid]
                existing_reasons = set(str(existing["selection_reasons"]).split("; "))
                new_reasons = set(str(record["selection_reasons"]).split("; "))

                existing["selection_reasons"] = "; ".join(
                    sorted(r for r in existing_reasons.union(new_reasons) if r)
                )
                existing["selection_score"] = max(
                    float(existing["selection_score"]),
                    float(record["selection_score"])
                )

        if not merged:
            return pd.DataFrame()

        return pd.DataFrame(list(merged.values()))

    def limit_per_group(
        self,
        dataset: pd.DataFrame,
        *group_cols,
        max_rows_per_group: int
    ) -> pd.DataFrame:
        if dataset.empty:
            return dataset

        if not group_cols:
            raise ValueError("Не переданы колонки для группировки")

        missing_cols = [col for col in group_cols if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"В датасете отсутствуют колонки для группировки: {missing_cols}")

        work = dataset.copy()

        work["reason_count"] = work["selection_reasons"].map(
            lambda x: len(str(x).split("; "))
        )

        work = work.sort_values(
            by=["reason_count", "selection_score"],
            ascending=[False, False]
        )

        parts = []
        for _, group in work.groupby(list(group_cols), dropna=False):
            parts.append(group.head(max_rows_per_group))

        result = pd.concat(parts, ignore_index=True)
        return result.drop(columns=["reason_count"], errors="ignore")

    def deduplicate(
        self,
        dataset: pd.DataFrame,
        text_column: str,
        similarity_threshold: float,
        embedder,
    ) -> pd.DataFrame:
        if dataset.empty or len(dataset) <= 1:
            return dataset.copy()

        work = dataset.copy().reset_index(drop=True)

        work["reason_count"] = work["selection_reasons"].map(
            lambda x: len(str(x).split("; "))
        )

        work = work.sort_values(
            by=["reason_count", "selection_score"],
            ascending=[False, False]
        ).reset_index(drop=True)

        texts = work[text_column].tolist()
        emb = embedder.encode(texts)
        sim_matrix = cosine_similarity(emb)

        keep = np.ones(len(work), dtype=bool)

        for i in range(len(work)):
            if not keep[i]:
                continue

            for j in range(i + 1, len(work)):
                if not keep[j]:
                    continue

                if sim_matrix[i, j] >= similarity_threshold:
                    keep[j] = False

        result = work[keep].copy()
        return result.drop(columns=["reason_count"], errors="ignore").reset_index(drop=True)