from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.services.preprocessing.interfaces import BaseSemanticSelector


class SimpleSemanticSelector(BaseSemanticSelector):
    def select(
        self,
        dataset_stratum: pd.DataFrame,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        record_key_column: str,
        center_per_cluster: int,
        outlier_per_cluster: int,
    ) -> dict[str, dict]:
        selected: dict[str, dict] = {}
        for cluster_id in sorted(set(cluster_labels)):
            local_idx = np.where(cluster_labels == cluster_id)[0]
            if len(local_idx) == 0:
                continue
            cluster_vecs = embeddings[local_idx]
            centroid = cluster_vecs.mean(axis=0, keepdims=True)
            sim = cosine_similarity(cluster_vecs, centroid).reshape(-1)
            order_desc = np.argsort(-sim)
            order_asc = np.argsort(sim)

            for pos in order_desc[:center_per_cluster]:
                row = dataset_stratum.iloc[local_idx[pos]]
                self._append_reason(selected, row, record_key_column, "semantic_central", float(sim[pos]))

            for pos in order_asc[:outlier_per_cluster]:
                row = dataset_stratum.iloc[local_idx[pos]]
                self._append_reason(selected, row, record_key_column, "semantic_outlier", float(1 - sim[pos]))
        return selected

    @staticmethod
    def _append_reason(records: dict[str, dict], row: pd.Series, record_key_column: str, reason: str, score: float) -> None:
        rid = str(row[record_key_column])
        if rid not in records:
            record = row.to_dict()
            record["selection_reasons"] = reason
            record["semantic_selection_score"] = float(score)
            records[rid] = record
            return
        existing = records[rid]
        reasons = set(str(existing["selection_reasons"]).split("; "))
        reasons.add(reason)
        existing["selection_reasons"] = "; ".join(sorted(r for r in reasons if r))
        existing["semantic_selection_score"] = max(float(existing.get("semantic_selection_score", 0)), float(score))
