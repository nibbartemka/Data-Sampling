import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .interfaces import BaseSemanticSelector


__all__ = [
    'SimpleSemanticSelector',
]


class SimpleSemanticSelector(BaseSemanticSelector):
    def select(
        self,
        dataset_stratum: pd.DataFrame,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        id_column: str,
        center_per_cluster: int,
        outlier_per_cluster: int,
    ) -> dict:
        selected = {}

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
                self._append_reason(selected, row, id_column, "semantic_central", float(sim[pos]))

            for pos in order_asc[:outlier_per_cluster]:
                row = dataset_stratum.iloc[local_idx[pos]]
                self._append_reason(selected, row, id_column, "semantic_outlier", float(1 - sim[pos]))

        return selected

    def _append_reason(self, records: dict, row: pd.Series,
                       id_column: str, reason: str, score: float) -> None:
        rid = str(row[id_column])

        if rid not in records:
            record = row.to_dict()
            record["selection_reasons"] = reason
            record["selection_score"] = float(score)
            records[rid] = record
            return

        existing = records[rid]
        reasons = set(str(existing["selection_reasons"]).split("; "))
        reasons.add(reason)
        existing["selection_reasons"] = "; ".join(sorted(r for r in reasons if r))
        existing["selection_score"] = max(float(existing["selection_score"]), float(score))
