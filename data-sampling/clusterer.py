import numpy as np
from sklearn.cluster import KMeans

from interfaces import BaseClusterer


class KMeansClusterer(BaseClusterer):
    def _choose_cluster_count(self, n_texts: int, default_k: int) -> int:
        if n_texts <= 3:
            return 1
        if n_texts <= 10:
            return min(2, n_texts)
        if n_texts <= 25:
            return min(3, n_texts)
        return min(default_k, max(2, n_texts // 8))

    def cluster(
        self,
        embeddings: np.ndarray,
        default_k: int
    ) -> np.ndarray:
        n_texts = len(embeddings)
        n_clusters = self._choose_cluster_count(n_texts, default_k)

        if n_clusters <= 1:
            return np.zeros(n_texts, dtype=int)

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        return model.fit_predict(embeddings)
