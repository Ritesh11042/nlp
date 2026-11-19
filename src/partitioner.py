"""
Micro-Cluster Partitioner
Partitions incoming heterogeneous text batches into semantic-syntactic micro-clusters.
Architecture:
- Dense representations via SentenceTransformer ('all-MiniLM-L6-v2')
- Spherical L2 Normalization
- Spherical K-Means Clustering (k=2 to 6, default k=3)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


class MicroClusterPartitioner:
    """Partitions text streams into spherical semantic clusters using MiniLM embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name, device=self.device)
            except Exception:
                self._embedder = None
        return self._embedder

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Computes dense L2-normalized embeddings for a list of texts."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self.embedder
        if model is not None:
            try:
                raw_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                return normalize(raw_embeddings, norm="l2")
            except Exception:
                pass

        # Robust lightweight fallback using character/word n-gram hashing if sentence-transformers fails
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vec = TfidfVectorizer(max_features=128, analyzer="char_wb", ngram_range=(3, 5))
        X = vec.fit_transform(texts)
        n_comp = min(X.shape[1] - 1, 32)
        if n_comp > 1 and X.shape[0] > n_comp:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            emb = svd.fit_transform(X)
        else:
            emb = X.toarray()
        return normalize(emb, norm="l2")

    def partition(self, texts: List[str], k: int = 3) -> Dict[int, List[Tuple[str, np.ndarray]]]:
        """
        Partitions texts into k micro-clusters using spherical K-Means.
        Returns a dictionary mapping cluster_id -> list of (text, embedding) tuples.
        """
        n_texts = len(texts)
        if n_texts == 0:
            return {}

        embeddings = self.embed_texts(texts)

        # Restrict k to [1, n_texts]
        k_eff = max(1, min(k, n_texts))

        if k_eff == 1:
            return {0: list(zip(texts, embeddings))}

        kmeans = KMeans(n_clusters=k_eff, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)

        clusters: Dict[int, List[Tuple[str, np.ndarray]]] = {i: [] for i in range(k_eff)}
        for text, emb, label in zip(texts, embeddings, cluster_labels):
            clusters[int(label)].append((text, emb))

        return clusters
