"""
Adaptive Preprocessing Gateway
End-to-end gateway orchestrating:
1. Batch partitioning via MiniLM embeddings + Spherical K-Means
2. Local 9-D statistical meta-profiling per micro-cluster
3. Feature alignment & surrogate policy routing via LightGBM
4. Policy execution via DomainPolicyEngine
"""

import os
from typing import Any, Dict, List, Optional
import numpy as np

from .policy_engine import DomainPolicyEngine, POLICY_NAMES, POLICY_DESCRIPTIONS
from .profiler import LocalMetaProfiler
from .partitioner import MicroClusterPartitioner
from .router import LightGBMSurrogateRouter


class AdaptivePreprocessingGateway:
    """End-to-end adaptive preprocessing gateway for heterogeneous text streams."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedder_name: str = "all-MiniLM-L6-v2",
        enable_alignment: bool = True
    ):
        print("[Gateway] Initializing Adaptive Preprocessing Gateway...")
        self.enable_alignment = enable_alignment
        self.partitioner = MicroClusterPartitioner(model_name=embedder_name)
        self.profiler = LocalMetaProfiler()
        self.router = LightGBMSurrogateRouter(model_path=model_path)
        print("[Gateway] Gateway ready for inference.")

    def process_batch(self, texts: List[str], k_clusters: int = 3) -> List[Dict[str, Any]]:
        """
        Processes a heterogeneous batch of texts through:
        Partition -> Profile -> Route -> Clean.

        Returns a list of result dictionaries corresponding to input order.
        """
        if not texts:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)

        # Track multiple identical texts correctly with index queues
        text_indices: Dict[str, List[int]] = {}
        for idx, t in enumerate(texts):
            text_indices.setdefault(t, []).append(idx)

        # 1. Partition stream into semantic micro-clusters
        clusters = self.partitioner.partition(texts, k=k_clusters)

        for cluster_id, items in clusters.items():
            if not items:
                continue

            c_texts = [item[0] for item in items]
            c_embeddings = np.array([item[1] for item in items])

            # 2. Extract 9-D meta-vector profile
            raw_meta_vector = self.profiler.extract(c_texts, c_embeddings)

            # 3. Align and query LightGBM surrogate policy router
            aligned_vector = self.router.align_features(raw_meta_vector, enable_alignment=self.enable_alignment)
            policy_id, probs, confidence = self.router.predict(raw_meta_vector, align=self.enable_alignment)
            policy_name = POLICY_NAMES[policy_id]

            # 4. Clean each text under the predicted policy
            for t in c_texts:
                cleaned_t = DomainPolicyEngine.apply_policy(t, policy_id)
                target_idx = text_indices[t].pop(0)

                results[target_idx] = {
                    "original_text": t,
                    "cleaned_text": cleaned_t,
                    "assigned_cluster": int(cluster_id),
                    "cluster_size": len(c_texts),
                    "policy_id": int(policy_id),
                    "routing_decision": policy_name,
                    "policy_description": POLICY_DESCRIPTIONS[policy_id],
                    "confidence": f"{confidence * 100:.1f}%",
                    "confidence_score": float(confidence),
                    "class_probabilities": {POLICY_NAMES[i]: float(probs[i]) for i in range(4)},
                    "raw_profile": [round(float(v), 4) for v in raw_meta_vector],
                    "aligned_profile": [round(float(v), 4) for v in aligned_vector],
                    "feature_names": self.profiler.FEATURE_NAMES
                }

        # Safe fallback for any unprocessed items
        for idx, r in enumerate(results):
            if r is None:
                orig = texts[idx]
                results[idx] = {
                    "original_text": orig,
                    "cleaned_text": DomainPolicyEngine.apply_policy(orig, 0),
                    "assigned_cluster": 0,
                    "cluster_size": 1,
                    "policy_id": 0,
                    "routing_decision": POLICY_NAMES[0],
                    "policy_description": POLICY_DESCRIPTIONS[0],
                    "confidence": "50.0%",
                    "confidence_score": 0.5,
                    "class_probabilities": {POLICY_NAMES[i]: 0.25 for i in range(4)},
                    "raw_profile": [0.0] * 9,
                    "aligned_profile": [0.0] * 9,
                    "feature_names": self.profiler.FEATURE_NAMES
                }

        return results
