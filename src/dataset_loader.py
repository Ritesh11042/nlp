"""
Dataset Loader and Benchmark Corpus Manager
Provides programmatic access to the real domain benchmark datasets:
- BC5CDR (Biomedical)
- Financial PhraseBank (Financial)
- TweetEval (Social)
- AG News (Technical/News)
- Mixed Heterogeneous Stream
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class DatasetLoader:
    """Manages loading, filtering, and batch generation from real domain datasets."""

    @staticmethod
    def get_summary() -> Dict[str, Any]:
        """Loads metadata summary of all available datasets."""
        summary_path = os.path.join(DATA_DIR, "dataset_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"total_records": 0, "datasets": {}}

    @staticmethod
    def load_dataset(name: str) -> pd.DataFrame:
        """Loads a specific dataset by name (bc5cdr, financial, tweeteval, ag_news, mixed)."""
        name_lower = name.lower()
        if "bio" in name_lower or "bc5" in name_lower:
            path = os.path.join(DATA_DIR, "bc5cdr_biomedical.csv")
        elif "fin" in name_lower or "phrase" in name_lower:
            path = os.path.join(DATA_DIR, "financial_phrasebank.csv")
        elif "soc" in name_lower or "tweet" in name_lower:
            path = os.path.join(DATA_DIR, "tweeteval_social.csv")
        elif "news" in name_lower or "tech" in name_lower or "ag" in name_lower:
            path = os.path.join(DATA_DIR, "ag_news_technical.csv")
        else:
            path = os.path.join(DATA_DIR, "mixed_stream_benchmark.csv")

        if os.path.exists(path):
            return pd.read_csv(path)
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    @classmethod
    def load_all_domains(cls) -> Dict[str, List[str]]:
        """Loads raw text lists organized by domain name."""
        return {
            "Biomedical": cls.load_dataset("bc5cdr")["text"].tolist(),
            "Financial": cls.load_dataset("financial")["text"].tolist(),
            "Social": cls.load_dataset("tweeteval")["text"].tolist(),
            "Technology": cls.load_dataset("ag_news")["text"].tolist(),
        }

    @classmethod
    def get_training_samples(cls) -> List[Tuple[str, int]]:
        """Returns list of (text, target_policy_id) for router training."""
        mixed_df = cls.load_dataset("mixed")
        return list(zip(mixed_df["text"].tolist(), mixed_df["target_policy"].tolist()))
