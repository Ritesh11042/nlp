"""
Local Meta-Profiler: 9-Dimensional Feature Extractor
Extracts domain-sensitive, label-free text statistics from micro-clusters:
1. Type-Token Ratio (TTR) - Lexical diversity
2. Uppercase Ratio - Acronyms, entities, case preservation signal
3. Digit Density - Numeric and financial content
4. Punctuation Entropy - Shannon entropy of punctuation symbols (informal/social text)
5. POS Entropy - Shannon entropy of Part-of-Speech tags (syntactic variety)
6. Dependency Depth - Mean dependency tree depth to root <= 10 (syntactic complexity)
7. OOV Rate - Out-of-vocabulary rate relative to a core reference vocabulary
8. BPE Fragmentation - Mean BPE/subword tokens per word (vocabulary specialization)
9. Embedding Dispersion - Mean distance to cluster centroid (semantic spread)
"""

import math
import string
from collections import Counter
from typing import List, Optional, Set
import numpy as np


class LocalMetaProfiler:
    """Computes a 9-dimensional statistical meta-vector for a micro-cluster of texts."""

    DEFAULT_REF_VOCAB: Set[str] = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
        "will", "my", "one", "all", "would", "there", "their", "what", "so",
        "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
        "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see",
        "other", "than", "then", "now", "look", "only", "come", "its", "over",
        "think", "also", "back", "after", "use", "two", "how", "our", "work",
        "first", "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "is", "was", "are", "been", "has", "had",
        "were", "said", "did", "having", "may", "should", "does", "being"
    }

    FEATURE_NAMES: List[str] = [
        "ttr",
        "upper_ratio",
        "digit_density",
        "punct_entropy",
        "pos_entropy",
        "dep_depth",
        "oov_rate",
        "bpe_frag",
        "geom_dispersion"
    ]

    def __init__(self, ref_vocab: Optional[Set[str]] = None, bpe_tokenizer=None, spacy_nlp=None):
        self.ref_vocab = ref_vocab or self.DEFAULT_REF_VOCAB
        self._bpe_tokenizer = bpe_tokenizer
        self._spacy_nlp = spacy_nlp

    @property
    def bpe_tokenizer(self):
        if self._bpe_tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._bpe_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            except Exception:
                self._bpe_tokenizer = None
        return self._bpe_tokenizer

    @property
    def spacy_nlp(self):
        if self._spacy_nlp is None:
            try:
                import spacy
                self._spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])
            except Exception:
                self._spacy_nlp = None
        return self._spacy_nlp

    @staticmethod
    def _entropy(counts: Counter) -> float:
        """Calculates base-2 Shannon entropy from a Counter distribution."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

    def extract(self, texts: List[str], cluster_embeddings: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extracts the 9-dimensional routing profile for a micro-cluster.
        Returns a numpy array of shape (9,).
        """
        if not texts:
            return np.zeros(9, dtype=np.float32)

        all_tokens: List[str] = []
        raw_chars = 0
        upper_chars = 0
        digit_chars = 0
        punct_counts: Counter = Counter()
        pos_counts: Counter = Counter()
        dep_depths: List[float] = []
        bpe_token_counts: List[int] = []
        doc_word_counts: List[int] = []

        # 1. Syntactic Parsing (POS tags & dependency depth) on sample of texts (max 25)
        nlp = self.spacy_nlp
        sample_size = min(len(texts), 25)
        if nlp is not None:
            try:
                for doc in nlp.pipe(texts[:sample_size]):
                    for token in doc:
                        pos_counts[token.pos_] += 1
                        depth = 0
                        head = token.head
                        while head != head.head and depth < 10:
                            depth += 1
                            head = head.head
                        dep_depths.append(depth)
            except Exception:
                pass

        # Fallback syntactic heuristics if spaCy is unavailable
        if not pos_counts:
            for text in texts[:sample_size]:
                words = text.split()
                for w in words:
                    cw = w.strip(string.punctuation).lower()
                    if cw in {"in", "on", "at", "by", "for", "with", "from", "to", "of"}:
                        pos_counts["ADP"] += 1
                    elif cw in {"the", "a", "an", "this", "that"}:
                        pos_counts["DET"] += 1
                    elif cw.isdigit() or any(c.isdigit() for c in cw):
                        pos_counts["NUM"] += 1
                    elif cw in self.ref_vocab:
                        pos_counts["VERB" if cw.endswith("ed") or cw.endswith("ing") else "NOUN"] += 1
                    else:
                        pos_counts["PROPN" if any(c.isupper() for c in w) else "NOUN"] += 1
                dep_depths.append(min(len(words) / 4.0, 4.0))

        # 2. Token, character, punctuation, and BPE analysis
        tokenizer = self.bpe_tokenizer
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue

            raw_chars += len(text)
            upper_chars += sum(1 for c in text if c.isupper())
            digit_chars += sum(1 for c in text if c.isdigit())

            for c in text:
                if c in string.punctuation:
                    punct_counts[c] += 1

            words = text.split()
            doc_word_counts.append(len(words))
            all_tokens.extend(words)

            if tokenizer is not None:
                try:
                    bpe_ids = tokenizer.encode(text, truncation=True, max_length=128)
                    bpe_token_counts.append(len(bpe_ids))
                except Exception:
                    bpe_token_counts.append(int(len(words) * 1.3))
            else:
                # Heuristic subword fragmentation based on character length & affixes
                subwords = sum(max(1, len(w) // 5 + (1 if any(c.isupper() for c in w[1:]) or '-' in w else 0)) for w in words)
                bpe_token_counts.append(subwords)

        # 3. Aggregate statistics
        total_words = max(len(all_tokens), 1)
        raw_chars = max(raw_chars, 1)

        # Feature 1: Type-Token Ratio
        ttr = len(set(all_tokens)) / total_words

        # Feature 2: Uppercase Ratio
        upper_ratio = upper_chars / raw_chars

        # Feature 3: Digit Density
        digit_density = digit_chars / raw_chars

        # Feature 4: Punctuation Entropy
        punct_entropy = self._entropy(punct_counts)

        # Feature 5: POS Entropy
        pos_entropy = self._entropy(pos_counts)

        # Feature 6: Mean Dependency Depth
        dep_depth = float(np.mean(dep_depths)) if dep_depths else 1.0

        # Feature 7: Out-of-Vocabulary Rate
        clean_tokens = [w.strip(string.punctuation).lower() for w in all_tokens if w.strip(string.punctuation)]
        oov_words = [w for w in clean_tokens if w not in self.ref_vocab]
        oov_rate = len(oov_words) / max(len(clean_tokens), 1)

        # Feature 8: BPE Fragmentation Ratio
        total_bpe = sum(bpe_token_counts)
        bpe_frag = total_bpe / total_words

        # Feature 9: Geometric Embedding Dispersion (Mean Euclidean distance to cluster centroid)
        geom_disp = 0.0
        if cluster_embeddings is not None and len(cluster_embeddings) > 1:
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            geom_disp = float(np.mean(distances))

        return np.array([
            ttr,
            upper_ratio,
            digit_density,
            punct_entropy,
            pos_entropy,
            dep_depth,
            oov_rate,
            bpe_frag,
            geom_disp
        ], dtype=np.float32)
