# Adaptive Preprocessing Selection for Domain-Specific NLP Pipelines: A Meta-Learning Approach
## Comprehensive Technical Project Report

**Academic Year:** 2025–2026  
**Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering  
**Project Authors:**
- **Ritesh Sharma** (USN: `ENG25CSE012`)
- **Longjam Nirish Singh** (USN: `ENG25CSE031`)  
**Project Repository:** [https://github.com/Ritesh11042/nlp](https://github.com/Ritesh11042/nlp)

---

## Executive Summary

Natural Language Processing (NLP) models increasingly process heterogeneous and domain-shifted text streams spanning biomedical literature, financial disclosures, and social media interactions. Historically, text preprocessing has been treated as a static, heuristic preliminary chore—routinely applying global lowercasing, punctuation stripping, and stopword removal. 

However, empirical evidence demonstrates that **static preprocessing destroys domain-critical signals**:
1. Applying aggressive social media cleaning ($P_1$) to financial reports wipes out **86% of quantitative and numerical signals** (currency signs, percentages, decimal figures).
2. Applying aggressive cleaning to biomedical records causes a **100% loss of case sensitivity and hyphenated clinical compounds** (e.g., *Sprague-Dawley*, *VEGFR2*, *anti-CTLA4*), irreversibly degrading downstream transformer tokenization and entity recognition.
3. Conversely, preserving all noise in informal conversational streams preserves uninformative artifacts and degrades inference efficiency.

To solve this fundamental trade-off, this project presents the **Adaptive Preprocessing Selection Gateway**—a meta-learning pipeline built around **authentic domain benchmark datasets** that dynamically partitions unlabelled, mixed-domain text streams, extracts a 9-dimensional label-free statistical signature, and routes each micro-cluster to an optimal domain-specific preprocessing policy via a calibrated LightGBM surrogate classifier.

The entire architecture is grounded in real benchmark datasets:
- **BC5CDR:** Biomedical Chemical-Disease Relations corpus (75 records).
- **Financial PhraseBank:** Corporate financial earnings and disclosures (75 records).
- **TweetEval:** Social microblogs with hashtags, user mentions, emojis, and negation cues (65 records).
- **AG News:** Technology, science, and computing reports (35 records).

Furthermore, this project integrates a **Multi-Domain Word Intelligence Engine** featuring:
- **Dataset-Trained Contextual Missing Word Predictor & Restorer** across Social, Financial, and Medical domains (e.g., solving syntactic gaps such as `hi my ___ nitish` &rarr; predicting `name` with dataset provenance attribution and restoring `hi my name is nitish`).
- **Empirical N-Gram Language Model** computing co-occurrence transition frequencies directly from the domain corpus.
- **Precision Right vs. Wrong Word Inspector & Spell Checker** that detects genuine typos while strictly protecting specialized clinical and financial entities from corruption.
- **100+ Unicode & ASCII Emoji/Emoticon Semantic Normalizer**.
- An interactive **7-Tab Streamlit Web Application** with an integrated **Dataset Hub & Corpus Explorer**.

Experimental benchmarks prove that this framework achieves an optimal **Normalized Regret of 0.0001**, a **Spearman rank correlation of 0.738–0.949**, and a **97.7%–99.4% routing confidence** on unseen heterogeneous data streams.

---

## 1. Introduction & Problem Statement

### 1.1 The Preprocessing Dilemma
In modern text pipelines, text normalization is usually applied uniformly before feeding tokens to transformer architectures (e.g., BERT, RoBERTa, DeBERTa). While high-capacity models possess subword tokenizers, preprocessing choices dictate the exact vocabulary representation and sequence structure.

Different domains exhibit diametrically opposed preprocessing requirements:
- **Biomedical Literature:** Highly sensitive to letter casing (*IL-2* cytokine vs *il-2* gene), hyphenated chemical nomenclature (*ischemia-reperfusion*), and Greek/alphanumeric descriptors (*VEGFR2*, *CD4+*).
- **Financial Disclosures:** Critically reliant on numerical precision, directional symbols, percentages, and currencies (*EUR 12.4M*, *+14.5%*, *FY2026*).
- **Social Media & Microblogs:** Cluttered with conversational noise, user mentions (`@handle`), URLs, elongated words (`soooo`), emojis, and emoticons that carry high sentiment charge but obscure lexical representations.

### 1.2 The "One-Size-Fits-All" Failure Mode
When an aggressive cleaning regimen is applied globally across a multi-domain stream:
- Clinical drug names are corrupted or lowercased, blending with common vocabulary.
- Financial figures and currency signs are stripped, turning quantitative statements into ambiguous text.
- Negation polarity is detached, causing sentiment inversions (*"not good"* &rarr; *"good"*).

Table 1.1 highlights the empirical signal destruction observed under static policies:

| Signal / Feature | $P_0$: Minimal Technical | $P_1$: Aggressive Social | $P_2$: Negation Shielded | $P_3$: Biomedical Entity |
| :--- | :---: | :---: | :---: | :---: |
| **Number Retention** | 100% | **14% (86% Lost)** | 100% | 100% |
| **Case Retention** | 100% | **0% (100% Lost)** | 100% | 100% |
| **Hyphen Retention** | 100% | **0% (100% Lost)** | 100% | 100% |
| **Negation Preserved** | Implicit | Lost | **Explicit (`NOT_`)** | Implicit |
| **Emoji Semantic Decoded** | Kept Raw | Stripped | Stripped | **Decoded to Tokens** |

---

## 2. Real Domain Benchmark Datasets (`data/`)

Unlike systems relying on synthetic rules or hand-crafted vectors, this system is built and calibrated directly on real domain benchmark corpora stored in the `data/` directory:

| Dataset Name | Domain | Record Count | Target Policy | Description |
| :--- | :--- | :---: | :---: | :--- |
| **BC5CDR** | Biomedical | 75 | $P_3$ | Chemical-Disease Relations, PubMed oncology, pharmacology, dosages, and clinical trials. |
| **Financial PhraseBank** | Financial | 75 | $P_0$ | Corporate earnings disclosures, quarterly metrics, YoY growth, dividend payouts, net debt. |
| **TweetEval** | Social | 65 | $P_1$ / $P_2$ | Social conversational microblogs, hashtags, user mentions, rich emojis, and negation cues. |
| **AG News** | Technology/Science | 35 | $P_0$ | Hardware, semiconductor, AI, quantum computing, and systems engineering reports. |
| **Mixed Stream Benchmark** | Heterogeneous | 250 | Adaptive | Master interleaved stream combining instances from all four domains. |

---

## 3. System Architecture

The end-to-end processing lifecycle is outlined below:

```
[ Incoming Heterogeneous Text Stream ]
                 │
                 ▼
     ┌───────────────────────┐
     │ Extended Emoji Engine │  <-- Normalizes 100+ Unicode & ASCII Emojis
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ Micro-Batch           │  <-- MiniLM Dense Embedding +
     │ Partitioner           │      Spherical K-Means Clustering (k = 2 to 6)
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ 9-D Local Profiler    │  <-- Label-free Statistical & Linguistic Signature
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ Calibrated LightGBM   │  <-- Trained on 400 Real Dataset Profiles:
     │ Surrogate Router      │      Predicts P0, P1, P2, or P3 (97-99% Confidence)
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ Domain Policy Engine  │  <-- Executes Tailored Transformations
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ Word Intelligence &   │  <-- Empirical N-Gram Language Model,
     │ Dataset Corpus Engine │      Spell Checking & Missing Word Infilling
     └───────────┬───────────┘
                 │
                 ▼
[ Cleaned, Restored, Signal-Preserved Output ]
```

---

## 4. Core Technical Modules & Implementation

### 4.1 Domain Policy Engine (`src/policy_engine.py`)
- **$P_0$ — Minimal / Technical Policy:** Preserves letter casing, punctuation, and numerical tokens; strips URLs and extra spaces.
- **$P_1$ — Aggressive Social Policy:** Strips user mentions (`@user`), hashtags, punctuation, and character elongations; lowercases all tokens.
- **$P_2$ — Negation-Shielded Policy:** Attaches explicit negation scope prefixes (`NOT_happy`, `NOT_acceptable`), preserving polarity.
- **$P_3$ — Biomedical Entity-Preserving Policy:** Decodes emojis into semantic domain tokens (`[EMOJI_DNA]`, `[EMOJI_PILL]`) and strictly shields casing, chemical hyphens, and dosages.

### 4.2 9-Dimensional Label-Free Meta-Profiler (`src/profiler.py`)
1. **Type-Token Ratio (TTR):** Lexical diversity ($\frac{|V|}{N}$).
2. **Uppercase Character Ratio:** Acronym and case density.
3. **Digit Density:** Numerical and financial token frequency.
4. **Punctuation Entropy:** Shannon entropy across punctuation distributions.
5. **POS Entropy:** Grammatical and syntactic diversity.
6. **Mean Dependency Parse Depth:** Structural syntactic complexity.
7. **Out-of-Vocabulary (OOV) Rate:** Proportion of words outside standard vocabulary.
8. **Byte-Pair Encoding (BPE) Fragmentation Ratio:** Subword token splits.
9. **Embedding Space Dispersion:** Vector variance around cluster centroid.

### 4.3 Training LightGBM on Real Dataset Profiles (`src/router.py`)
The surrogate router is trained directly on 400 empirical micro-batch profile vectors extracted from the real datasets:
- 100 Biomedical BC5CDR micro-batches &rarr; labeled $P_3$
- 100 Financial PhraseBank micro-batches &rarr; labeled $P_0$
- 100 TweetEval non-negated social micro-batches &rarr; labeled $P_1$
- 100 TweetEval negated social micro-batches &rarr; labeled $P_2$
The LightGBM booster achieves **100.0% training accuracy** and outputs calibrated softmax probabilities.

---

## 5. Multi-Domain Word Intelligence Engine (`src/word_intelligence.py`)

- **Dataset-Trained N-Gram Language Model (`DatasetCorpusEngine`):** Indexes unigrams, forward bigrams $P(w_i \mid w_{i-1})$, backward bigrams $P(w_i \mid w_{i+1})$, and trigram contexts $P(w_i \mid w_{i-1}, w_{i+1})$ across all 250 dataset records.
- **Missing Word Infilling:** Predicts missing words and attributes dataset provenance:
  - `hi my ___ nitish` &rarr; predicts **`name`** (Provenance: *TweetEval Social Corpus*) &rarr; restores `"hi my name is nitish"`
  - `Operating profit ___ by 18.4%` &rarr; predicts **`increased`** (Provenance: *Financial PhraseBank Corpus*)
  - `Patient was ___ with high-dose Ciprofloxacin` &rarr; predicts **`treated`** (Provenance: *BC5CDR Clinical Corpus*)
- **Dataset-Backed Word Classification:** Cross-references tokens against real dataset occurrences. Specialized words appearing in BC5CDR or Financial PhraseBank are verified and protected from being altered as typos.
- **100+ Emojis & ASCII Emoticons:** Full normalization of faces, sentiment, clinical (`🩺`, `💊`, `💉`, `🧬`), and financial symbols (`💰`, `📈`, `📉`, `💵`).

---

## 6. Experimental Results & Performance Benchmarks

### 6.1 Signal Retention Rates (Table 6.1)

| Benchmark Corpus | $P_0$: Minimal Technical | $P_1$: Aggressive Social | $P_2$: Negation Shielded | $P_3$: Biomedical Entity |
| :--- | :---: | :---: | :---: | :---: |
| **Biomedical (BC5CDR)** | 0.94 | **0.42** | 0.88 | **0.99** |
| **Financial PhraseBank** | **0.98** | **0.54** | 0.91 | 0.95 |
| **Social / Twitter** | 0.72 | **0.96** | 0.89 | 0.76 |

### 6.2 Micro-Batch Router Output on Heterogeneous Stream (Table 6.2)

| Text Item | Content Preview | Assigned Cluster | Selected Policy | Confidence |
| :---: | :--- | :---: | :---: | :---: |
| 1 | *Targeting of VEGFR2 signaling with ramucirumab...* | Cluster 2 | **$P_3$: Biomedical Entity Preserving** | **97.7%** |
| 2 | *The company reported EUR 12.4M in pre-tax operating losses...* | Cluster 0 | **$P_0$: Minimal Technical** | **96.7%** |
| 3 | *no way, this is absolutely unacceptable! never purchasing again...* | Cluster 1 | **$P_2$: Negation Shielded** | **98.4%** |

### 6.3 Downstream Representation Evaluation (Table 6.4)
- **Mean Normalized Regret:** `0.0001` (Near-zero regret against oracle selection).
- **Mean Spearman Rank Correlation:** `0.7379` (Strong rank alignment).

---

## 7. Interactive Streamlit Web Application (7 Tabs)

The web dashboard (`app.py`) provides 7 comprehensive modules:
1. **🧩 Missing Word Detector & Restorer:** Real-time gap infilling with dataset provenance attribution.
2. **🔍 Word Inspector & Spell Checker:** Color-coded Right vs Wrong word breakdown verified against real corpus vocabulary.
3. **🚀 Live Gateway Router:** Micro-clustering, 9-D profiling, and LightGBM policy routing.
4. **📚 Real Benchmark Datasets & Corpus Explorer:** Browse all 250 dataset records, execute full-text corpus search, and 1-click retrain the router.
5. **😀 Extended Emoji & Emoticon Reader:** Real-time emoji normalization and dictionary browser.
6. **🔬 Signal Retention Lab:** Interactive visualization and charts for Table 6.1 retention rates.
7. **📈 Downstream Evaluation & Regret:** Interactive tables and metrics for Normalized Regret and Spearman rank correlation.

---

## 8. Execution Commands & Verification

| Task | Command Line | Description |
| :--- | :--- | :--- |
| **Web UI Dashboard** | `python -m streamlit run app.py` | Launches interactive UI at `http://localhost:8501` |
| **Feature Verification** | `python demo_changes.py` | Runs CLI demo testing datasets, emojis, spell check, and gap infilling |
| **Missing Word Suite** | `python test_missing_words.py` | Runs standalone tests on gaps like `hi my ___ nitish` |
| **Dataset Ingestion** | `python scripts/build_datasets.py` | Generates 250 benchmark records in `data/` |
| **Router Training** | `python scripts/train_router_on_datasets.py` | Trains LightGBM on 400 real dataset profiles |
| **Full Benchmark Suite** | `python main.py` | Executes full thesis evaluation reproducing Tables 6.1–6.5 |

**Live Application:** [http://localhost:8501](http://localhost:8501)  
**GitHub Repository:** [https://github.com/Ritesh11042/nlp](https://github.com/Ritesh11042/nlp)
