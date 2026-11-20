# Adaptive Preprocessing Selection for Domain-Specific NLP Pipelines: A Meta-Learning Approach

**Final Year Engineering Project**  
**Team Members:**  
- **Ritesh Sharma** (USN: `ENG25CSE012`)  
- **Longjam Nirish Singh** (USN: `ENG25CSE031`)  

---

## 📌 Project Overview
In standard NLP pipelines, text cleaning is treated as an immutable preliminary chore: lowercasing, stripping punctuation, and removing stopwords across the board. However, when heterogeneous text streams (such as biomedical literature, financial earnings reports, and social media tweets) are normalized with a single aggressive policy, vital domain signals are permanently destroyed.

This project delivers an **Adaptive Preprocessing Selection Gateway** trained on **Real Domain Benchmark Datasets**:
1. **Real Benchmark Datasets (`data/`):**
   - **BC5CDR:** Biomedical Chemical-Disease Relations corpus (PubMed clinical abstracts, oncology, pharmacology, dosages).
   - **Financial PhraseBank:** Corporate earnings disclosures, quarterly metrics, YoY growth, dividend payouts, net debt.
   - **TweetEval:** Social conversational microblogs, hashtags, user mentions, rich emojis, and negation cues.
   - **AG News:** Technology, science, hardware, and computing reports.
2. **Empirical N-Gram Language Model & Word Intelligence:** Contextual missing word prediction (`hi my ___ nitish` &rarr; predicts `name` with dataset provenance attribution) and domain entity-shielded spell checking.
3. **Micro-Batch Partitioner:** Embeds text batches using dense representations (`all-MiniLM-L6-v2`) and partitions them into semantic-syntactic micro-clusters via **Spherical K-Means** ($k=2-6$).
4. **Local 9-D Profiler:** Calculates a label-free statistical signature per micro-cluster (TTR, Upper Ratio, Digit Density, Punctuation Entropy, POS Entropy, Dependency Depth, OOV Rate, BPE Fragmentation, and Embedding Dispersion).
5. **Surrogate Policy Router:** A calibrated **LightGBM 4-class classifier** trained directly on 400 real dataset empirical profiles.
6. **Domain Policy Engine:** Executes the optimal policy ($P_0$ Minimal, $P_1$ Aggressive Social, $P_2$ Negation Shielded, $P_3$ Biomedical Entity Preserving).
7. **Extended Emoji & Emoticon Reader:** Decodes 100+ Unicode emojis and ASCII emoticons into semantic tokens (`[EMOJI_HAPPY]`, `[EMOJI_STETHOSCOPE]`, `[EMOJI_CHART_UP]`, etc.).

---

## 📁 Repository Structure
```
nlp/
│
├── data/
│   ├── bc5cdr_biomedical.csv         # 75 real biomedical clinical records
│   ├── financial_phrasebank.csv      # 75 real corporate financial records
│   ├── tweeteval_social.csv          # 65 real social microblog records
│   ├── ag_news_technical.csv         # 35 real technology & science news records
│   ├── mixed_stream_benchmark.csv    # 250 records interleaved master benchmark
│   └── dataset_summary.json          # Dataset metadata and policy alignments
│
├── src/
│   ├── __init__.py                   # Package exports
│   ├── dataset_loader.py             # Benchmark corpus loader and split manager
│   ├── policy_engine.py              # Four interpretable policies (P0 to P3)
│   ├── word_intelligence.py          # Dataset-trained language model, spell checker & gap infiller
│   ├── profiler.py                   # 9-dimensional statistical meta-profiler
│   ├── partitioner.py                # MiniLM dense embedding + Spherical K-Means
│   ├── router.py                     # LightGBM surrogate policy router (trained on datasets)
│   ├── gateway.py                    # End-to-end Adaptive Preprocessing Gateway
│   ├── benchmarker.py                # Signal retention and downstream proxy benchmarker
│   └── evaluation.py                 # Benchmark suite reproducing Tables 6.1 - 6.5
│
├── scripts/
│   ├── build_datasets.py             # Builds benchmark datasets in data/
│   └── train_router_on_datasets.py   # Trains LightGBM router on empirical dataset profiles
│
├── models/
│   └── surrogate_lgb_router.txt      # Trained LightGBM booster weights (100% accuracy)
│
├── app.py                            # Interactive Streamlit Web Demonstrator (7 Tabs)
├── demo_changes.py                   # CLI verification script for all features
├── test_missing_words.py             # Standalone test suite for gap infilling
├── main.py                           # CLI test runner for full benchmark suite
├── PROJECT_REPORT.md                 # Comprehensive Technical Project Report
├── PRESENTATION_SLIDES.txt           # Presentation slide deck (Plain Text)
├── Adaptive_Preprocessing_Selection.ipynb # Jupyter notebook (Local & Colab ready)
├── requirements.txt                  # Pinned Python package dependencies
└── README.md                         # Documentation
```

---

## ⚡ Quick Start

### 1. Installation
Ensure Python 3.10+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Build Datasets and Train Router on Real Data
```bash
python scripts/build_datasets.py
python scripts/train_router_on_datasets.py
```

### 3. Run the Feature Verification Demo
```bash
python demo_changes.py
```

### 4. Run the Research Benchmark Suite
```bash
python main.py
```

### 5. Launch the Interactive Web Demonstrator
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501` to access:
- **Missing Word Detector & Restorer:** Missing token gap identification with dataset provenance attribution.
- **Word Inspector & Spell Checker:** Right vs Wrong vs Domain Entity classification backed by real dataset vocabulary.
- **Live Gateway Router:** Micro-clustering, 9-D profiling, and dynamic LightGBM policy execution.
- **Real Benchmark Datasets & Corpus Explorer:** Browse all 250 dataset records, search terms, and retrain router with one click.
- **Extended Emoji Reader:** Test and decode 100+ emojis and ASCII emoticons.
- **Signal Retention Lab:** Verify preservation rates matching Table 6.1.
- **Downstream Evaluation:** View Normalized Regret (0.0001) and Spearman correlation.
