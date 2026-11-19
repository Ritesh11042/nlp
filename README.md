# Adaptive Preprocessing Selection for Domain-Specific NLP Pipelines: A Meta-Learning Approach

**Final Year Engineering Project**  
**Team Members:**  
- **Ritesh Sharma** (USN: `ENG25CSE012`)  
- **Longjam Nirish Singh** (USN: `ENG25CSE031`)  

---

## 📌 Project Overview
In standard NLP pipelines, text cleaning is treated as an immutable preliminary chore: lowercasing, stripping punctuation, and removing stopwords across the board. However, when heterogeneous text streams (such as biomedical literature, financial earnings reports, and social media tweets) are normalized with a single aggressive policy, vital domain signals are permanently destroyed.

This project delivers an **Adaptive Preprocessing Selection Gateway** with **Word Intelligence & Semantic Restoration**:
1. **Micro-Batch Partitioner:** Embeds text batches using dense representations (`all-MiniLM-L6-v2`) and partitions them into semantic-syntactic micro-clusters via **Spherical K-Means** ($k=2-6$).
2. **Local 9-D Profiler:** Calculates a label-free statistical signature per micro-cluster (TTR, Upper Ratio, Digit Density, Punctuation Entropy, POS Entropy, Dependency Depth, OOV Rate, BPE Fragmentation, and Embedding Dispersion).
3. **Surrogate Policy Router:** A calibrated **LightGBM 4-class classifier** predicts the optimal policy per cluster with high confidence.
4. **Policy Engine:** Executes the selected policy ($P_0$ Minimal, $P_1$ Aggressive Social, $P_2$ Negation Shielded, $P_3$ Biomedical Entity Preserving).
5. **Extended Emoji & Emoticon Reader:** Decodes and normalizes over 80+ emojis and ASCII emoticons (faces, hearts, gestures, clinical symbols, financial tickers) into semantic domain tokens (`[EMOJI_HAPPY]`, `[EMOJI_STETHOSCOPE]`, `[EMOJI_CHART_UP]`, etc.).
6. **Word Intelligence & Domain-Aware Spell Checker:** Intelligently classifies words into `VALID_STANDARD`, `DOMAIN_ENTITY` (protecting terms like `Ciprofloxacin`, `ramucirumab`, `EBITDA` from alteration), and `MISSPELLED` with contextual corrections.
7. **Missing Word Gap Detector & Restorer:** Detects explicit mask placeholders (`<mask=...>`, `[MASK]`) or implicit grammatical gaps and predicts top candidate words.

---

## 📁 Repository Structure
```
nlp/
│
├── src/
│   ├── __init__.py               # Package exports
│   ├── policy_engine.py          # Four interpretable policies (P0 to P3)
│   ├── word_intelligence.py      # Emoji decoder, word classifier, spell check & gap detection
│   ├── profiler.py               # 9-dimensional statistical meta-profiler
│   ├── partitioner.py            # MiniLM dense embedding + Spherical K-Means
│   ├── router.py                 # Calibrated LightGBM surrogate policy router
│   ├── gateway.py                # End-to-end Adaptive Preprocessing Gateway
│   ├── benchmarker.py            # Signal retention and downstream proxy benchmarker
│   └── evaluation.py             # Benchmark suite reproducing Tables 6.1 - 6.5
│
├── models/
│   └── surrogate_lgb_router.txt  # Calibrated LightGBM booster weights
│
├── app.py                        # Interactive Streamlit Web Demonstrator
├── main.py                       # CLI test runner for full benchmark suite
├── Adaptive_Preprocessing_Selection.ipynb # Jupyter notebook (Local & Colab ready)
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Documentation
```

---

## ⚡ Quick Start

### 1. Installation
Ensure Python 3.10+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Research Benchmark Suite
```bash
python main.py
```

### 3. Launch the Interactive Web Demonstrator
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501` to access:
- **Live Gateway Router:** Micro-clustering, 9-D profiling, and dynamic policy execution.
- **Word Inspector & Spell Checker:** Right vs Wrong vs Domain Entity classification with error correction.
- **Missing Word Detector & Restorer:** Missing token gap identification and ranked restoration.
- **Extended Emoji Reader:** Test and decode 80+ emojis and ASCII emoticons.
- **Signal Retention Lab:** Verify preservation rates matching Table 6.1.
- **Downstream Evaluation:** View Normalized Regret and Spearman correlation.
