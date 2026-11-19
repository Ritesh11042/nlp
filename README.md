# Adaptive Preprocessing Selection for Domain-Specific NLP Pipelines: A Meta-Learning Approach

**Final Year Engineering Project**  
**Team Members:**  
- **Ritesh Sharma** (USN: `ENG25CSE012`)  
- **Longjam Nirish Singh** (USN: `ENG25CSE031`)  

---

## 📌 Project Overview
In standard NLP pipelines, text cleaning is treated as an immutable preliminary chore: lowercasing, stripping punctuation, and removing stopwords across the board. However, when heterogeneous text streams (such as biomedical literature, financial earnings reports, and social media tweets) are normalized with a single aggressive policy, vital domain signals are permanently destroyed.

This project delivers an **Adaptive Preprocessing Selection Gateway** using **Meta-Learning**:
1. **Micro-Batch Partitioner:** Embeds text batches using dense representations (`all-MiniLM-L6-v2`) and partitions them into semantic-syntactic micro-clusters via **Spherical K-Means** ($k=2-6$).
2. **Local 9-D Profiler:** Calculates a label-free statistical signature per micro-cluster:
   - Type-Token Ratio (TTR)
   - Uppercase Ratio (Acronyms, Case)
   - Digit Density (Financial metrics, decimals, percentages)
   - Punctuation Entropy (Informal text structure)
   - POS Entropy (Syntactic variety)
   - Dependency Depth (Syntactic complexity)
   - Out-of-Vocabulary (OOV) Rate
   - Subword / BPE Fragmentation
   - Geometric Embedding Dispersion
3. **Surrogate Policy Router:** A calibrated **LightGBM 4-class classifier** predicts the optimal policy per cluster with high confidence.
4. **Policy Engine:** Executes the selected policy:
   - **$P_0$ Minimal / Technical:** Whitespace normalization only. Case, numbers, and symbols preserved.
   - **$P_1$ Aggressive Social:** Lowercases, strips noise and handles (`@user`), removes punctuation.
   - **$P_2$ Negation Shielded:** Scopes negation markers (`not`, `never`, `no`, `n't`) with `NOT_` prefix to protect polarity.
   - **$P_3$ Biomedical Entity Preserving:** Protects hyphens (`Sprague-Dawley`, `ischemia-reperfusion`), gene/drug acronyms (`VEGFR2`), units, and chemical delimiters.

---

## 📁 Repository Structure
```
nlp project/
│
├── src/
│   ├── __init__.py               # Package exports
│   ├── policy_engine.py          # Four interpretable policies (P0 to P3)
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
Executes all experiments from the project report (Signal Retention, Stream Routing, Downstream Regret):
```bash
python main.py
```

### 3. Launch the Interactive Web Demonstrator
Start the Streamlit application:
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore:
- **Live Gateway Router:** Route custom or preset batches with live micro-clustering and cleaning.
- **Signal Retention Lab:** Test text and inspect signal preservation rates across all 4 policies.
- **9-D Profiler Visualizer:** Inspect extracted meta-profiles and alignment rules.
- **Downstream Evaluation:** View normalized regret and Spearman rank correlation.
- **Missing-Word Restoration:** Test masked token restoration.

---

## 📊 Key Results (From Project Report)

### Table 6.1: Signal Retention Rate by Policy
| Signal (Domain) | $P_0$ Minimal | $P_1$ Aggressive Social | $P_2$ Negation Shield | $P_3$ Biomedical Entity |
| :--- | :---: | :---: | :---: | :---: |
| **Numeric integrity (Finance)** | **1.00** | 0.14 *(86% lost)* | 0.86 | **1.00** |
| **Case retention (Biomedical)** | **1.00** | 0.00 *(100% lost)* | 0.00 | **1.00** |
| **Hyphen-entity integrity (Biomedical)** | **1.00** | 0.00 *(100% lost)* | **1.00** | **1.00** |
| **Negation scope marked (Social)** | 0.00 | 0.00 | **1.00** | 0.00 |
| **@-handle removal (Social)** | 0.00 | **1.00** | 0.00 | 0.00 |

### Table 6.2: Gateway Routing Decisions on Heterogeneous Stream
| Input Domain | Sample Text Snippet | Assigned Cluster | Policy Chosen | Confidence |
| :--- | :--- | :---: | :--- | :---: |
| **Biomedical** | *Targeting of VEGFR2 signaling with ramucirumab...* | Cluster 2 | **$P_3$ Biomedical Entity** | **99.9%** |
| **Financial** | *The company reported EUR 12.4M in pre-tax losses...* | Cluster 0 | **$P_0$ Minimal / Technical** | **99.9%** |
| **Social** | *no way, this is absolutely unacceptable! never...* | Cluster 1 | **$P_1$ Aggressive Social** | **99.9%** |
