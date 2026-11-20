import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lightgbm as lgb
from src.dataset_loader import DatasetLoader
from src.profiler import LocalMetaProfiler

os.makedirs("models", exist_ok=True)
random.seed(42)
np.random.seed(42)

domains = DatasetLoader.load_all_domains()
profiler = LocalMetaProfiler()

X_train = []
y_train = []

# 1. Biomedical (BC5CDR) -> Policy 3 (100 samples)
bio_texts = domains["Biomedical"]
for _ in range(100):
    k = random.choice([1, 1, 2, 3, 4])
    sample = random.sample(bio_texts, min(k, len(bio_texts)))
    X_train.append(profiler.extract(sample))
    y_train.append(3)

# 2. Financial (Financial PhraseBank) -> Policy 0 (100 samples)
fin_texts = domains["Financial"]
for _ in range(100):
    k = random.choice([1, 1, 2, 3, 4])
    sample = random.sample(fin_texts, min(k, len(fin_texts)))
    X_train.append(profiler.extract(sample))
    y_train.append(0)

# 3. Social Media (TweetEval)
soc_texts = domains["Social"]
# Separate social into non-negated (P1) and negated (P2)
neg_keywords = ["not", "never", "cannot", "no", "n't", "barely", "neither", "nor", "won't", "can't"]
soc_neg = [t for t in soc_texts if any(w in t.lower().split() for w in neg_keywords)]
soc_non_neg = [t for t in soc_texts if not any(w in t.lower().split() for w in neg_keywords)]

# Policy 1: Aggressive Social (100 samples)
for _ in range(100):
    k = random.choice([1, 1, 2, 3])
    sample = random.sample(soc_non_neg, min(k, len(soc_non_neg)))
    X_train.append(profiler.extract(sample))
    y_train.append(1)

# Policy 2: Negation Shielded (100 samples)
for _ in range(100):
    k = random.choice([1, 1, 2, 3])
    sample = random.sample(soc_neg, min(k, len(soc_neg)))
    X_train.append(profiler.extract(sample))
    y_train.append(2)

X_mat = np.array(X_train, dtype=np.float32)
y_vec = np.array(y_train, dtype=np.int32)

print(f"Empirical Dataset Matrix: {X_mat.shape}, Classes: {dict(zip(*np.unique(y_vec, return_counts=True)))}")

dtrain = lgb.Dataset(X_mat, label=y_vec)
params = {
    "objective": "multiclass",
    "num_class": 4,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 3,
    "feature_fraction": 0.8,
    "verbose": -1,
    "seed": 42
}

booster = lgb.train(params, dtrain, num_boost_round=80)
booster.save_model("models/surrogate_lgb_router.txt")

# Test on the 3 benchmark sentences
s1 = ["Targeting of VEGFR2 signaling with ramucirumab successfully limited metastatic progression 🧬🔬"]
s2 = ["The company reported EUR 12.4M in pre-tax operating losses due to restructuring costs 📉💳"]
s3 = ["no way, this is absolutely unacceptable! never purchasing again, worst custmer servce ever!! 😡👎 :((( #fail"]

for idx, s in enumerate([s1, s2, s3]):
    p = profiler.extract(s)
    probs = booster.predict(p.reshape(1, -1))[0]
    print(f"Sentence #{idx+1}: prob={probs.round(3)} -> Pred Policy: P{probs.argmax()}")
