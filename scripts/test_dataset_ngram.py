import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter, defaultdict
from src.dataset_loader import DatasetLoader

domains = DatasetLoader.load_all_domains()
bigrams = defaultdict(Counter)
trigrams = defaultdict(Counter)
unigrams = Counter()

for d_name, texts in domains.items():
    for text in texts:
        clean_text = text.replace("___", "").replace("[blank]", "")
        words = [w.strip(".,!?:;()[]\"'").lower() for w in clean_text.split() if w.strip(".,!?:;()[]\"'")]
        for i, w in enumerate(words):
            unigrams[w] += 1
            if i > 0:
                bigrams[words[i-1]][w] += 1
            if i > 0 and i < len(words) - 1:
                trigrams[(words[i-1], words[i+1])][w] += 1

print(f"Total unique dataset words indexed: {len(unigrams)}")
print(f"Followers of 'operating': {bigrams['operating'].most_common(4)}")
print(f"Followers of 'patient': {bigrams['patient'].most_common(4)}")
print(f"Trigram ('operating', 'by'): {trigrams[('operating', 'by')].most_common(4)}")
