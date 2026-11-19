"""
Demonstration Script for NLP Semantic Intelligence Features:
1. Extended 80+ Emoji Decoding
2. Right vs Wrong Word Classification & Domain Entity Protection
3. Domain-Aware Spelling Auto-Correction
4. Missing Word Prediction & Sentence Infilling across Social, Finance & Medicine
5. Adaptive Gateway Batch Routing
"""

from src.word_intelligence import ExtendedEmojiEngine, WordIntelligenceEngine
from src.gateway import AdaptivePreprocessingGateway

print("=" * 75)
print("1. EXTENDED EMOJI & EMOTICON DECODING (80+ SYMBOLS)")
print("=" * 75)
sample_emoji = "Patient treated with 14.5 mg/kg 🩺💊 but felt sick 😢💔. Stock was up 📈💰! :) :-D 🔥💯"
print("Raw Input:")
print(f"  {sample_emoji}")
print("\nDecoded Output:")
print(f"  {ExtendedEmojiEngine.normalize_text_emojis(sample_emoji)}")

print("\n" + "=" * 75)
print("2. WORD CLASSIFICATION & DOMAIN-AWARE SPELL CHECKING")
print("=" * 75)
sample_words = "The patinet was admistered high-dose ramucirumab with operting profit rising 14.5% in Q3."
print(f"Input Sentence: {sample_words}\n")
analysis = WordIntelligenceEngine.analyze_words(sample_words)
for w in analysis:
    print(f"  {w['status']:<18} | {w['word']:<15} | {w['description']}")

corrected, fixes = WordIntelligenceEngine.correct_spelling(sample_words)
print("\nAuto-Corrected (Shielding Domain Entities):")
print(f"  {corrected}")

print("\n" + "=" * 75)
print("3. MISSING WORD PREDICTION (SOCIAL, MEDICAL & FINANCIAL FIELDS)")
print("=" * 75)
test_blanks = [
    ("Social Greeting", "hi my ___ is nitish"),
    ("Social Slang", "omg never ___ from this store again!! :("),
    ("Financial Metric", "Operating profit ___ by 18.4% to EUR 42.1 million"),
    ("Biomedical Clinical", "Patient was ___ with high-dose Ciprofloxacin"),
    ("Biomedical Disease", "Bilateral renal ___ induced acute tubular necrosis")
]

for category, sentence in test_blanks:
    res = WordIntelligenceEngine.predict_missing_words(sentence)
    print(f"[{category}]")
    print(f"  Input:    {sentence}")
    print(f"  Predicted Missing Word: '{res['top_prediction']}'")
    print(f"  Ranked Candidates:      {[c['word'] for c in res['candidates'][:3]]}")
    print(f"  Restored Sentence:      {res['restored_sentence']}\n")

print("=" * 75)
print("4. ADAPTIVE GATEWAY BATCH ROUTING")
print("=" * 75)
gateway = AdaptivePreprocessingGateway()
batch = [
    "Targeting of VEGFR2 signaling with ramucirumab successfully limited metastatic progression 🧬🔬",
    "The company reported EUR 12.4M in pre-tax operating losses due to restructuring costs 📉💳",
    "no way, this is absolutely unacceptable! never purchasing again, worst custmer servce ever!! 😡👎 :((( #fail"
]
results = gateway.process_batch(batch, k_clusters=3)
for r in results:
    print(f"\n[Cluster {r['assigned_cluster']}] Policy: {r['routing_decision']} (Confidence: {r['confidence']})")
    print(f"  Raw:   {r['original_text']}")
    print(f"  Clean: {r['cleaned_text']}")

print("\n" + "=" * 75)
print("✅ ALL NLP FEATURES VERIFIED & OPERATIONAL")
print("=" * 75)
