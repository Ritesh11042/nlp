from src.word_intelligence import WordIntelligenceEngine

test_cases = [
    "hi my ___ is nitish",
    "hi my ___ nitish",
    "The company ___ losses due to restructuring costs",
    "Operating profit ___ by 18.4% to EUR 42.1 million",
    "Patient was ___ with high-dose Ciprofloxacin",
    "Bilateral renal ___ induced acute tubular necrosis",
    "omg never ___ from this store again!! :("
]

for t in test_cases:
    res = WordIntelligenceEngine.predict_missing_words(t)
    print("=" * 60)
    print("Input:   ", t)
    print("Blank:   ", res["blank"])
    print("Top Word:", res.get("top_prediction"))
    print("Choices: ", [f"{c['word']} ({c['confidence']*100:.0f}%)" for c in res["candidates"]])
    print("Restored:", res["restored_sentence"])
