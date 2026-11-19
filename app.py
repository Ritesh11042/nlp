"""
Adaptive NLP Preprocessing Selection: Local Meta-Learning Gateway
Interactive Streamlit Application

Authors: Ritesh Sharma (ENG25CSE012), Longjam Nirish Singh (ENG25CSE031)
"""

import sys
import os
import re
import string
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.policy_engine import DomainPolicyEngine, POLICY_NAMES, POLICY_DESCRIPTIONS
from src.profiler import LocalMetaProfiler
from src.partitioner import MicroClusterPartitioner
from src.router import LightGBMSurrogateRouter
from src.gateway import AdaptivePreprocessingGateway
from src.benchmarker import SignalRetentionAnalyzer, DownstreamTransformerBenchmarker
from src.word_intelligence import ExtendedEmojiEngine, WordIntelligenceEngine


st.set_page_config(
    page_title="Adaptive NLP Preprocessing & Intelligence Gateway",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    .prediction-box {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 16px;
        font-size: 1.25rem;
        font-weight: 600;
        color: #065F46;
        text-align: center;
        margin: 10px 0;
    }
    .policy-tag-0 { background-color: #E0E7FF; color: #3730A3; padding: 4px 10px; border-radius: 4px; font-weight: 600; }
    .policy-tag-1 { background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 4px; font-weight: 600; }
    .policy-tag-2 { background-color: #D1FAE5; color: #065F46; padding: 4px 10px; border-radius: 4px; font-weight: 600; }
    .policy-tag-3 { background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_gateway():
    return AdaptivePreprocessingGateway()


gateway = load_gateway()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
st.sidebar.title("Adaptive NLP Gateway")
st.sidebar.markdown("**Project Authors:**\n- Ritesh Sharma (ENG25CSE012)\n- Longjam Nirish Singh (ENG25CSE031)")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    [
        "🧩 Missing Word Detector & Restorer",
        "🔍 Word Inspector & Spell Checker",
        "🚀 Live Gateway Router",
        "😀 Extended Emoji & Emoticon Reader",
        "🔬 Signal Retention Lab",
        "📈 Downstream Evaluation & Regret"
    ]
)

# ==============================================================================
# TAB 1: MISSING WORD DETECTOR & RESTORER
# ==============================================================================
if app_mode == "🧩 Missing Word Detector & Restorer":
    st.markdown("<div class='main-title'>🧩 Universal Missing Word Predictor & Restorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Understands sentence context across Social, Financial, and Medical fields. Enter any sentence with blanks (e.g. <code>___</code>, <code>[blank]</code>, or <code>&lt;mask&gt;</code>) to predict missing words!</div>", unsafe_allow_html=True)

    gap_presets = {
        "Social: Name Greeting": "hi my ___ is nitish",
        "Social: Informal Slang": "omg never ___ from this store again, terrible service!! :(",
        "Social: Question": "how ___ you doing today?",
        "Financial: Profit Metric": "Operating profit ___ by 18.4% to EUR 42.1 million in Q3.",
        "Financial: Corporate Trajectory": "The company ___ pre-tax losses due to restructuring costs.",
        "Biomedical: Drug Administration": "Patient was ___ with high-dose Ciprofloxacin.",
        "Biomedical: Clinical Pathology": "Bilateral renal ___ induced acute tubular necrosis in Sprague-Dawley models.",
        "Biomedical: Biomarker Test": "Elevated levels of serum ___ indicated acute renal damage."
    }

    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        selected_gap = st.selectbox("Choose a scenario or type custom text below:", list(gap_presets.keys()))
        default_val = gap_presets[selected_gap]
    with col_p2:
        st.info("💡 **Tip:** Use `___` (3 underscores) anywhere in a sentence to indicate the missing word position!")

    input_gap_text = st.text_area("Input sentence with missing word:", default_val, height=85)

    if st.button("🚀 Predict Missing Word & Restore Sentence", type="primary"):
        res = WordIntelligenceEngine.predict_missing_words(input_gap_text)

        if res["detected"]:
            top_word = res["top_prediction"]

            st.markdown(f"<div class='prediction-box'>✨ Predicted Missing Word: <u>{top_word}</u></div>", unsafe_allow_html=True)

            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                st.markdown("#### 🎯 Top Contextual Candidates & Confidence")
                cand_records = []
                for idx, c in enumerate(res["candidates"]):
                    cand_records.append({
                        "Rank": f"#{idx+1}",
                        "Candidate Word": c["word"],
                        "Confidence Score": f"{c['confidence'] * 100:.1f}%",
                        "Probability Bar": c["confidence"]
                    })
                df_cands = pd.DataFrame(cand_records)
                st.dataframe(df_cands[["Rank", "Candidate Word", "Confidence Score"]], use_container_width=True)

            with col_res2:
                st.markdown("#### 📄 Restored Complete Sentence")
                st.success(res["restored_sentence"])
                st.caption(f"Context detected: Left = `'{res['left_context']}'` | Right = `'{res['right_context']}'`")
        else:
            st.warning("No missing blank pattern detected. Please include `___`, `[blank]`, or `<mask...>` in your sentence.")

# ==============================================================================
# TAB 2: WORD INSPECTOR & SPELL CHECKER (RIGHT VS WRONG)
# ==============================================================================
elif app_mode == "🔍 Word Inspector & Spell Checker":
    st.markdown("<div class='main-title'>🔍 Word Inspector & Domain-Aware Spell Checker</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Differentiates Valid Words, Protected Domain Entities, and Genuine Spelling Mistakes across Healthcare, Finance, and Social text.</div>", unsafe_allow_html=True)

    sample_spell_presets = [
        "The patinet was admistered 14.5 mg/kg Ciprofloxacin following acute onset of severe hepatotoxicity.",
        "Consolidated operting profit decrased 4% due to higher restructring costs and headwinds in FY2026.",
        "hi my name is nitish and i had an absolutly terible custmer servce experence!! :((( #fail",
        "Targeting of VEGFR2 with ramucirumab was evaluated in Sprague-Dawley rat models.",
        "Custom text entry..."
    ]

    choice_spell = st.selectbox("Select example sentence or choose custom:", sample_spell_presets)

    if choice_spell == "Custom text entry...":
        eval_text = st.text_area("Enter your custom text to analyze:", "The patinet recieved high-dose ramucirumab with operting profit rising 14.5% in Q3.")
    else:
        eval_text = st.text_area("Input sentence:", choice_spell)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 🔬 Word Classification Breakdown")
        words_data = WordIntelligenceEngine.analyze_words(eval_text)

        classified_records = []
        for w in words_data:
            st_color = "VALID STANDARD"
            if w["status"] == "DOMAIN_ENTITY":
                st_color = "DOMAIN ENTITY (PROTECTED)"
            elif w["status"] == "MISSPELLED":
                st_color = "MISSPELLED (ERROR)"
            elif w["status"] == "NUMERIC_DATA":
                st_color = "NUMERIC METRIC"
            elif w["status"] == "EMOJI_TOKEN":
                st_color = "EMOJI TOKEN"
            elif w["status"] == "MISSING_BLANK":
                st_color = "MISSING BLANK"

            classified_records.append({
                "Word": w["word"],
                "Classification": st_color,
                "Suggestion / Meaning": w["suggestion"] or w.get("domain") or "—",
                "Explanation": w["description"]
            })

        st.dataframe(pd.DataFrame(classified_records), use_container_width=True)

    with col2:
        st.markdown("#### ✨ Intelligent Auto-Correction")
        corrected_str, corrections = WordIntelligenceEngine.correct_spelling(eval_text)

        if corrections:
            st.warning(f"Detected {len(corrections)} spelling errors:")
            for c in corrections:
                st.write(f"- 🔴 `{c['original']}` ➔ 🟢 **`{c['corrected']}`**")
            st.markdown("**Corrected Text (Entities Intact):**")
            st.success(corrected_str)
        else:
            st.success("✅ No spelling mistakes found! All words are valid vocabulary or protected domain entities.")

        st.markdown("---")
        st.markdown("#### 📖 Domain Semantic Glossary Lookup")
        glossary_lookup = st.text_input("Look up definition of any domain term:", "ramucirumab")
        if glossary_lookup:
            info = WordIntelligenceEngine.get_domain_glossary_lookup(glossary_lookup)
            if info:
                st.info(f"**[{info['domain']}] `{glossary_lookup}`**\n\n{info['meaning']}")
            else:
                st.caption(f"No entry found in built-in domain dictionary for '{glossary_lookup}'.")

# ==============================================================================
# TAB 3: LIVE GATEWAY ROUTER
# ==============================================================================
elif app_mode == "🚀 Live Gateway Router":
    st.markdown("<div class='main-title'>⚡ Adaptive Preprocessing Gateway</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Partition mixed-domain text into micro-clusters, decode 80+ emojis, profile label-free statistical signatures, and route domain-adaptive preprocessing via calibrated LightGBM.</div>", unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([3, 1])

    with col_ctrl1:
        preset_choice = st.selectbox(
            "Select Preset Stream or Custom Input:",
            [
                "Mixed Heterogeneous Stream (Biomedical + Financial + Social)",
                "Biomedical Clinical Stream (BC5CDR style)",
                "Financial Earnings Stream (Financial PhraseBank style)",
                "Social Media Stream with Rich Emojis & Slang",
                "Custom Text Stream"
            ]
        )

    with col_ctrl2:
        k_val = st.slider("Micro-clusters (k):", min_value=1, max_value=5, value=3)

    if preset_choice == "Mixed Heterogeneous Stream (Biomedical + Financial + Social)":
        default_text = "\n".join([
            "Targeting of VEGFR2 signaling with ramucirumab successfully limited metastatic progression in Sprague-Dawley models 🧬🔬",
            "The company reported EUR 12.4M in pre-tax operating losses due to unexpected restructuring costs in Q3 📉💳",
            "no way, this is absolutely unacceptable! never purchasing again, worst custmer servce ever!! 😡👎 :((( #fail"
        ])
    elif preset_choice == "Biomedical Clinical Stream (BC5CDR style)":
        default_text = "\n".join([
            "Bilateral renal ischemia-reperfusion induced acute tubular necrosis in Sprague-Dawley models 🩺",
            "Ciprofloxacin administration was terminated following acute onset of severe hepatotoxicity 💊",
            "Serum creatinine and BUN levels exhibited sustained elevation after 48 hours of perfusion 💉"
        ])
    elif preset_choice == "Financial Earnings Stream (Financial PhraseBank style)":
        default_text = "\n".join([
            "Operating profit increased by 14.5% to EUR 24.8 million compared to EUR 21.6 million in 2021 📈💰",
            "The board proposed an increased dividend payout of 0.85 per share for the fiscal cycle 💵",
            "Consolidated net sales declined 4% due to foreign exchange headwinds and higher raw costs 📊"
        ])
    elif preset_choice == "Social Media Stream with Rich Emojis & Slang":
        default_text = "\n".join([
            "RT @shop_now: flash sale alert! 50% off all hoodies today only 🔥❤️ link: bit.ly/deal 🎉💯",
            "omg never buying from @badbrand again!! worst custmer servce ever :((( 😡💔 #scam",
            "loved the movie so much, totally recommended to all sci-fi fans! :) #mustwatch ✨🍿"
        ])
    else:
        default_text = "Enter sentences, one per line..."

    user_stream = st.text_area("Input Stream (One text sentence per line):", default_text, height=140)

    if st.button("🚀 Process Batch Through Gateway", type="primary"):
        lines = [line.strip() for line in user_stream.split("\n") if line.strip()]

        if not lines:
            st.warning("Please provide at least one text sentence.")
        else:
            with st.spinner("Decoding emojis, partitioning into micro-clusters, and routing batch..."):
                results = gateway.process_batch(lines, k_clusters=k_val)

            st.success(f"Successfully processed {len(results)} texts across {min(k_val, len(lines))} micro-clusters!")

            st.markdown("### 📋 Gateway Routing Decisions & Output")
            for idx, r in enumerate(results):
                with st.container():
                    p_id = r["policy_id"]
                    color_class = f"policy-tag-{p_id}"

                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 1.1rem; font-weight: 600;'>Text #{idx+1} [Cluster {r['assigned_cluster']}]</span>
                            <span class='{color_class}'>{r['routing_decision']} ({r['confidence']})</span>
                        </div>
                        <p style='margin: 8px 0 4px 0; color: #1F2937;'><b>Original:</b> <code>{r['original_text']}</code></p>
                        <p style='margin: 0; color: #047857;'><b>Cleaned:</b> <code>{r['cleaned_text']}</code></p>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: EXTENDED EMOJI READER
# ==============================================================================
elif app_mode == "😀 Extended Emoji & Emoticon Reader":
    st.markdown("<div class='main-title'>😀 Extended Emoji & Emoticon Reader</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Comprehensive support for 80+ emojis and ASCII emoticons across Social, Healthcare, and Financial domains.</div>", unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.markdown("#### 🧪 Test Custom Emojis & Emoticons")
        test_emoji_input = st.text_area(
            "Enter text containing any mix of emojis and ASCII emoticons:",
            "Doctor prescribed new pills 🩺💊 but patient felt terrible 😢💔. Stock was up 📈💰! :) :-D <3"
        )
        if test_emoji_input:
            decoded_text = ExtendedEmojiEngine.normalize_text_emojis(test_emoji_input)
            st.markdown("**Decoded Output (Semantic Tokens):**")
            st.info(decoded_text)

    with col_t2:
        st.markdown("#### 📚 Supported Emoji & Emoticon Lexicon (Sample)")
        lexicon_sample = ExtendedEmojiEngine.list_supported_emojis()
        sample_df = pd.DataFrame([
            {"Symbol / Emoticon": k, "Decoded Domain Token": v}
            for k, v in lexicon_sample[:30]
        ])
        st.dataframe(sample_df, height=320, use_container_width=True)

# ==============================================================================
# TAB 5: SIGNAL RETENTION LAB
# ==============================================================================
elif app_mode == "🔬 Signal Retention Lab":
    st.markdown("<div class='main-title'>🔬 Signal Retention Laboratory</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Empirical proof that one global aggressive policy destroys domain-critical signals (Table 1.1 & Table 6.1).</div>", unsafe_allow_html=True)

    retention_scores = SignalRetentionAnalyzer.evaluate_all_policies()
    df_retention = pd.DataFrame(retention_scores)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### Table 6.1: Signal Retention Rate by Policy")
        st.dataframe(df_retention.style.format("{:.2f}").highlight_max(axis=1, color="#D1FAE5"), use_container_width=True)
        st.markdown("#### Signal Retention Visual Comparison")
        st.bar_chart(df_retention.T)

    with col2:
        st.markdown("#### 🚨 Key Experimental Insights")
        st.error("**P1 Aggressive Social Destroys Signal:**\n- 86% of numeric tokens lost in finance\n- 100% of case lost in biomedical text\n- 100% of hyphenated entities destroyed")
        st.success("**P3 Biomedical Preserves:**\n- 100% case & hyphenated drug/disease entities kept\n- Preserves clinical tokens (e.g. Sprague-Dawley)")
        st.info("**P2 Negation Shield Protects Polarity:**\n- Tags words in scope with `NOT_` prefix, preventing sentiment flips")

# ==============================================================================
# TAB 6: DOWNSTREAM EVALUATION & REGRET
# ==============================================================================
elif app_mode == "📈 Downstream Evaluation & Regret":
    st.markdown("<div class='main-title'>📈 Downstream Representation Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Normalized Regret and Spearman Rank Correlation (Table 6.4 & Slide 10).</div>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Mean Normalized Regret", value="0.0001", delta="Near-Zero Regret")
    with col_m2:
        st.metric(label="Mean Spearman Rank Correlation", value="0.7379", delta="Strong Rank Alignment")

    table_6_4 = pd.DataFrame([
        {"Batch": "Biomedical", "P0 Minimal": 0.03902, "P1 Aggressive": 0.04246, "P2 NegShield": 0.03832, "P3 BioEntity": 0.03902, "Oracle": "P1", "Regret": 0.0000, "Spearman": 0.9487},
        {"Batch": "Financial", "P0 Minimal": 0.04772, "P1 Aggressive": 0.05084, "P2 NegShield": 0.04574, "P3 BioEntity": 0.04772, "Oracle": "P1", "Regret": 0.0000, "Spearman": 0.9487},
        {"Batch": "Social", "P0 Minimal": 0.04128, "P1 Aggressive": 0.05522, "P2 NegShield": 0.04832, "P3 BioEntity": 0.04128, "Oracle": "P1", "Regret": 0.0000, "Spearman": 0.3162}
    ])

    st.markdown("#### Table 6.4: Proxy Scores and Routing Outcome")
    st.dataframe(table_6_4, use_container_width=True)

    st.markdown("#### Table 6.5: Comparison of Router Versions on Unseen Stream")
    table_6_5 = pd.DataFrame([
        {"Router Version": "v1: Proxy-oracle labels (9 clusters)", "Biomedical": "P0 Minimal", "Financial": "P0 Minimal", "Social": "P0 Minimal"},
        {"Router Version": "v2: Calibrated on 400 profile vectors", "Biomedical": "P3 BioEntity", "Financial": "P0 Minimal", "Social": "P1 Aggressive"}
    ])
    st.table(table_6_5)
