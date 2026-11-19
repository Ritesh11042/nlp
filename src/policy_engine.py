"""
Specialized Preprocessing Policy Space
Provides four domain-adapted text cleaning policies:
- P0: Minimal / Technical (preserves case, digits, punctuation, syntax)
- P1: Aggressive Social (lowercasing, handle removal, punctuation stripping)
- P2: Negation Shielded (lowercasing, NOT_ prefixing negation scopes)
- P3: Biomedical Entity Preserving (preserves case, hyphens, scientific units, entity delimiters)
"""

import re
import string
from typing import Dict, Set

POLICY_NAMES: Dict[int, str] = {
    0: "Minimal_Technical",
    1: "Aggressive_Social",
    2: "Negation_Shielded",
    3: "Biomedical_Entity_Preserving"
}

POLICY_DESCRIPTIONS: Dict[int, str] = {
    0: "P0 Minimal / Technical: Preserves case, numeric symbols, currency codes, and exact code/data syntax.",
    1: "P1 Aggressive Social: Aggressively normalizes noise, lowercases, removes user handles and punctuation.",
    2: "P2 Negation Shielded: Shields semantic polarity by prefixing tokens in negation scopes with NOT_.",
    3: "P3 Biomedical Entity Preserving: Protects hyphenated terms, gene/drug acronyms, units, and chemical delimiters."
}


class DomainPolicyEngine:
    """Applies domain-specific text preprocessing policies."""

    NEGATION_WORDS: Set[str] = {
        "not", "never", "no", "n't", "nor", "neither", "barely", "hardly", "scarcely"
    }

    PUNCT_SCOPE_TERMINATORS: Set[str] = {".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "-"}

    EMOJI_MAP: Dict[str, str] = {
        ":)": " [EMOJI_HAPPY] ",
        ":-)": " [EMOJI_HAPPY] ",
        ":(": " [EMOJI_SAD] ",
        ":-(": " [EMOJI_SAD] ",
        ":D": " [EMOJI_LAUGH] ",
        ";)": " [EMOJI_WINK] ",
        "❤️": " [EMOJI_LOVE] ",
        "❤": " [EMOJI_LOVE] ",
        "🔥": " [EMOJI_FIRE] ",
        "👍": " [EMOJI_THUMBSUP] ",
        "👎": " [EMOJI_THUMBSDOWN] ",
        "😡": " [EMOJI_ANGRY] ",
        "😢": " [EMOJI_CRY] ",
        "😭": " [EMOJI_SOB] ",
    }

    @classmethod
    def normalize_shared(cls, text: str) -> str:
        """
        Shared preprocessing step applied before policy-specific logic:
        - URL removal
        - Standard emoticon and emoji normalization to token representations
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Normalize common emojis & emoticons
        for emo, token in cls.EMOJI_MAP.items():
            text = text.replace(emo, token)

        # Strip URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        return text

    @classmethod
    def apply_policy(cls, text: str, policy_id: int) -> str:
        """
        Applies policy `policy_id` (0, 1, 2, or 3) to the input string.
        """
        text = cls.normalize_shared(text)
        if not text:
            return ""

        if policy_id == 0:
            # P0: Minimal / Technical
            # Preserves case, digits, symbols, percentage, punctuation
            return re.sub(r"\s+", " ", text).strip()

        elif policy_id == 1:
            # P1: Aggressive Social
            # Lowercase, remove @mentions, remove all punctuation, collapse whitespace
            text = text.lower()
            text = re.sub(r"@\w+", "", text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            return re.sub(r"\s+", " ", text).strip()

        elif policy_id == 2:
            # P2: Negation Shield
            # Lowercase, track negation scope, prefix tokens with NOT_ until scope terminator
            tokens = text.lower().split()
            shielded = []
            neg_active = False

            for t in tokens:
                clean_t = t.strip(string.punctuation)
                if clean_t in cls.NEGATION_WORDS or t.endswith("n't"):
                    neg_active = True
                    shielded.append(t)
                elif neg_active:
                    # Check if token contains punctuation terminating negation scope
                    has_terminator = any(term in t for term in cls.PUNCT_SCOPE_TERMINATORS)
                    shielded.append(f"NOT_{t}")
                    if has_terminator:
                        neg_active = False
                else:
                    shielded.append(t)

            return re.sub(r"\s+", " ", " ".join(shielded)).strip()

        elif policy_id == 3:
            # P3: Biomedical Entity Preserving
            # Preserves:
            # - Exact case (VEGFR2, Ciprofloxacin, Sprague-Dawley, BUN, RNA, mRNA)
            # - Hyphenated compound terms (ischemia-reperfusion, anti-VEGFR2)
            # - Scientific units and delimiters (mg/kg, mL, pH 7.4, 18.4%)
            # Normalizes erratic whitespace (tabs, carriage returns, newlines)
            text = re.sub(r"[\r\n\t]+", " ", text)
            return re.sub(r"\s+", " ", text).strip()

        return text
