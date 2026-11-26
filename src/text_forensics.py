"""
Multi-Signal Text Forensics Engine: text_forensics.py
=====================================================
Extracts deterministic, interpretable linguistic forensic signals:
  1. Lexical Burstiness (Sentence length & rhythm variability)
  2. Text Compressibility & Redundancy Entropy (zlib ratio)
  3. Vocabulary Diversity (Type-Token Ratio)
  4. Discourse Transition Density (Overused AI signposts)
  5. Contraction & Conversational Naturalness (Authentic voice)
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any, Dict, List, Tuple

import numpy as np

from src.detection_engine import ForensicSignal


def compute_burstiness(sentences: List[str]) -> Tuple[float, str, str]:
    """
    Compute sentence length burstiness (coefficient of variation of sentence lengths).
    Human writing has high burstiness (varied rhythm).
    AI writing has low burstiness (uniform, monotonic rhythm).
    """
    if len(sentences) <= 1:
        return 0.0, "NEUTRAL", "WEAK"

    lengths = [len(s.split()) for s in sentences if s.strip()]
    if not lengths or np.mean(lengths) == 0:
        return 0.0, "NEUTRAL", "WEAK"

    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    cv = std_len / mean_len  # Coefficient of Variation

    # Typical human writing: CV >= 0.45; AI writing: CV < 0.35
    if cv < 0.28:
        indication = "SUPPORTS_AI"
        strength = "STRONG"
    elif cv < 0.38:
        indication = "SUPPORTS_AI"
        strength = "MODERATE"
    elif cv > 0.55:
        indication = "SUPPORTS_HUMAN"
        strength = "STRONG"
    else:
        indication = "NEUTRAL"
        strength = "WEAK"

    return round(cv, 3), indication, strength


def compute_compressibility(text: str) -> Tuple[float, str, str]:
    """
    Compute text compressibility ratio via zlib.
    AI text exhibits repetitive syntactic patterns and lower Kolmogorov complexity,
    resulting in a lower compression ratio (higher compressibility).
    """
    raw_bytes = text.strip().encode("utf-8")
    if len(raw_bytes) < 80:
        return 1.0, "NEUTRAL", "WEAK"

    compressed_bytes = zlib.compress(raw_bytes, level=9)
    ratio = len(compressed_bytes) / len(raw_bytes)

    # For texts >= 100 chars:
    # Highly compressible (ratio < 0.62) supports AI; low compressibility (> 0.72) supports Human
    if ratio < 0.58:
        indication = "SUPPORTS_AI"
        strength = "MODERATE"
    elif ratio > 0.74:
        indication = "SUPPORTS_HUMAN"
        strength = "MODERATE"
    else:
        indication = "NEUTRAL"
        strength = "WEAK"

    return round(ratio, 3), indication, strength


def extract_text_forensic_signals(text: str) -> List[ForensicSignal]:
    """Extract all forensic signals for an input text."""
    clean_text = str(text).strip()
    words = clean_text.split()
    word_count = len(words)

    if word_count == 0:
        return []

    signals: List[ForensicSignal] = []

    # 1. Burstiness (Sentence Rhythm)
    raw_sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]
    cv_score, cv_ind, cv_str = compute_burstiness(raw_sentences)
    signals.append(ForensicSignal(
        name="Sentence Rhythm (Burstiness)",
        category="linguistic",
        description="Measures variability in sentence lengths. AI text is typically uniform; human text has high rhythm diversity.",
        value=cv_score,
        indication=cv_ind,
        strength=cv_str,
    ))

    # 2. Compressibility
    comp_ratio, comp_ind, comp_str = compute_compressibility(clean_text)
    signals.append(ForensicSignal(
        name="Syntactic Complexity Ratio",
        category="information_theory",
        description="Measures structural redundancy via lossless byte entropy. Lower ratios indicate formulaic AI syntactic templates.",
        value=comp_ratio,
        indication=comp_ind,
        strength=comp_str,
    ))

    # 3. Vocabulary Diversity (Type-Token Ratio)
    unique_words = len(set(w.lower() for w in words))
    ttr = round(unique_words / max(word_count, 1), 3)
    ttr_ind = "SUPPORTS_HUMAN" if (ttr > 0.85 and word_count >= 20) else ("SUPPORTS_AI" if ttr < 0.55 else "NEUTRAL")
    signals.append(ForensicSignal(
        name="Vocabulary Diversity (TTR)",
        category="linguistic",
        description="Proportion of unique words in text. Natural human writing balances repetition and varied expressive phrasing.",
        value=ttr,
        indication=ttr_ind,
        strength="MODERATE" if ttr_ind != "NEUTRAL" else "WEAK",
    ))

    # 4. Discourse Transitions
    transition_words = [
        r"\bfurthermore\b", r"\bmoreover\b", r"\bin conclusion\b", r"\bin addition\b",
        r"\bhowever\b", r"\btherefore\b", r"\bthus\b", r"\bconsequently\b",
        r"\bfirstly\b", r"\bsecondly\b", r"\bfinally\b", r"\bto summarize\b",
        r"\bon the other hand\b", r"\bas a result\b", r"\bin contrast\b"
    ]
    trans_count = len(re.findall("|".join(transition_words), clean_text, re.IGNORECASE))
    trans_density = round((trans_count / max(word_count, 1)) * 100, 2)
    # If >= 2.5% of words are formal transition signposts, strongly indicates AI essay generation
    if trans_density >= 2.5 and trans_count >= 2:
        trans_ind = "SUPPORTS_AI"
        trans_str = "STRONG"
    elif trans_density >= 1.5:
        trans_ind = "SUPPORTS_AI"
        trans_str = "MODERATE"
    else:
        trans_ind = "NEUTRAL"
        trans_str = "WEAK"

    signals.append(ForensicSignal(
        name="Formal Discourse Signposts",
        category="stylistic",
        description="Density of formulaic transition words ('furthermore', 'moreover', 'in conclusion'). High density is an AI hallmark.",
        value=f"{trans_density}% ({trans_count} transitions)",
        indication=trans_ind,
        strength=trans_str,
    ))

    # 5. Contraction & Natural Voice
    contractions = [
        r"can't", r"cannot", r"won't", r"don't", r"doesn't", r"didn't", r"isn't", r"aren't",
        r"wasn't", r"weren't", r"haven't", r"hasn't", r"hadn't", r"i'm", r"i've",
        r"i'll", r"i'd", r"you're", r"you've", r"you'll", r"you'd", r"he's",
        r"she's", r"it's", r"that's", r"we're", r"we've", r"we'll", r"they'd"
    ]
    cont_count = len(re.findall(r"\b(" + "|".join(c.replace("'", r"['\u2019]") for c in contractions) + r")\b", clean_text, re.IGNORECASE))
    cont_density = round((cont_count / max(word_count, 1)) * 100, 2)
    cont_ind = "SUPPORTS_HUMAN" if cont_count >= 2 else "NEUTRAL"
    signals.append(ForensicSignal(
        name="Conversational Contractions",
        category="stylistic",
        description="Use of natural contractions ('don't', 'it's', 'we're'). AI assistant outputs often default to overly formal uncontracted forms.",
        value=f"{cont_density}% ({cont_count} contractions)",
        indication=cont_ind,
        strength="MODERATE" if cont_ind != "NEUTRAL" else "WEAK",
    ))

    return signals
