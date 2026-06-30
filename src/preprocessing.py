"""
Text preprocessing and urgency label engineering.

Functions:
    clean_text(text): Basic text cleaning for ML
    assign_urgency(text): Rule-based urgency (High/Medium/Low)
    urgency_score(text): Fine-grained 0.0-1.0 urgency score
    load_and_preprocess(path): Full pipeline to load + clean + label
"""
import pandas as pd
import re
import numpy as np
from src.config import (
    HIGH_URGENCY_KEYWORDS, MEDIUM_URGENCY_KEYWORDS,
    URGENCY_LEVELS, URGENCY_MAP,
    URL_PATTERN, EMAIL_PATTERN, PHONE_PATTERN, REDACTED_PATTERN,
    SPECIAL_CHARS, MULTI_SPACE, DATA_PATH,
)


def clean_text(text: str) -> str:
    """Clean complaint text for ML processing.

    Steps: lowercase, remove URLs/emails/phones/redacted patterns,
    remove special chars, collapse whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(' ', text)
    text = EMAIL_PATTERN.sub(' ', text)
    text = PHONE_PATTERN.sub(' ', text)
    text = REDACTED_PATTERN.sub(' ', text)
    text = SPECIAL_CHARS.sub(' ', text)
    text = MULTI_SPACE.sub(' ', text)
    return text.strip()


def assign_urgency(text: str) -> int:
    """Rule-based urgency label.

    Returns: 2 (High), 1 (Medium), or 0 (Low)
    """
    if not isinstance(text, str) or not text.strip():
        return 0

    text_lower = text.lower()

    # Count keyword matches
    high_count = sum(1 for kw in HIGH_URGENCY_KEYWORDS if kw in text_lower)
    medium_count = sum(1 for kw in MEDIUM_URGENCY_KEYWORDS if kw in text_lower)

    # Check for exclamation markers (urgency signals)
    exclamation_count = text_lower.count('!')
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # High urgency triggers
    if high_count >= 1:
        return 2
    if high_count >= 1 and medium_count >= 1:
        return 2
    if exclamation_count >= 3:
        return 2
    if caps_ratio > 0.4 and len(text) > 50:
        # Lots of CAPS indicates frustration
        return 2

    # Medium urgency triggers
    if medium_count >= 1:
        return 1
    if medium_count >= 0 and exclamation_count >= 1 and caps_ratio > 0.2:
        return 1

    return 0  # Low urgency


def urgency_score(text: str) -> float:
    """Fine-grained urgency score from 0.0 (routine) to 1.0 (critical).

    Combines keyword density, text features, and structural indicators.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)
    if word_count == 0:
        return 0.0

    # Keyword density
    high_count = sum(1 for kw in HIGH_URGENCY_KEYWORDS if kw in text_lower)
    medium_count = sum(1 for kw in MEDIUM_URGENCY_KEYWORDS if kw in text_lower)

    high_density = high_count / max(word_count, 1) * 100
    medium_density = medium_count / max(word_count, 1) * 100

    # Structural signals
    exclamation_density = text_lower.count('!') / max(word_count, 1) * 100
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # Compute components
    score = 0.0

    # High keywords: 0.0 to 0.6
    score += min(high_density * 3.0, 0.6)

    # Medium keywords: 0.0 to 0.2
    score += min(medium_density * 1.0, 0.2)

    # Exclamation bonus: 0.0 to 0.15
    score += min(exclamation_density * 5.0, 0.15)

    # Caps ratio bonus: 0.0 to 0.05
    if caps_ratio > 0.3:
        score += min(caps_ratio * 0.1, 0.05)

    return min(score, 1.0)


def load_and_preprocess(path: str = DATA_PATH) -> pd.DataFrame:
    """Load data, clean text, assign urgency labels and scores.

    Returns DataFrame with columns:
        grievance_id, complaint_text, cleaned_text, category,
        urgency (0/1/2), urgency_label (Low/Medium/High),
        urgency_score (0.0-1.0), state, district
    """
    df = pd.read_csv(path)

    # Clean text
    df['cleaned_text'] = df['complaint_text'].apply(clean_text)

    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 10].copy()
    df = df.reset_index(drop=True)

    # Assign urgency
    df['urgency'] = df['complaint_text'].apply(assign_urgency)
    df['urgency_label'] = df['urgency'].map({0: 'Low', 1: 'Medium', 2: 'High'})
    df['urgency_score'] = df['complaint_text'].apply(urgency_score)

    print(f"Loaded {len(df)} records")
    print(f"Category distribution:\n{df['category'].value_counts()}")
    print(f"\nUrgency distribution:\n{df['urgency_label'].value_counts()}")

    return df


if __name__ == '__main__':
    df = load_and_preprocess()
    print(f"\nSample cleaned text:\n{df['cleaned_text'].iloc[0][:200]}")
    print(f"Urgency: {df['urgency_label'].iloc[0]} (score: {df['urgency_score'].iloc[0]:.3f})")
