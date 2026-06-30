"""
Feature engineering: TF-IDF vectorization + meta-features.

Functions:
    build_tfidf(corpus): TF-IDF vectorizer
    extract_meta_features(df): Numerical features from text
    prepare_features(df): Combine TF-IDF + meta features
    stratified_split(df): Train/test split preserving distributions
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
from src.config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TEST_SIZE,
    RANDOM_STATE, MODEL_DIR,
)


def build_tfidf(corpus, max_features=None, ngram_range=None):
    """Fit and return a TF-IDF vectorizer + transformed matrix."""
    if max_features is None:
        max_features = TFIDF_MAX_FEATURES
    if ngram_range is None:
        ngram_range = TFIDF_NGRAM_RANGE

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(corpus)
    print(f"TF-IDF: {X_tfidf.shape[0]} docs, {X_tfidf.shape[1]} features")
    return vectorizer, X_tfidf


def extract_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numerical features from complaint text.

    Features:
        text_length, word_count, avg_word_length,
        caps_ratio, exclamation_count, urgent_keyword_count,
        has_urgent_keyword, has_medium_keyword
    """
    texts = df['complaint_text'].fillna('')
    cleaned = df.get('cleaned_text', texts)

    features = pd.DataFrame(index=df.index)
    features['text_length'] = texts.str.len()
    features['word_count'] = cleaned.str.split().str.len()
    features['avg_word_length'] = cleaned.str.len() / features['word_count'].replace(0, 1)

    features['caps_ratio'] = texts.apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    features['exclamation_count'] = texts.str.count('!')
    features['question_count'] = texts.str.count(r'\?')

    # Fill NaN/Inf
    features = features.replace([np.inf, -np.inf], 0).fillna(0)

    return features


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare combined feature set from DataFrame.

    Returns: (vectorizer, X_tfidf, X_meta, feature_names)
    """
    # TF-IDF on cleaned text
    vectorizer, X_tfidf = build_tfidf(df['cleaned_text'])

    # Meta features
    meta_df = extract_meta_features(df)
    X_meta = meta_df.values
    meta_names = list(meta_df.columns)

    print(f"Meta features: {len(meta_names)} -> {list(meta_names)}")
    return vectorizer, X_tfidf, X_meta, meta_names


def stratified_split(df, target_col='category', test_size=None, random_state=None):
    """Perform stratified train/test split.

    Returns: (train_idx, test_idx)
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    train_idx, test_idx = next(splitter.split(df, df[target_col]))
    print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

    # Verify distribution is preserved
    train_dist = df.iloc[train_idx][target_col].value_counts(normalize=True)
    test_dist = df.iloc[test_idx][target_col].value_counts(normalize=True)
    print(f"Train distribution:\n{train_dist}")
    print(f"Test distribution:\n{test_dist}")

    return train_idx, test_idx


if __name__ == '__main__':
    from src.preprocessing import load_and_preprocess
    df = load_and_preprocess()
    train_idx, test_idx = stratified_split(df, 'category')
    vec, X_tfidf, X_meta, meta_names = prepare_features(df)
    print("\nFeature preparation complete.")
    print(f"TF-IDF shape: {X_tfidf.shape}")
    print(f"Meta features shape: {X_meta.shape}")
