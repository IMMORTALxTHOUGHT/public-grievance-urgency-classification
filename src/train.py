#!/usr/bin/env python3
"""
Training pipeline for category and urgency classifiers.

Trains both models, evaluates, performs cross-validation for model selection,
and saves best models to disk.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_DIR, FIGURE_DIR, CATEGORIES, RANDOM_STATE,
)
from src.preprocessing import load_and_preprocess, clean_text
from src.features import build_tfidf, extract_meta_features, stratified_split
from src.models import CategoryClassifier, UrgencyClassifier, save_model, load_model
from src.escalation import recommend_escalation, generate_response


def select_best_model(X_train, y_train, model_types, classifier_class, cv=5):
    """Select the best model type using cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        model_types: List of model type strings (e.g., ['logistic', 'random_forest'])
        classifier_class: Class to instantiate (CategoryClassifier or UrgencyClassifier)
        cv: Number of cross-validation folds

    Returns:
        best_model: Trained model of the selected type (retrained on full training set)
        best_type: String name of the best model type
        cv_results: Dict of model_type -> mean CV score
    """
    print(f"\nSelecting best model via {cv}-fold cross-validation...")
    cv_results = {}

    for model_type in model_types:
        clf = classifier_class(model_type=model_type)
        scores = cross_val_score(
            clf.clf, X_train, y_train,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
            scoring='f1_weighted',
            n_jobs=-1,
        )
        mean_score = scores.mean()
        cv_results[model_type] = mean_score
        print(f"  {model_type}: CV F1 = {mean_score:.4f} (+/- {scores.std():.4f})")

    best_type = max(cv_results, key=cv_results.get)
    print(f"\n  Selected: {best_type} (CV F1 = {cv_results[best_type]:.4f})")

    best_model = classifier_class(model_type=best_type)
    best_model.fit(X_train, y_train)
    return best_model, best_type, cv_results


def train_category_model(X_train, y_train, X_test, y_test):
    """Train and evaluate category classifier with CV-based model selection."""
    print("\n" + "=" * 60)
    print("TRAINING CATEGORY CLASSIFIER")
    print("=" * 60)

    best_model, best_type, cv_results = select_best_model(
        X_train, y_train,
        ['logistic', 'random_forest'],
        CategoryClassifier,
    )

    results = best_model.evaluate(X_test, y_test)
    print(f"\n  Test weighted F1: {results['f1_weighted']:.4f}")

    results['cv_scores'] = cv_results
    results['best_type'] = best_type

    return best_model, results


def train_urgency_model(X_train, y_train, X_test, y_test):
    """Train and evaluate urgency classifier with CV-based model selection."""
    print("\n" + "=" * 60)
    print("TRAINING URGENCY CLASSIFIER")
    print("=" * 60)

    best_model, best_type, cv_results = select_best_model(
        X_train, y_train,
        ['logistic', 'random_forest'],
        UrgencyClassifier,
    )

    results = best_model.evaluate(X_test, y_test)
    print(f"\n  Test weighted F1: {results['f1_weighted']:.4f}")

    results['cv_scores'] = cv_results
    results['best_type'] = best_type

    return best_model, results


def generate_eda_figures(df):
    """Generate EDA visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(FIGURE_DIR, exist_ok=True)
    sns.set_style("darkgrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)

    # 1. Category distribution
    plt.figure()
    cat_order = df['category'].value_counts().index
    ax = sns.countplot(data=df, y='category', order=cat_order, hue='category', legend=False)
    ax.set_title('Distribution of Complaint Categories', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Complaints')
    ax.set_ylabel('Category')
    for i, v in enumerate(df['category'].value_counts().values):
        ax.text(v + 10, i, str(v), va='center')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/category_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURE_DIR}/category_distribution.png")

    # 2. Urgency distribution
    plt.figure()
    urg_order = ['Low', 'Medium', 'High']
    ax = sns.countplot(data=df, x='urgency_label', order=urg_order,
                        hue='urgency_label', palette={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
                        legend=False)
    ax.set_title('Distribution of Urgency Levels', fontsize=16, fontweight='bold')
    ax.set_xlabel('Urgency Level')
    ax.set_ylabel('Number of Complaints')
    for i, v in enumerate(df['urgency_label'].value_counts().values):
        ax.text(i, v + 10, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/urgency_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURE_DIR}/urgency_distribution.png")

    # 3. Category vs Urgency heatmap
    plt.figure()
    ct = pd.crosstab(df['category'], df['urgency_label'], normalize='index')
    for col in ['Low', 'Medium', 'High']:
        if col not in ct.columns:
            ct[col] = 0.0
    ct = ct[['Low', 'Medium', 'High']]
    sns.heatmap(ct, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5)
    plt.title('Urgency Proportion by Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/category_urgency_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURE_DIR}/category_urgency_heatmap.png")

    # 4. Text length distribution by urgency
    plt.figure()
    for i, (urg, color) in enumerate(zip(['Low', 'Medium', 'High'], ['green', 'orange', 'red'])):
        subset = df[df['urgency_label'] == urg]['cleaned_text'].str.len()
        plt.hist(subset, bins=50, alpha=0.5, label=urg, color=color, density=True)
    plt.title('Text Length Distribution by Urgency Level', fontsize=14, fontweight='bold')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(0, 2000)
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/text_length_by_urgency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURE_DIR}/text_length_by_urgency.png")

    # 5. Urgency score distribution
    plt.figure()
    plt.hist(df['urgency_score'], bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Urgency Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Urgency Score (0-1)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/urgency_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURE_DIR}/urgency_score_distribution.png")

    # 6. Top 10 states with most complaints
    if 'state' in df.columns and df['state'].notna().any():
        plt.figure()
        top_states = df['state'].value_counts().head(10)
        ax = sns.barplot(x=top_states.values, y=top_states.index, hue=top_states.index, palette='viridis', legend=False)
        ax.set_title('Top 10 States by Complaint Volume', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Complaints')
        for i, v in enumerate(top_states.values):
            ax.text(v + 5, i, str(v), va='center')
        plt.tight_layout()
        plt.savefig(f'{FIGURE_DIR}/top_states.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {FIGURE_DIR}/top_states.png")

    print("\nAll EDA figures generated ✓")


def plot_confusion_matrix(cm, class_names, title, filename):
    """Save confusion matrix plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{FIGURE_DIR}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGURE_DIR}/{filename}")


def main():
    print("=" * 60)
    print("PUBLIC GRIEVANCE URGENCY CLASSIFICATION SYSTEM")
    print("Training Pipeline")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    df = load_and_preprocess()
    print(f"  Total samples: {len(df)}")

    # Step 2: Generate EDA figures (full dataset - EDA is exploratory, not predictive)
    print("\n[Step 2] Generating EDA visualizations...")
    generate_eda_figures(df)

    # Step 3: Train/test split FIRST (prevents data leakage)
    print("\n[Step 3] Creating train/test splits...")
    train_idx, test_idx = stratified_split(df, 'category')
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # Step 4: Build features on TRAIN set only
    print("\n[Step 4] Building features on training set...")
    vectorizer, X_tfidf_train = build_tfidf(train_df['cleaned_text'])
    X_tfidf_test = vectorizer.transform(test_df['cleaned_text'])

    # Meta features for urgency model
    meta_train_df = extract_meta_features(train_df)
    meta_test_df = extract_meta_features(test_df)

    # Scale meta features (fit on train, transform both)
    scaler = StandardScaler(with_mean=False)
    X_meta_train = scaler.fit_transform(meta_train_df.values)
    X_meta_test = scaler.transform(meta_test_df.values)

    # Combine TF-IDF + scaled meta features
    X_meta_train_sparse = csr_matrix(X_meta_train)
    X_meta_test_sparse = csr_matrix(X_meta_test)
    X_combined_train = hstack([X_tfidf_train, X_meta_train_sparse])
    X_combined_test = hstack([X_tfidf_test, X_meta_test_sparse])

    # Labels
    y_cat_train = train_df['category']
    y_cat_test = test_df['category']
    y_urg_train = train_df['urgency']
    y_urg_test = test_df['urgency']

    # Step 5: Train category classifier
    print("\n[Step 5] Training category classifier...")
    cat_model, cat_results = train_category_model(
        X_tfidf_train, y_cat_train, X_tfidf_test, y_cat_test
    )

    # Step 6: Train urgency classifier
    print("\n[Step 6] Training urgency classifier...")
    urg_model, urg_results = train_urgency_model(
        X_combined_train, y_urg_train, X_combined_test, y_urg_test
    )

    # Step 7: Save models + vectorizer + scaler
    print("\n[Step 7] Saving models...")
    save_model(cat_model, f"{MODEL_DIR}/category_model.pkl")
    save_model(urg_model, f"{MODEL_DIR}/urgency_model.pkl")
    save_model(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")
    save_model(scaler, f"{MODEL_DIR}/meta_scaler.pkl")

    metadata = {
        'categories': sorted(train_df['category'].unique().tolist()),
        'category_accuracy': float(cat_results['accuracy']),
        'category_f1': float(cat_results['f1_weighted']),
        'category_best_model': cat_results['best_type'],
        'category_cv_scores': cat_results['cv_scores'],
        'urgency_accuracy': float(urg_results['accuracy']),
        'urgency_f1': float(urg_results['f1_weighted']),
        'urgency_best_model': urg_results['best_type'],
        'urgency_cv_scores': urg_results['cv_scores'],
    }
    joblib.dump(metadata, f"{MODEL_DIR}/metadata.pkl")
    print(f"  Metadata saved")

    # Step 8: Plot confusion matrices
    print("\n[Step 8] Saving confusion matrix plots...")
    cat_names = sorted(train_df['category'].unique())
    urg_names = ['Low', 'Medium', 'High']
    plot_confusion_matrix(
        cat_results['confusion_matrix'], cat_names,
        'Category Classification - Confusion Matrix',
        'category_confusion_matrix.png'
    )
    plot_confusion_matrix(
        urg_results['confusion_matrix'], urg_names,
        'Urgency Classification - Confusion Matrix',
        'urgency_confusion_matrix.png'
    )

    # Step 9: Demo predictions
    print("\n[Step 9] Sample predictions...")
    sample_texts = [
        "There is no water supply in our colony for the past one week. Residents are suffering from severe shortage. Please arrange water tanker urgently.",
        "The road in front of our house has a large pothole that has been there for months. It causes accidents regularly.",
        "I would like to inquire about the status of my application for a new electricity connection.",
    ]

    for text in sample_texts:
        cleaned = clean_text(text)
        tfidf_feats = vectorizer.transform([cleaned])
        meta_temp = extract_meta_features(pd.DataFrame({'complaint_text': [text], 'cleaned_text': [cleaned]}))
        meta_scaled = scaler.transform(meta_temp.values)
        urg_feats = hstack([tfidf_feats, csr_matrix(meta_scaled)])

        cat_pred = cat_model.predict(tfidf_feats)[0]
        urg_pred = urg_model.predict(urg_feats)[0]
        urg_label = ['Low', 'Medium', 'High'][urg_pred]

        escalation = recommend_escalation(cat_pred, urg_label)
        response = generate_response(text, cat_pred, urg_label, escalation)
        print(response)
        print()

    print("\n✓ Training pipeline complete!")
    print(f"  Category model: {MODEL_DIR}/category_model.pkl")
    print(f"  Urgency model: {MODEL_DIR}/urgency_model.pkl")
    print(f"  Vectorizer: {MODEL_DIR}/tfidf_vectorizer.pkl")
    print(f"  Meta scaler: {MODEL_DIR}/meta_scaler.pkl")
    print(f"  EDA figures: {FIGURE_DIR}/")


if __name__ == '__main__':
    main()
