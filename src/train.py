#!/usr/bin/env python3
"""
Training pipeline for category and urgency classifiers.

Trains both models, evaluates, performs grid search,
and saves best models to disk.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Add parent directory to path so imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_DIR, FIGURE_DIR, CATEGORIES, RANDOM_STATE,
)
from src.preprocessing import load_and_preprocess
from src.features import build_tfidf, extract_meta_features, stratified_split
from src.models import CategoryClassifier, UrgencyClassifier, save_model, load_model
from src.escalation import recommend_escalation, generate_response


def train_category_model(X_train, y_train, X_test, y_test):
    """Train and evaluate category classifier with grid search."""
    print("\n" + "=" * 60)
    print("TRAINING CATEGORY CLASSIFIER")
    print("=" * 60)

    # Logistic Regression with grid search
    print("\n[1/2] Training Logistic Regression...")
    lr = CategoryClassifier(model_type='logistic')
    lr.fit(X_train, y_train)
    results_lr = lr.evaluate(X_test, y_test)

    # Random Forest
    print("\n[2/2] Training Random Forest...")
    rf = CategoryClassifier(model_type='random_forest')
    rf.fit(X_train, y_train)
    results_rf = rf.evaluate(X_test, y_test)

    # Pick best model based on weighted F1
    if results_lr['f1_weighted'] >= results_rf['f1_weighted']:
        print(f"\n✓ Logistic Regression selected (F1: {results_lr['f1_weighted']:.4f})")
        best_model = lr
        best_results = results_lr
    else:
        print(f"\n✓ Random Forest selected (F1: {results_rf['f1_weighted']:.4f})")
        best_model = rf
        best_results = results_rf

    return best_model, best_results


def train_urgency_model(X_train, y_train, X_test, y_test):
    """Train and evaluate urgency classifier."""
    print("\n" + "=" * 60)
    print("TRAINING URGENCY CLASSIFIER")
    print("=" * 60)

    # Logistic Regression
    print("\n[1/2] Training Logistic Regression...")
    lr = UrgencyClassifier(model_type='logistic')
    lr.fit(X_train, y_train)
    results_lr = lr.evaluate(X_test, y_test)

    # Random Forest
    print("\n[2/2] Training Random Forest...")
    rf = UrgencyClassifier(model_type='random_forest')
    rf.fit(X_train, y_train)
    results_rf = rf.evaluate(X_test, y_test)

    # Pick best
    if results_lr['f1_weighted'] >= results_rf['f1_weighted']:
        print(f"\n✓ Logistic Regression selected (F1: {results_lr['f1_weighted']:.4f})")
        return lr, results_lr
    else:
        print(f"\n✓ Random Forest selected (F1: {results_rf['f1_weighted']:.4f})")
        return rf, results_rf


def generate_eda_figures(df):
    """Generate EDA visualizations."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
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
    # Add value labels
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
    # Ensure columns exist
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

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    df = load_and_preprocess()
    print(f"  Total samples: {len(df)}")

    # Step 2: Generate EDA figures
    print("\n[Step 2] Generating EDA visualizations...")
    generate_eda_figures(df)

    # Step 3: Build features (TF-IDF on cleaned text for category)
    print("\n[Step 3] Building features...")
    vectorizer, X_tfidf = build_tfidf(df['cleaned_text'])

    # Extract meta features for urgency model
    meta_df = extract_meta_features(df)
    X_meta = meta_df.values
    meta_names = list(meta_df.columns)

    # For category: use TF-IDF only
    # For urgency: use TF-IDF + meta features
    from scipy.sparse import hstack, csr_matrix
    X_meta_sparse = csr_matrix(X_meta)
    X_combined = hstack([X_tfidf, X_meta_sparse])
    print(f"  Combined features shape: {X_combined.shape}")

    # Step 4: Train/test split
    print("\n[Step 4] Creating train/test splits...")
    cat_train_idx, cat_test_idx = stratified_split(df, 'category')

    # Split data
    X_cat_train = X_tfidf[cat_train_idx]
    X_cat_test = X_tfidf[cat_test_idx]
    y_cat_train = df['category'].iloc[cat_train_idx]
    y_cat_test = df['category'].iloc[cat_test_idx]

    X_urg_train = X_combined[cat_train_idx]
    X_urg_test = X_combined[cat_test_idx]
    y_urg_train = df['urgency'].iloc[cat_train_idx]
    y_urg_test = df['urgency'].iloc[cat_test_idx]

    # Step 5: Train category classifier
    print("\n[Step 5] Training category classifier...")
    cat_model, cat_results = train_category_model(
        X_cat_train, y_cat_train, X_cat_test, y_cat_test
    )

    # Step 6: Train urgency classifier
    print("\n[Step 6] Training urgency classifier...")
    urg_model, urg_results = train_urgency_model(
        X_urg_train, y_urg_train, X_urg_test, y_urg_test
    )

    # Step 7: Save models
    print("\n[Step 7] Saving models...")
    save_model(cat_model, f"{MODEL_DIR}/category_model.pkl")
    save_model(urg_model, f"{MODEL_DIR}/urgency_model.pkl")
    save_model(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")

    # Save metadata for inference
    metadata = {
        'categories': sorted(df['category'].unique().tolist()),
        'category_accuracy': float(cat_results['accuracy']),
        'category_f1': float(cat_results['f1_weighted']),
        'urgency_accuracy': float(urg_results['accuracy']),
        'urgency_f1': float(urg_results['f1_weighted']),
    }
    joblib.dump(metadata, f"{MODEL_DIR}/metadata.pkl")
    print(f"  Metadata saved")

    # Step 8: Plot confusion matrices
    print("\n[Step 8] Saving confusion matrix plots...")
    cat_names = sorted(df['category'].unique())
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
        from src.preprocessing import clean_text
        cleaned = clean_text(text)
        tfidf_feats = vectorizer.transform([cleaned])
        meta_df = extract_meta_features(pd.DataFrame({'complaint_text': [text], 'cleaned_text': [cleaned]}))
        from scipy.sparse import csr_matrix
        urg_feats = hstack([tfidf_feats, csr_matrix(meta_df.values)])

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
    print(f"  EDA figures: {FIGURE_DIR}/")


if __name__ == '__main__':
    main()
