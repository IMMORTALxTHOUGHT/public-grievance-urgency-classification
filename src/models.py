"""
Classification models for category and urgency prediction.

Classes:
    CategoryClassifier: Multi-class logistic regression for complaint category
    UrgencyClassifier: Multi-class classifier for urgency level (High/Medium/Low)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
import joblib


class CategoryClassifier:
    """Predicts complaint category from text features."""

    def __init__(self, model_type='logistic', **kwargs):
        self.model_type = model_type
        if model_type == 'logistic':
            self.clf = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                **kwargs
            )
        elif model_type == 'random_forest':
            self.clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def evaluate(self, X, y, label_names=None):
        preds = self.predict(X)
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)

        print(f"\n=== {self.model_type.upper()} Category Classifier ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1: {f1_score(y, preds, average='weighted'):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y, preds, zero_division=0))
        print(f"\nConfusion Matrix:\n{cm}")

        return {
            'accuracy': acc,
            'f1_weighted': f1_score(y, preds, average='weighted'),
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': preds,
        }


class UrgencyClassifier:
    """Predicts urgency level (High/Medium/Low) from combined features."""

    def __init__(self, model_type='logistic', **kwargs):
        self.model_type = model_type
        if model_type == 'logistic':
            self.clf = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                **kwargs
            )
        elif model_type == 'random_forest':
            self.clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)

        print(f"\n=== {self.model_type.upper()} Urgency Classifier ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Weighted F1: {f1_score(y, preds, average='weighted'):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(
            y, preds,
            target_names=['Low (0)', 'Medium (1)', 'High (2)'],
            zero_division=0
        ))
        print(f"\nConfusion Matrix:\n{cm}")

        return {
            'accuracy': acc,
            'f1_weighted': f1_score(y, preds, average='weighted'),
            'classification_report': classification_report(
                y, preds, output_dict=True, zero_division=0
            ),
            'confusion_matrix': cm,
            'predictions': preds,
        }


def save_model(model, path):
    """Save a trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """Load a trained model from disk."""
    return joblib.load(path)
