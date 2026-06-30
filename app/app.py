"""
Streamlit Demo App — Public Grievance Urgency Classification System

Run with: streamlit run app/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must be first Streamlit command
st.set_page_config(
    page_title="Grievance Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import CATEGORIES, CATEGORY_DEPT, FIGURE_DIR, MODEL_DIR
from src.preprocessing import clean_text, assign_urgency, urgency_score
from src.features import extract_meta_features
from src.models import load_model
from src.escalation import recommend_escalation, generate_response, get_category_department

# ── Constants ──────────────────────────────────────────────────────
MODEL_PATH = MODEL_DIR.rstrip('/') + '/category_model.pkl'
URGENCY_MODEL_PATH = MODEL_DIR.rstrip('/') + '/urgency_model.pkl'
VECTORIZER_PATH = MODEL_DIR.rstrip('/') + '/tfidf_vectorizer.pkl'
METADATA_PATH = MODEL_DIR.rstrip('/') + '/metadata.pkl'

URGENCY_COLORS = {
    "High": "#dc3545",
    "Medium": "#ffc107",
    "Low": "#28a745",
}
URGENCY_EMOJIS = {"High": " CRITICAL", "Medium": " ATTENTION", "Low": " ROUTINE"}

SAMPLE_COMPLAINTS = [
    "There is no water supply in our colony for the past one week. Residents are suffering from severe shortage. Please arrange water tanker urgently.",
    "The road in front of our house has a large pothole that has been there for months. It causes accidents regularly.",
    "Power cut for the last 12 hours in our area. There is no information from the electricity department. We have elderly and sick people at home.",
    "The government hospital in our district has no doctor available for the past 3 days. Patients are being turned away. This is a matter of life and death.",
    "I would like to inquire about the status of my application for a new electricity connection.",
    "The garbage in our street has not been collected for two weeks. Stray dogs are tearing the bags and there is a bad smell.",
    "The pension of my father has not been credited for the last 3 months. We have submitted all documents multiple times but no action taken.",
    "A group of antisocial elements is harassing women in our locality after dark. Police patrol is needed urgently.",
    "The school building in our village is in dilapidated condition. Classes are being held under trees. Need immediate repair.",
    "The water from the tap is muddy and has a strange smell. Some people in our area have fallen sick after drinking it.",
]


# ── Cached model loading ───────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load trained models with caching."""
    try:
        cat_model = load_model(MODEL_PATH)
        urg_model = load_model(URGENCY_MODEL_PATH)
        vectorizer = load_model(VECTORIZER_PATH)
        try:
            metadata = load_model(METADATA_PATH)
        except Exception:
            metadata = {}
        return cat_model, urg_model, vectorizer, metadata
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Run `python3 src/train.py` first to train the models.")
        return None, None, None, {}
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, {}


def predict_complaint(text, cat_model, urg_model, vectorizer):
    """Run full prediction pipeline on a complaint text."""
    # Clean
    cleaned = clean_text(text)

    # TF-IDF for category
    tfidf_feats = vectorizer.transform([cleaned])

    # Category prediction
    cat_pred = cat_model.predict(tfidf_feats)[0]

    # Get category probabilities
    cat_probs = cat_model.predict_proba(tfidf_feats)[0]
    cat_prob_dict = dict(zip(cat_model.classes_, cat_probs))

    # Meta features for urgency
    meta_df = extract_meta_features(pd.DataFrame({
        'complaint_text': [text],
        'cleaned_text': [cleaned],
    }))

    from scipy.sparse import csr_matrix, hstack
    urg_feats = hstack([tfidf_feats, csr_matrix(meta_df.values)])

    # Urgency prediction
    urg_pred = urg_model.predict(urg_feats)[0]
    urg_label = ['Low', 'Medium', 'High'][urg_pred]

    # Urgency probabilities
    urg_probs = urg_model.predict_proba(urg_feats)[0]

    # Rule-based urgency score (complement to ML)
    rule_score = urgency_score(text)

    # Escalation
    escalation = recommend_escalation(cat_pred, urg_label)

    return {
        'cleaned_text': cleaned,
        'category': cat_pred,
        'category_probs': cat_prob_dict,
        'urgency': urg_label,
        'urgency_score': rule_score,
        'urgency_probs': dict(zip(['Low', 'Medium', 'High'], urg_probs)),
        'escalation': escalation,
    }


# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title(" Grievance Classifier")
st.sidebar.markdown("---")

model_info = st.sidebar.container()
with model_info:
    cat_model, urg_model, vectorizer, metadata = load_models()
    if metadata:
        st.metric("Category Accuracy", f"{metadata.get('category_accuracy', 0):.1%}")
        st.metric("Urgency Accuracy", f"{metadata.get('urgency_accuracy', 0):.1%}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Sample")
st.sidebar.markdown("Click a sample to load it:")

for i, sample in enumerate(SAMPLE_COMPLAINTS):
    if st.sidebar.button(f"Sample {i+1}", key=f"sample_{i}", use_container_width=True):
        st.session_state['complaint_text'] = sample
        st.session_state['predict_trigger'] = True

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This system classifies public grievances by **category** and **urgency level**, "
    "then provides escalation recommendations. Built with TF-IDF + Logistic Regression."
)


# ── Main content ────────────────────────────────────────────────────
st.title(" Public Grievance Urgency Classification System")
st.markdown("---")

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    default_text = st.session_state.get('complaint_text', '')
    complaint_text = st.text_area(
        "Enter your complaint:",
        value=default_text,
        height=180,
        placeholder="Describe your grievance in detail...",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("###  Enter Complaint")
    st.markdown("Describe the issue you're facing clearly.")
    predict_btn = st.button(" Analyze Complaint", type="primary", use_container_width=True)
    clear_btn = st.button(" Clear", use_container_width=True)

if clear_btn:
    st.session_state['complaint_text'] = ''
    st.session_state['predict_trigger'] = False
    st.rerun()

st.markdown("---")

# Prediction
should_predict = predict_btn or st.session_state.get('predict_trigger', False)

if should_predict and complaint_text and len(complaint_text.strip()) > 10:
    if cat_model is None:
        st.error("Models not loaded. Run `python3 src/train.py` first.")
        st.stop()

    with st.spinner("Analyzing complaint..."):
        result = predict_complaint(complaint_text, cat_model, urg_model, vectorizer)

    # ── Results display ────────────────────────────────────────────
    st.markdown("## 📊 Analysis Results")
    st.markdown("---")

    # Top row: Category + Urgency + Escalation
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.markdown("###  Category")
        cat = result['category']
        st.markdown(
            f"<div style='background:#f0f2f6; padding:15px; border-radius:10px; text-align:center;'>"
            f"<h2 style='margin:0;'>{cat}</h2>"
            f"<p style='margin:5px 0 0 0; color:#666; font-size:0.9em;'>{get_category_department(cat)}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown("###  Urgency Level")
        urg = result['urgency']
        color = URGENCY_COLORS.get(urg, "#666")
        emoji = URGENCY_EMOJIS.get(urg, "")
        bg_color = "#f8d7da" if urg == "High" else "#fff3cd" if urg == "Medium" else "#d4edda"
        st.markdown(
            f"<div style='background:{bg_color}; padding:15px; border-radius:10px; text-align:center;'>"
            f"<h2 style='margin:0; color:{color};'>{emoji}</h2>"
            f"<p style='margin:5px 0 0 0; color:#666; font-size:0.9em;'>"
            f"Score: {result['urgency_score']:.1%}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with res_col3:
        st.markdown("### ⏱ Escalation")
        esc = result['escalation']
        st.markdown(
            f"<div style='background:#f0f2f6; padding:15px; border-radius:10px; text-align:center;'>"
            f"<p style='margin:0; font-weight:bold; font-size:1.1em;'>{esc['escalate_to']}</p>"
            f"<p style='margin:5px 0 0 0; color:{color}; font-weight:bold;'>⏰ {esc['timeline']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Urgency score bar
    st.markdown("### 📈 Urgency Score")
    score = result['urgency_score']
    bar_color = "red" if score > 0.6 else "orange" if score > 0.3 else "green"
    st.progress(score, text=f"{score:.0%} urgency score")
    st.caption("Based on keyword density, exclamation frequency, and text structure analysis.")

    # Confidence scores
    st.markdown("### 🎯 Prediction Confidence")

    conf_col1, conf_col2 = st.columns(2)

    with conf_col1:
        st.markdown("**Category Confidence**")
        cat_probs = result['category_probs']
        sorted_cats = sorted(cat_probs.items(), key=lambda x: -x[1])[:5]
        cat_df = pd.DataFrame({
            'Category': [c for c, _ in sorted_cats],
            'Confidence': [p for _, p in sorted_cats],
        })
        st.dataframe(cat_df, hide_index=True, use_container_width=True)

    with conf_col2:
        st.markdown("**Urgency Confidence**")
        urg_probs = result['urgency_probs']
        urg_df = pd.DataFrame({
            'Level': list(urg_probs.keys()),
            'Confidence': list(urg_probs.values()),
        })
        st.dataframe(urg_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Action card
    st.markdown(f"###   Recommended Action")
    if result['urgency'] == "High":
        st.error(result['escalation']['action_text'])
    elif result['urgency'] == "Medium":
        st.warning(result['escalation']['action_text'])
    else:
        st.success(result['escalation']['action_text'])

    # Full response
    with st.expander("📄 View Full Analysis Report"):
        report = generate_response(complaint_text, result['category'],
                                    result['urgency'], result['escalation'])
        st.text(report)

elif should_predict:
    st.warning("Please enter a complaint description (at least 10 characters).")

# ── Footer info ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Model Performance")
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

if metadata:
    with perf_col1:
        st.metric("Category Accuracy", f"{metadata.get('category_accuracy', 0):.1%}")
    with perf_col2:
        st.metric("Category F1 (weighted)", f"{metadata.get('category_f1', 0):.3f}")
    with perf_col3:
        st.metric("Urgency Accuracy", f"{metadata.get('urgency_accuracy', 0):.1%}")
    with perf_col4:
        st.metric("Urgency F1 (weighted)", f"{metadata.get('urgency_f1', 0):.3f}")
else:
    st.info("Model metadata not loaded. Performance metrics unavailable.")

st.markdown("---")
st.caption(
    "Built with TF-IDF + Logistic Regression | "
    "Dataset: Government of India Grievance Report (Kaggle) | "
    "Capstone Project - AIML"
)
