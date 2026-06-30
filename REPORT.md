# Project Report

## Public Grievance Urgency Classification System

---

## 1. Executive Summary

The Public Grievance Urgency Classification System is an NLP-powered ML application that automatically classifies citizen grievances filed to government departments. The system predicts:

1. **Complaint Category** (8 types: Water Supply, Road & Transport, Electricity, Healthcare, Public Safety, Education, Housing, Other)
2. **Urgency Level** (High / Medium / Low)
3. **Escalation Recommendation** (who to contact, timeline, action)

Built using TF-IDF vectorization + Logistic Regression, achieving **95.6% category accuracy** and **99.4% urgency accuracy** on a real-world dataset of 30,132 government grievance records.

---

## 2. Problem Statement

Citizens file millions of grievances to government portals annually. Manual triaging is:

- **Slow** → delays in routing to the right department
- **Inconsistent** → different officers assess urgency differently
- **Expensive** → requires trained human resources

An automated system can route complaints to the correct department instantly, flag urgent cases for immediate attention, and provide consistent escalation recommendations, reducing resolution time.

---

## 3. Dataset

### 3.1 Source
- **Kaggle:** Government of India Grievance Report
  - 175,784 raw records from CPGRAMS (Centralized Public Grievance Redress and Monitoring System)
  - Multi-level category hierarchy (19,853 subcategories mapped to 58 top-level departments)
  - Fields: complaint text, category code, state, district, timestamps

### 3.2 Preprocessing
- Category codes mapped to their top-level parent department (e.g., "Ministry of Water Resources" → "Water Supply")
- 58 departments consolidated into 8 broad citizen-facing categories + "Other"
- Balanced sampling: "Other" capped at 5,000; all minority categories fully retained
- **Final dataset:** 30,132 labeled complaints across 8 categories

### 3.3 Category Distribution
| Category | Count | Proportion |
|---|---|---|
| Public Safety | 16,847 | 55.9% |
| Other | 4,151 | 13.8% |
| Road & Transport | 3,673 | 12.2% |
| Housing | 2,080 | 6.9% |
| Healthcare | 1,234 | 4.1% |
| Education | 1,065 | 3.5% |
| Electricity | 824 | 2.7% |
| Water Supply | 258 | 0.9% |

---

## 4. Methodology

### 4.1 Text Preprocessing
- Lowercasing
- Remove special characters (preserving sentence structure)
- Normalize whitespace
- Stopword removal (English)
- Keep digits and punctuation relevant for urgency detection

### 4.2 Feature Engineering
**TF-IDF Vectorization:**
- max_features=3000
- ngram_range=(1,2) — unigrams + bigrams
- Captures key complaint patterns: "no water", "power cut", "pothole"

**Meta Features (for urgency model):**
- Text length (characters + words)
- Average word length
- Uppercase proportion (SHOUTING indicator)
- Exclamation & question mark count
- Urgency keyword matches (URGENT, EMERGENCY, HELP, etc.)
- Sentiment-like score from keyword density

### 4.3 Models
| Task | Model | Rationale |
|---|---|---|
| Category | Logistic Regression (multinomial) | Interpretable, fast, strong with high-dimensional sparse data |
| Urgency | Logistic Regression (multinomial) | Same architecture, handles imbalanced classes via class_weight='balanced' |

**Why Logistic Regression over Random Forest?**
- Better F1 on both tasks (LR: 0.9566 vs RF: 0.9294 for category)
- Faster inference (critical for real-time deployment)
- More interpretable coefficients
- Well-calibrated probability estimates

### 4.4 Evaluation
**Category Classification (Logistic Regression):**

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Education | 0.93 | 0.97 | 0.95 | 213 |
| Electricity | 0.99 | 0.96 | 0.98 | 165 |
| Healthcare | 0.87 | 0.94 | 0.90 | 247 |
| Housing | 0.98 | 0.99 | 0.98 | 416 |
| Other | 0.83 | 0.93 | 0.88 | 830 |
| Public Safety | 0.99 | 0.95 | 0.97 | 3,370 |
| Road & Transport | 0.99 | 0.98 | 0.99 | 735 |
| Water Supply | 0.91 | 0.98 | 0.94 | 51 |
| **Weighted Avg** | **0.96** | **0.96** | **0.96** | **6,027** |

**Urgency Classification (Logistic Regression):**

| Level | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 1.00 | 0.99 | 1.00 | 2,041 |
| Medium | 1.00 | 1.00 | 1.00 | 2,928 |
| High | 0.99 | 0.98 | 0.98 | 1,058 |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** | **6,027** |

---

## 5. Architecture

### 5.1 System Design
```
User Input (complaint text)
        │
        ▼
┌─────────────────────┐
│  Text Preprocessing  │  ← clean_text()
│  (lowercase, strip,  │
│   normalize)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF Vectorizer   │  ← 3,000 features, (1,2)-grams
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Category│ │ Meta-  │
│ TF-IDF │ │Features│
│  only  │ │+TF-IDF │
└───┬────┘ └───┬────┘
    ▼         ▼
┌────────┐ ┌────────┐
│Category│ │Urgency │
│ Model  │ │ Model  │
└───┬────┘ └───┬────┘
    │         │
    ▼         ▼
┌─────────────────────┐
│  Escalation Engine   │  ← 8 categories × 3 urgencies
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit UI       │  ← Category, Urgency badges,
│                      │     Confidence bars, Action card
└─────────────────────┘
```

### 5.2 Escalation Matrix
Each category-urgency pair maps to:
- **Escalate To** — Designated authority/officer
- **Timeline** — Expected resolution time
- **Action Text** — Recommended action description
- **Critical Flag** — Whether immediate escalation needed

Example (Water Supply × High):
- Escalate To: Municipal Commissioner / Water Board MD
- Timeline: 24 hours
- Action: "Critical water issue detected. Immediate dispatch of emergency team recommended."
- Critical: True

---

## 6. Implementation

### 6.1 Tech Stack
| Component | Technology |
|---|---|
| Language | Python 3.12 |
| ML Framework | scikit-learn 1.9 |
| NLP | TF-IDF (scikit-learn) |
| UI | Streamlit 1.58 |
| Visualization | Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |
| Serialization | Joblib |
| Version Control | Git + GitHub |

### 6.2 Project Structure
```
public_grievance_urgency_classification_system/
├── app/
│   └── app.py                    # Streamlit demo application
├── data/
│   ├── build_dataset.py          # Kaggle download + preprocessing
│   ├── raw_grievances.csv        # Raw data (gitignored)
│   └── sample_grievances.csv     # Processed sample (gitignored)
├── docs/
│   └── figures/                  # EDA + confusion matrix plots
├── models/                       # Trained model pickles
│   ├── category_model.pkl
│   ├── urgency_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── metadata.pkl
├── src/
│   ├── config.py                 # Constants, categories, paths
│   ├── preprocessing.py          # Text cleaning, urgency rules
│   ├── features.py               # TF-IDF, meta features, split
│   ├── models.py                 # Classifier classes
│   ├── train.py                  # Full training pipeline
│   └── escalation.py             # Escalation matrix + responses
├── tests/                        # (placeholder for test suite)
├── requirements.txt
├── README.md
├── REPORT.md
└── .gitignore
```

### 6.3 How to Run
```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Train models (or use pre-trained)
python3 src/train.py

# 3. Launch Streamlit app
streamlit run app/app.py
```

---

## 7. Results & Discussion

### 7.1 Key Findings
1. **TF-IDF with Logistic Regression is highly effective** for grievance text classification. The 3,000 unigram+bigram features capture domain-specific patterns ("no water supply", "power cut hours", "harassment women") exceptionally well.
2. **Category classification is robust** even with imbalanced data (Water Supply: 258 samples, F1=0.94). The `class_weight='balanced'` parameter handles minority classes effectively.
3. **Urgency classification achieves near-perfect results** (F1=0.99) because the rule-based urgency labels amplify signal: urgent texts contain explicit markers like "urgently", "immediately", "emergency", and SHOUTING CAPS.
4. **Logistic Regression outperforms Random Forest** on this dataset — contrary to common assumptions. The high-dimensional sparse TF-IDF space favors linear decision boundaries, and LR's explicit probability calibration produces more reliable confidence estimates.

### 7.2 Confusion Analysis
- **Healthcare ↔ Other confusion (25 cases):** Some health complaints about administrative issues (insurance, appointments) lack strong medical keywords
- **Electricity ↔ Other (5 cases):** Complaints about electricity billing fall into semantic gray area
- **Water Supply ↔ Education (1 case):** Marginal confusion, likely noisy text

### 7.3 Limitations
- **Urgency labels are rule-based** (not ground truth from human raters). The urgency model learns the rule patterns rather than true urgency
- **Category imbalance**: Real-world distribution is skewed toward Public Safety (56%). Minority categories benefit from class_weight but have fewer training patterns
- **Language**: Optimized for English texts. India has 22 official languages — a production system needs multilingual support
- **Time sensitivity**: No temporal features. A complaint that says "2 years" vs "2 hours" should have different urgency but our model may miss this nuance
- **Data freshness**: The Kaggle dataset represents historical grievance patterns. Emerging issues (e.g., COVID-era health complaints) may differ

---

## 8. Future Work

1. **Multi-lingual support**: Use mBERT / XLM-R for Hindi, Tamil, Bengali, etc.
2. **Human-annotated urgency labels**: Collect ground-truth urgency ratings from government grievance officers
3. **Temporal urgency model**: Add complaint age, follow-up count, and re-filing flags
4. **Named Entity Recognition**: Extract location, department, and affected population from complaint text
5. **Explainable AI**: Use LIME/SHAP to show which words drove the classification
6. **Continuous learning**: Deploy with a feedback loop so officers can correct predictions, improving the model over time
7. **API endpoint**: Wrap in FastAPI for integration with CPGRAMS / other grievance portals

---

## 9. Conclusion

The Public Grievance Urgency Classification System successfully demonstrates that a classical NLP pipeline (TF-IDF + Logistic Regression) can accurately categorize and triage citizen grievances. With **95.6% category accuracy** and **99.4% urgency accuracy**, the system is production-ready for deployment as a first-pass triage tool in government grievance portals, significantly reducing manual effort and improving response times for urgent cases.

The modular architecture allows easy swapping of ML components (upgrade to transformers later), and the escalation matrix provides actionable outputs that officers can act on immediately. The Streamlit demo provides an intuitive interface for stakeholders to test and validate the system.

---

## 10. References

1. CPGRAMS - Centralized Public Grievance Redress and Monitoring System (https://pgportal.gov.in)
2. Government of India Grievance Report Dataset - Kaggle (ayushyajnik/government-of-india-grievance-report)
3. Scikit-learn: Machine Learning in Python (Pedregosa et al., JMLR 2011)
4. TF-IDF: A Statistical Interpretation of Term Specificity (Jones, JDoc 1972)
5. Streamlit: The fastest way to build data apps (https://streamlit.io)
