# Public Grievance Urgency Classification System

Classify public grievances by **category** and **urgency level** with automated **escalation recommendations**.

**Track:** AIML  
**Duration:** 10 Days  
**Domain:** Public service & e-governance

---

##   Overview

Citizens file millions of grievances to government portals. This system automates triaging:

1. **Predicts complaint category** (8 types including Water Supply, Healthcare, Public Safety, etc.)
2. **Determines urgency level** (High / Medium / Low)
3. **Recommends escalation** (who to contact, timeline, action item)

**Accuracy:** Category 95.6% | Urgency 99.4% (Logistic Regression)

> **Note:** Urgency labels are generated from keyword rules, so the urgency model's high accuracy reflects learning those rules rather than independent urgency assessment. See [limitations](#limitations) for details.

---

##   Quick Start

```bash
# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train models (or use pre-trained in models/)
python3 src/train.py

# Launch the Streamlit app
streamlit run app/app.py
```

---

##   Project Structure

```
├── app/
│   └── app.py               # Streamlit demo UI
├── data/
│   └── build_dataset.py      # Kaggle data downloader + processor
├── docs/figures/             # EDA plots + confusion matrices
├── models/                   # Trained model pickles
├── src/
│   ├── config.py             # Constants & category definitions
│   ├── preprocessing.py      # Text cleaning + urgency rules
│   ├── features.py           # TF-IDF + meta feature extraction
│   ├── models.py             # Classifier classes (LR, RF)
│   ├── train.py              # Full training pipeline
│   └── escalation.py         # Escalation matrix (8×3)
├── tests/                    # Unit tests (pytest)
├── requirements.txt
├── README.md
└── REPORT.md                 # Full project report
```

---

##   ML Pipeline

```
Raw Text → Clean → TF-IDF (3K features) → Logistic Regression → Category + Urgency
                                                  ↓
                                    Escalation Engine (8×3 matrix)
                                                  ↓
                                    Streamlit Dashboard
```

**Feature set:**
- TF-IDF vectorization: 3,000 unigram+bigram features
- Meta features: text length, word count, uppercase ratio, punctuation count, urgency keyword density

**Models compared:**
- Logistic Regression (selected — 95.6% cat, 99.4% urg F1)
- Random Forest (92.9% cat, 97.9% urg F1)

---

##   Escalation Engine

8 categories × 3 urgency levels = 24 routing rules

| Category | High (CRITICAL) | Medium | Low (Routine) |
|---|---|---|---|
| Water Supply | Municipal Commissioner / 24h | Zonal Officer / 48h | Local Office / 7d |
| Road & Transport | PWD Minister / 12h | PWD Engineer / 3d | Maintenance / 7d |
| Electricity | Power Corp MD / 4h | Circle Engineer / 24h | Local Board / 5d |
| Healthcare | Hospital Director / 2h | Med Supdt / 24h | Admin / 7d |
| Public Safety | Police Commissioner / 1h | SHO / 4h | Local Station / 3d |
| Education | Education Secretary / 2d | DEO / 5d | School Admin / 10d |
| Housing | Housing Board Chair / 3d | Zonal Officer / 7d | Local Office / 14d |
| Other | Dept Secretary / 3d | Section Officer / 7d | Helpdesk / 15d |

---

##   Dataset

- **Source:** Government of India Grievance Report (CPGRAMS via Kaggle)
- **Size:** 30,132 processed records across 8 categories
- **Classes:** Public Safety (55.9%), Other (13.8%), Road & Transport (12.2%), Housing (6.9%), Healthcare (4.1%), Education (3.5%), Electricity (2.7%), Water Supply (0.9%)

---

##   Demo

The Streamlit app lets you:
- Type any grievance or pick from 10 sample complaints
- See real-time category + urgency prediction
- View confidence scores for all classes
- Get escalation recommendations with timeline
- Flag critical complaints for immediate action

```bash
streamlit run app/app.py
```

---

##   Tech Stack

| Component | Tool |
|---|---|
| ML | scikit-learn (TF-IDF + Logistic Regression) |
| UI | Streamlit |
| Viz | matplotlib, seaborn |
| Data | pandas, numpy |
| Source | CPGRAMS (Kaggle) |

---

##   Limitations

1. **Urgency labels are rule-based** — not human-annotated. The urgency model learns to replicate keyword patterns rather than assess genuine urgency. Accuracy numbers reflect consistency with the rule system, not real-world triage performance.
2. **Category imbalance** — Public Safety dominates (56%), minority categories have fewer training samples.
3. **English-only** — Optimized for English; India has 22 official languages.
4. **No temporal features** — Complaint age and re-filing patterns are not considered.
5. **Data freshness** — Based on historical CPGRAMS data; emerging issues may differ.

---

##   License

MIT
