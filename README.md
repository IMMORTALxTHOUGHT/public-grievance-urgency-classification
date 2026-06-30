# Public Grievance Urgency Classification System

Classify public grievances by category and urgency, with escalation recommendations.

**Track:** AIML  
**Duration:** 10 Days  
**Domain:** Public service and civic operations

---

## Overview

Government grievance systems receive thousands of complaints daily. This project builds a text classification pipeline that:

1. Predicts the **complaint category** (Water Supply, Road & Transport, Electricity, etc.)
2. Assigns an **urgency level** (High / Medium / Low)
3. Provides **escalation recommendations** (who to contact, how fast)

Built with Python, Scikit-learn, Pandas, and Streamlit.

---

## Project Structure

```
├── data/               # Dataset (raw + processed)
├── notebooks/          # EDA and exploration
├── src/                # Core ML pipeline
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   └── escalation.py
├── app/                # Streamlit demo app
├── models/             # Trained model pickles
├── docs/               # Report, figures, presentation
├── requirements.txt
└── README.md
```

---

*Full documentation coming soon.*
