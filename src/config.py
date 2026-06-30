"""
Configuration and constants for the Grievance Classification System.
"""
import re

# Paths
DATA_PATH = "data/raw_grievances.csv"
SAMPLE_PATH = "data/sample_grievances.csv"
MODEL_DIR = "models/"
FIGURE_DIR = "docs/figures/"

# Target categories for classification
CATEGORIES = [
    "Education",
    "Electricity",
    "Healthcare",
    "Housing",
    "Other",
    "Public Safety",
    "Road & Transport",
    "Water Supply",
]

# Category to department mapping for display
CATEGORY_DEPT = {
    "Education": "Ministry of Education / School Education Dept",
    "Electricity": "Ministry of Power / Electricity Board",
    "Healthcare": "Ministry of Health & Family Welfare",
    "Housing": "Housing & Urban Affairs / PWD",
    "Other": "Concerned Department (varies)",
    "Public Safety": "Home Affairs / Labour / Justice / Police",
    "Road & Transport": "Ministry of Road Transport / Railways / PWD",
    "Water Supply": "Water Resources Dept / Municipal Corporation",
}

# Urgency levels
URGENCY_LEVELS = ["Low", "Medium", "High"]
URGENCY_MAP = {"Low": 0, "Medium": 1, "High": 2}

# Urgency keyword patterns (case-insensitive)
HIGH_URGENCY_KEYWORDS = [
    "emergency", "accident", "death", "died", "fatal", "collapse",
    "collapsed", "fire", "flood", "burst", "life threatening", "critical",
    "immediate", "urgent", "hospitalise", "hospitalized", "suicide",
    "kidnap", "attack", "assault", "explosion", "poison", "outbreak",
    "sewage overflow", "water contamination", "no water", "pipe burst",
    "electrocution", "short circuit", "gas leak", "building collapse",
    "road cave", "bridge collapse", "child labour", "sexual",
    "molestation", "harassment at work", "unlawful detention",
    "police brutality", "encroachment", "illegal construction",
]

MEDIUM_URGENCY_KEYWORDS = [
    "delay", "delayed", "poor service", "not working", "broken",
    "shortage", "inadequate", "insufficient", "lack of", "no facility",
    "no supply", "irregular", "non-payment", "pending since",
    "not received", "not provided", "not available", "not sanctioned",
    "overcharging", "overpriced", "duplicate", "fraud", "corruption",
    "bribe", "bribery", "extortion", "misappropriation",
    "unauthorized", "forgery", "fake", "counterfeit", "defective",
    "damaged", "disrepair", "dilapidated", "unhygienic",
    "unclean", "stagnant water", "garbage", "waste not collected",
    "mosquito", "rodent", "stray dogs", "street light",
    "pothole", "road damage", "water logging", "drainage",
    "power cut", "voltage fluctuation", "low pressure",
    "no response", "no action", "repeated complaint",
    "multiple complaints", "follow up", "no update",
    "discrimination", "nepotism", "favouritism",
    "appointment", "transfer", "promotion", "salary",
    "pension", "gratuity", "leave", "overtime",
]

# Text preprocessing patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
PHONE_PATTERN = re.compile(r'\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
REDACTED_PATTERN = re.compile(r'\b[X]+[0-9X]*[X]+\b')  # e.g., X0X7X0X8X5X5
SPECIAL_CHARS = re.compile(r'[^a-zA-Z\s]')
MULTI_SPACE = re.compile(r'\s+')

# Model hyperparameters
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42
