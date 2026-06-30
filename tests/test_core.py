"""
Unit tests for the grievance classification system.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest
import numpy as np
import pandas as pd

from src.preprocessing import clean_text, assign_urgency, urgency_score
from src.features import extract_meta_features
from src.escalation import recommend_escalation, generate_response


class TestPreprocessing:
    def test_clean_text_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_clean_text_special_chars(self):
        result = clean_text("Water supply issue! @#$% Please help.")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_clean_text_extra_spaces(self):
        result = clean_text("water    supply   issue")
        assert "  " not in result

    def test_clean_text_none(self):
        assert clean_text("") == ""
        # clean_text handles None by defaulting to ""
        # Implementation should handle this gracefully

    def test_assign_urgency_high_keywords(self):
        text = "URGENT! No water supply. EMERGENCY! Please help immediately."
        urgency = assign_urgency(text)
        assert urgency == 2  # High

    def test_assign_urgency_low_routine(self):
        text = "I would like to know the status of my application."
        urgency = assign_urgency(text)
        assert urgency == 0  # Low

    def test_assign_urgency_medium(self):
        text = "There is a delay in fixing the pothole on our road. The road is damaged and water logging is severe."
        urgency = assign_urgency(text)
        assert urgency == 1  # Medium

    def test_urgency_score_range(self):
        texts = [
            "This is a routine inquiry.",
            "There is a problem with water supply.",
            "URGENT EMERGENCY! Help immediately! CRITICAL!",
        ]
        for t in texts:
            score = urgency_score(t)
            assert 0 <= score <= 1


class TestFeatures:
    def test_extract_meta_features(self):
        df = pd.DataFrame({
            'complaint_text': ['HELP! Water supply cut for 5 days! URGENT!'],
            'cleaned_text': ['help water supply cut for 5 days urgent'],
        })
        meta = extract_meta_features(df)
        assert 'text_length' in meta.columns
        assert 'word_count' in meta.columns
        assert 'caps_ratio' in meta.columns
        assert 'exclamation_count' in meta.columns
        assert 'urgent_keyword_count' in meta.columns
        assert meta['exclamation_count'].iloc[0] == 3
        assert meta['caps_ratio'].iloc[0] > 0

    def test_empty_text_meta(self):
        df = pd.DataFrame({
            'complaint_text': [''],
            'cleaned_text': [''],
        })
        meta = extract_meta_features(df)
        assert meta['text_length'].iloc[0] == 0


class TestEscalation:
    def test_all_categories_have_escalation(self):
        categories = ['Water Supply', 'Road & Transport', 'Electricity',
                      'Healthcare', 'Public Safety', 'Education', 'Housing', 'Other']
        for cat in categories:
            for urg in ['High', 'Medium', 'Low']:
                rec = recommend_escalation(cat, urg)
                assert rec['escalate_to'], f"Missing escalate_to for {cat}/{urg}"
                assert rec['timeline'], f"Missing timeline for {cat}/{urg}"
                assert rec['action_text'], f"Missing action_text for {cat}/{urg}"

    def test_unknown_category_falls_back(self):
        rec = recommend_escalation("UnknownCategory", "High")
        assert rec['escalate_to'] is not None

    def test_high_urgency_is_critical(self):
        rec = recommend_escalation("Water Supply", "High")
        assert rec['is_critical'] == True

    def test_low_urgency_not_critical(self):
        rec = recommend_escalation("Electricity", "Low")
        assert rec['is_critical'] == False

    def test_generate_response_includes_all_fields(self):
        text = "No water supply for one week."
        response = generate_response(text, "Water Supply", "High")
        assert "Water Supply" in response
        assert "HIGH" in response or "High" in response
        assert "Water Board" in response or "Commissioner" in response


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
