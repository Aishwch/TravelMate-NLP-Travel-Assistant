"""
Unit tests for src/recommender.py
"""

import unittest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.semantic_search import SemanticSearchEngine
from src.retriever import HybridRetriever
from src.recommender import TravelRecommender
from src.preference_extractor import PreferenceExtractor


class TestTravelRecommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.destinations_df = pd.read_csv("data/processed/destinations.csv")
        cls.semantic_engine = SemanticSearchEngine()
        cls.retriever = HybridRetriever(cls.destinations_df, cls.semantic_engine)
        cls.recommender = TravelRecommender(cls.retriever)
        cls.pref_extractor = PreferenceExtractor()

    def test_multi_constraint_recommendation(self):
        query = "I have 4 days, ₹15000, I'm travelling with my parents, and I prefer historical and peaceful places."
        entities = {
            "destination": None,
            "duration_days": 4,
            "budget_inr": 15000,
            "travel_group": "Parents / Family",
            "walking_preference": "Low",
            "crowd_preference": "Low"
        }
        preferences = self.pref_extractor.extract_preferences(query)
        recs = self.recommender.recommend(query, entities, preferences, top_n=3)

        self.assertTrue(len(recs) > 0)
        top_dest = recs[0]
        self.assertIn("destination", top_dest)
        self.assertIn("composite_score", top_dest)
        self.assertIn("reasons", top_dest)
        self.assertTrue(len(top_dest["reasons"]) > 0)

    def test_peaceful_nature_recommendation(self):
        query = "I want somewhere peaceful with nature and low crowds near Mumbai"
        entities = {
            "destination": None,
            "origin": "Mumbai",
            "crowd_preference": "Low",
            "walking_preference": None
        }
        preferences = ["peaceful", "nature"]
        recs = self.recommender.recommend(query, entities, preferences, top_n=3)
        self.assertTrue(len(recs) > 0)
        # Matheran / Lonavala / Mahabaleshwar should rank highly
        rec_names = [r["destination"] for r in recs]
        self.assertTrue(any(d in rec_names for d in ["Matheran", "Mahabaleshwar", "Lonavala", "Gokarna"]))


if __name__ == "__main__":
    unittest.main()
