"""
Unit tests for src/itinerary.py
"""

import unittest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.itinerary import ItineraryGenerator


class TestItineraryGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.attractions_df = pd.read_csv("data/processed/attractions.csv")
        cls.food_df = pd.read_csv("data/processed/food.csv")
        cls.generator = ItineraryGenerator(cls.attractions_df, cls.food_df)

    def test_generate_3_day_itinerary_jaipur(self):
        itin = self.generator.generate_itinerary("Jaipur", days=3)
        self.assertTrue(itin["success"])
        self.assertEqual(len(itin["daily_plans"]), 3)

        # Check Day 1 slots
        day1 = itin["daily_plans"][0]
        self.assertEqual(day1["day"], 1)
        self.assertIsNotNone(day1["morning"])
        self.assertIsNotNone(day1["afternoon"])

    def test_format_as_text_markdown(self):
        itin = self.generator.generate_itinerary("Goa", days=2)
        text = self.generator.format_as_text(itin)
        self.assertIn("Customized Itinerary for Goa", text)
        self.assertIn("Day 1", text)
        self.assertIn("Day 2", text)
        self.assertIn("Morning", text)


if __name__ == "__main__":
    unittest.main()
