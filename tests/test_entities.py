"""
Unit tests for src/entity_extractor.py
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_extractor import TravelEntityExtractor


class TestTravelEntityExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = TravelEntityExtractor()

    def test_destination_extraction_exact(self):
        query = "I want to visit Goa for a holiday."
        entities = self.extractor.extract_all_entities(query)
        self.assertEqual(entities["destination"], "Goa")

    def test_destination_fuzzy_matching(self):
        # 'Mumbay' -> Mumbai
        mumbai_query = "Flights from Mumbay"
        entities1 = self.extractor.extract_all_entities(mumbai_query)
        self.assertEqual(entities1["origin"], "Mumbai")

        # 'Mahabaleshwr' -> Mahabaleshwar
        maha_query = "Suggest hotels in Mahabaleshwr"
        entities2 = self.extractor.extract_all_entities(maha_query)
        self.assertEqual(entities2["destination"], "Mahabaleshwar")

        # 'Gokarn' -> Gokarna
        gokarna_query = "Beaches in Gokarn"
        entities3 = self.extractor.extract_all_entities(gokarna_query)
        self.assertEqual(entities3["destination"], "Gokarna")

    def test_duration_extraction(self):
        # Numeric days
        e1 = self.extractor.extract_all_entities("Plan a trip for 4 days")
        self.assertEqual(e1["duration_days"], 4)

        # Weekend
        e2 = self.extractor.extract_all_entities("Quick weekend trip")
        self.assertEqual(e2["duration_days"], 2)

        # Word numbers
        e3 = self.extractor.extract_all_entities("I have three days for travel")
        self.assertEqual(e3["duration_days"], 3)

    def test_budget_extraction(self):
        # Currency symbol
        e1 = self.extractor.extract_all_entities("My budget is ₹12000 for the trip")
        self.assertEqual(e1["budget_inr"], 12000)

        # Rs. with commas
        e2 = self.extractor.extract_all_entities("I have Rs. 8,000 for travel")
        self.assertEqual(e2["budget_inr"], 8000)

        # 10k shorthand
        e3 = self.extractor.extract_all_entities("Can I travel under 10k?")
        self.assertEqual(e3["budget_inr"], 10000)

    def test_travel_group_extraction(self):
        # Friends
        e1 = self.extractor.extract_all_entities("I am travelling with 3 friends")
        self.assertEqual(e1["travel_group"], "Friends")
        self.assertEqual(e1["people_count"], 4)  # 3 friends + user

        # Parents
        e2 = self.extractor.extract_all_entities("Travelling with my parents and they can't walk too much")
        self.assertEqual(e2["travel_group"], "Parents / Family")
        self.assertEqual(e2["walking_preference"], "Low")

        # Solo
        e3 = self.extractor.extract_all_entities("I want to do a solo trip")
        self.assertEqual(e3["travel_group"], "Solo")
        self.assertEqual(e3["people_count"], 1)

    def test_crowd_constraint_extraction(self):
        query = "I want a peaceful place away from crowds"
        entities = self.extractor.extract_all_entities(query)
        self.assertEqual(entities["crowd_preference"], "Low")

    def test_destination_switching_detection(self):
        query = "Actually, forget Goa. What about Jaipur?"
        entities = self.extractor.extract_all_entities(query)
        self.assertEqual(entities["destination"], "Jaipur")
        self.assertEqual(entities["negated_destination"], "Goa")
        self.assertTrue(entities["is_destination_switch"])

    def test_origin_destination_separation(self):
        query = "I want to travel from Mumbai to Goa for 3 days"
        entities = self.extractor.extract_all_entities(query)
        self.assertEqual(entities["origin"], "Mumbai")
        self.assertEqual(entities["destination"], "Goa")
        self.assertEqual(entities["duration_days"], 3)


if __name__ == "__main__":
    unittest.main()
