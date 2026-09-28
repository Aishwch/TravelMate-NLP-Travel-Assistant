"""
Unit tests for src/intent_classifier.py
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent_classifier import IntentClassifier


class TestIntentClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = IntentClassifier(auto_train=False)

    def test_model_loaded(self):
        self.assertIsNotNone(self.classifier.model)
        self.assertIsNotNone(self.classifier.vectorizer)
        self.assertTrue(len(self.classifier.classes_) > 20)

    def test_standard_intent_predictions(self):
        # Itinerary
        intent1, conf1, _ = self.classifier.predict_with_confidence("Plan a 3 day trip to Jaipur")
        self.assertEqual(intent1, "itinerary_planning")
        self.assertTrue(conf1 > 0.15)

        # Food
        intent2, conf2, _ = self.classifier.predict_with_confidence("What food should I try in Hyderabad?")
        self.assertEqual(intent2, "food_recommendation")

        # Transport
        intent3, conf3, _ = self.classifier.predict_with_confidence("How can I travel from Mumbai to Goa?")
        self.assertEqual(intent3, "transportation")

        # Packing
        intent4, conf4, _ = self.classifier.predict_with_confidence("What should I pack for Ladakh?")
        self.assertEqual(intent4, "packing_advice")

        # Greeting
        intent5, conf5, _ = self.classifier.predict_with_confidence("Hello there!")
        self.assertEqual(intent5, "greeting")

    def test_open_ended_unseen_query(self):
        # A complex sentence never seen verbatim in training data
        query = "I have a small budget, love photography, don't like crowded places, and want a 2 day trip."
        intent, conf, top_k = self.classifier.predict_with_confidence(query)
        self.assertIsNotNone(intent)
        self.assertTrue(conf > 0.0)
        self.assertTrue(len(top_k) > 1)

    def test_multi_intent_detection(self):
        query = "I am going to Goa. Suggest beaches, local food and nightlife."
        multi_intents = self.classifier.detect_multi_intents(query)
        self.assertIn("food_recommendation", multi_intents)
        self.assertTrue(any(i in multi_intents for i in ["attraction_recommendation", "activity_recommendation"]))


if __name__ == "__main__":
    unittest.main()
