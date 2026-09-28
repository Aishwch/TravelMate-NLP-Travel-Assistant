"""
Unit tests for multi-turn conversational context, destination switching,
low-confidence fallback, and open-ended queries in TravelMate.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chatbot import TravelMateAssistant


class TestChatbotContextAndOpenEnded(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.assistant = TravelMateAssistant()

    def setUp(self):
        # Reset conversation context between tests
        self.assistant.context.reset()

    def test_multi_turn_pronoun_resolution(self):
        # Turn 1: Mention destination
        r1 = self.assistant.respond("I want to visit Goa.")
        self.assertEqual(self.assistant.context.active_destination, "Goa")

        # Turn 2: Follow-up using 'there'
        r2 = self.assistant.respond("What can I do there?")
        self.assertIn("Goa", r2["response"])
        self.assertEqual(self.assistant.context.active_destination, "Goa")

    def test_destination_switching(self):
        # Turn 1: Initial destination
        self.assistant.respond("Tell me about Goa.")
        self.assertEqual(self.assistant.context.active_destination, "Goa")

        # Turn 2: Switch destination
        r2 = self.assistant.respond("Actually, forget Goa. What about Jaipur?")
        self.assertEqual(self.assistant.context.active_destination, "Jaipur")
        self.assertIn("Jaipur", r2["response"])

        # Turn 3: Follow-up refers to new destination
        r3 = self.assistant.respond("What can I do there?")
        self.assertEqual(self.assistant.context.active_destination, "Jaipur")
        self.assertIn("Jaipur", r3["response"])

    def test_gibberish_handling(self):
        # Nonsensical random string must not crash
        res = self.assistant.respond("asdfghjkl")
        self.assertIn("couldn't quite understand", res["response"].lower())

    def test_vague_query_clarification(self):
        # Vague trip requests should ask for clarification
        res = self.assistant.respond("plan a trip for me")
        self.assertIn("where", res["response"].lower())
        self.assertIn("how many days", res["response"].lower())

    def test_general_travel_questions(self):
        # Duration question
        r1 = self.assistant.respond("How many days are enough for Jaipur?")
        self.assertIn("days", r1["response"].lower())
        self.assertIn("Jaipur", r1["response"])

        # Senior friendliness / parents
        r2 = self.assistant.respond("Is Matheran suitable for parents?")
        self.assertIn("Matheran", r2["response"])
        self.assertIn("walking", r2["response"].lower())

        # Best season
        r3 = self.assistant.respond("What is the best season for Manali?")
        self.assertIn("Manali", r3["response"])
        self.assertIn("best time", r3["response"].lower())

        # Solo travel safety
        r4 = self.assistant.respond("What precautions should I take while travelling solo?")
        self.assertIn("safety", r4["response"].lower())

    def test_multi_constraint_open_ended_recommendation(self):
        query = "I'm travelling from Mumbai for 2 days. Have 7000 budget, love photography and nature, and want somewhere peaceful and not crowded."
        res = self.assistant.respond(query)
        self.assertTrue(len(res.get("recommended_destinations", [])) > 0)
        self.assertIn("recommended destinations", res["response"].lower())


if __name__ == "__main__":
    unittest.main()
