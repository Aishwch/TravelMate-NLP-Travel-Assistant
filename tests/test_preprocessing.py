"""
Unit tests for src/preprocessing.py
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import (
    clean_text,
    tokenize,
    remove_stopwords,
    lemmatize_tokens,
    stem_tokens,
    preprocess_query
)


class TestNLPPreprocessing(unittest.TestCase):

    def test_clean_text(self):
        raw = "I  can't   visit \n\n Goa!!  "
        cleaned = clean_text(raw)
        self.assertIn("can not", cleaned)
        self.assertNotIn("  ", cleaned)

    def test_tokenize_preserves_travel_tokens(self):
        text = "I have ₹8000 and 3 days for a trip to Goa."
        tokens = tokenize(text)
        self.assertIn("₹8000", tokens)
        self.assertIn("3", tokens)
        self.assertIn("days", tokens)
        self.assertIn("Goa", tokens)

    def test_stopword_removal_preserves_travel_context(self):
        tokens = ["I", "want", "to", "travel", "with", "my", "parents", "without", "crowds"]
        filtered = remove_stopwords(tokens, preserve_travel_context=True)
        # 'with' and 'without' are travel constraint words and must be preserved
        self.assertIn("with", [t.lower() for t in filtered])
        self.assertIn("without", [t.lower() for t in filtered])
        self.assertNotIn("my", [t.lower() for t in filtered])

    def test_lemmatization(self):
        tokens = ["beaches", "visited", "cities", "waterfalls"]
        lemmas = lemmatize_tokens(tokens)
        self.assertIn("beach", lemmas)
        self.assertIn("city", lemmas)

    def test_preprocess_query_pipeline(self):
        query = "I want to visit beautiful beaches in Goa!"
        result = preprocess_query(query)
        self.assertEqual(result["original_query"], query)
        self.assertTrue(len(result["tokens"]) > 0)
        self.assertTrue(len(result["lemmas"]) > 0)
        self.assertIn("goa", result["normalized_text"].lower())


if __name__ == "__main__":
    unittest.main()
