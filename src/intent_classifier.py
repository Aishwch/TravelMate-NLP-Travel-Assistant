"""
TravelMate — Intent Classifier Inference Module
================================================
Loads the serialized TF-IDF vectorizer and Logistic Regression classifier
to provide fast runtime intent prediction with probability confidence scores,
multi-intent detection, and fallback handling.
"""

import os
import joblib
from typing import Dict, Tuple, List, Optional, Any
import numpy as np

from src.preprocessing import clean_text

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


class IntentClassifier:
    """
    Classifies natural language user queries into semantic travel intents.
    """

    def __init__(self, auto_train: bool = True):
        self.model = None
        self.vectorizer = None
        self.classes_ = []
        self._load_or_train(auto_train)

    def _load_or_train(self, auto_train: bool):
        """Loads serialized model artifacts, or trains if missing."""
        if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VEC_PATH)
                self.classes_ = list(self.model.classes_)
                return
            except Exception as e:
                print(f"Warning: Failed to load saved models ({e}). Retraining...")

        if auto_train:
            from src.train_intent_model import train_and_evaluate_model
            self.model, self.vectorizer, _ = train_and_evaluate_model()
            self.classes_ = list(self.model.classes_)
        else:
            raise FileNotFoundError("Intent classifier artifacts not found in models/.")

    def predict(self, query: str) -> str:
        """Predicts the single most likely intent for a query."""
        intent, _, _ = self.predict_with_confidence(query)
        return intent

    def predict_with_confidence(
        self, query: str, top_k: int = 5
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predicts intent along with confidence probability and top-K candidate distribution.
        Returns:
            (top_intent, top_confidence, top_k_dict)
        """
        cleaned = clean_text(query)
        if not cleaned:
            return "general_travel", 0.0, {}

        # Transform via TF-IDF
        vec = self.vectorizer.transform([cleaned])
        
        # Check if vector is all zeros (completely out of vocabulary tokens)
        if vec.nnz == 0:
            return "general_travel", 0.35, {"general_travel": 0.35}

        # Predict probabilities
        probabilities = self.model.predict_proba(vec)[0]
        top_indices = np.argsort(probabilities)[::-1]

        top_intent = self.classes_[top_indices[0]]
        top_confidence = float(probabilities[top_indices[0]])

        # Compile top-K distribution
        top_k_dist = {}
        for idx in top_indices[:top_k]:
            intent_name = self.classes_[idx]
            top_k_dist[intent_name] = round(float(probabilities[idx]), 4)

        return top_intent, round(top_confidence, 4), top_k_dist

    def detect_multi_intents(self, query: str, threshold_gap: float = 0.15) -> List[str]:
        """
        Detects multiple co-occurring intents in complex queries.
        For example: 'I'm going to Goa for 3 days. Suggest beaches, local food and nightlife.'
        -> ['attraction_recommendation', 'food_recommendation', 'activity_recommendation']
        """
        cleaned = query.lower()
        detected_intents = []

        # 1. Primary predicted intent
        primary, conf, top_dist = self.predict_with_confidence(query, top_k=4)
        detected_intents.append(primary)

        # 2. Check if secondary intent has high competitive probability
        items = list(top_dist.items())
        if len(items) > 1:
            second_intent, second_conf = items[1]
            if (conf - second_conf) <= threshold_gap and second_conf > 0.18:
                if second_intent not in detected_intents:
                    detected_intents.append(second_intent)

        # 3. Explicit multi-intent heuristic keywords
        if any(w in cleaned for w in ["food", "eat", "cuisine", "dishes", "restaurant"]) and "food_recommendation" not in detected_intents:
            detected_intents.append("food_recommendation")

        if any(w in cleaned for w in ["attraction", "places to see", "beaches", "sights", "forts"]) and "attraction_recommendation" not in detected_intents:
            detected_intents.append("attraction_recommendation")

        if any(w in cleaned for w in ["activities", "things to do", "fun", "nightlife", "watersports", "sports"]) and "activity_recommendation" not in detected_intents:
            detected_intents.append("activity_recommendation")

        if any(w in cleaned for w in ["how to reach", "travel from", "train to", "flight to", "bus to", "route"]) and "transportation" not in detected_intents:
            detected_intents.append("transportation")

        if any(w in cleaned for w in ["pack", "what to wear", "clothes", "shoes"]) and "packing_advice" not in detected_intents:
            detected_intents.append("packing_advice")

        return detected_intents
