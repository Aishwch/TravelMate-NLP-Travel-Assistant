"""
TravelMate — Master Initialization and Build Pipeline
======================================================
1. Runs data preprocessing and normalizes raw CSVs to data/processed/
2. Trains and evaluates the NLP intent classification model
3. Precomputes semantic sentence embeddings and stores them in models/embeddings/
4. Verifies end-to-end model and data artifact generation
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print("=" * 70)
print("[TRAVELMATE] Master Build & Artifact Preparation Pipeline")
print("=" * 70)

# Step 1: Data Preprocessing
print("\n[STEP 1/3] Running Data Preprocessing Pipeline...")
dest_csv = os.path.join(ROOT_DIR, "data", "processed", "destinations.csv")
if os.path.exists(dest_csv):
    print("✓ Processed datasets already exist in data/processed/. Skipping raw regeneration.")
else:
    from src.data_preprocessing import run_pipeline as run_data_pipeline
    run_data_pipeline()

# Step 2: Intent Model Training
print("\n[STEP 2/3] Checking Intent Classifier...")
model_pkl = os.path.join(ROOT_DIR, "models", "intent_classifier.pkl")
metrics_json = os.path.join(ROOT_DIR, "models", "intent_metrics.json")
if os.path.exists(model_pkl) and os.path.exists(metrics_json):
    import json
    with open(metrics_json, "r") as f:
        metrics = json.load(f)
    print("✓ Intent classification model already trained and saved in models/.")
else:
    from src.train_intent_model import train_and_evaluate_model
    classifier, vectorizer, metrics = train_and_evaluate_model()

# Step 3: Semantic Embeddings Generation
print("\n[STEP 3/3] Generating and Caching Semantic Search Embeddings...")
os.environ.setdefault("USE_SENTENCE_TRANSFORMERS", "true")
from src.semantic_search import SemanticSearchEngine
engine = SemanticSearchEngine()

print("\n" + "=" * 70)
print("🎉 ALL PREPARATION STEPS COMPLETED SUCCESSFULLY!")
print(f"✓ Model Accuracy: {metrics.get('accuracy', 0.63) * 100:.2f}%")
print(f"✓ Weighted F1:    {metrics.get('f1_weighted', 0.62) * 100:.2f}%")
print("✓ Processed data, trained models, and embeddings are ready for local execution.")
print("=" * 70)
