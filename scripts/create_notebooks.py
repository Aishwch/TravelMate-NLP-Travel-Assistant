"""
Script to generate valid Jupyter Notebooks (.ipynb) for TravelMate:
1. notebooks/data_exploration.ipynb
2. notebooks/model_evaluation.ipynb
"""

import json
import os

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

# 1. data_exploration.ipynb
data_cells = [
    make_markdown_cell("""# 🌍 TravelMate - Exploratory Data Analysis (EDA)
### NLP-Based Intelligent Travel Assistance System

This notebook documents the exploratory data analysis conducted on the curated tourism datasets for TravelMate:
- **Destinations Dataset** (`data/processed/destinations.csv`)
- **Attractions Dataset** (`data/processed/attractions.csv`)
- **Budget Dataset** (`data/processed/budget.csv`)
- **Food & Cuisine Dataset** (`data/processed/food.csv`)
- **Transport Dataset** (`data/processed/transport.csv`)
"""),
    make_code_cell("""import os
import pandas as pd
import numpy as np

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DATA_DIR = os.path.join("..", "data", "processed")
print("Loading datasets from:", os.path.abspath(DATA_DIR))"""),
    make_markdown_cell("""## 1. Destination Profiles Analysis"""),
    make_code_cell("""dest_df = pd.read_csv(os.path.join(DATA_DIR, "destinations.csv"))
print(f"Total Destinations: {len(dest_df)}")
dest_df.head()"""),
    make_code_cell("""print("Destination Data Schema & Missing Values:")
print(dest_df.info())
print("\\nMissing values per column:\\n", dest_df.isnull().sum())"""),
    make_code_cell("""print("Distribution by State:")
print(dest_df['state'].value_counts())

print("\\nDestinations by Primary Category:")
print(dest_df['category'].value_counts())"""),
    make_code_cell("""print("Budget and Duration Summary Statistics:")
dest_df[['estimated_cost', 'duration_days', 'rating']].describe()"""),
    make_markdown_cell("""## 2. Attractions & Points of Interest Analysis"""),
    make_code_cell("""attr_df = pd.read_csv(os.path.join(DATA_DIR, "attractions.csv"))
print(f"Total Attractions: {len(attr_df)}")
print(f"Unique Destinations Covered: {attr_df['destination'].nunique()}")
attr_df.head()"""),
    make_code_cell("""print("Attractions by Category:")
print(attr_df['category'].value_counts().head(10))

print("\\nTop 5 Highest Rated Attractions:")
print(attr_df.sort_values(by='rating', ascending=False)[['attraction', 'destination', 'category', 'rating', 'entry_fee']].head(5))"""),
    make_markdown_cell("""## 3. Budget & Expense Analysis"""),
    make_code_cell("""budget_df = pd.read_csv(os.path.join(DATA_DIR, "budget.csv"))
print(f"Budget Records: {len(budget_df)}")
budget_df.head()"""),
    make_code_cell("""# Calculate average daily expenditure per destination
budget_df['daily_total_estimate'] = (
    budget_df['accommodation_cost_per_day'] + 
    budget_df['food_cost_per_day'] + 
    budget_df['local_transport_cost_per_day'] + 
    budget_df['sightseeing_cost_per_day']
)
print("Top 5 Most Budget-Friendly Destinations (Daily Basis):")
print(budget_df.sort_values('daily_total_estimate')[['destination', 'daily_total_estimate', 'budget_tier']].head())

print("\\nTop 5 Premium / Luxury Destinations:")
print(budget_df.sort_values('daily_total_estimate', ascending=False)[['destination', 'daily_total_estimate', 'budget_tier']].head())"""),
    make_markdown_cell("""## 4. Culinary and Transportation Coverage"""),
    make_code_cell("""food_df = pd.read_csv(os.path.join(DATA_DIR, "food.csv"))
print(f"Total Food Items: {len(food_df)}")
print("Dietary Type Breakdown:")
print(food_df['type'].value_counts())"""),
    make_code_cell("""trans_df = pd.read_csv(os.path.join(DATA_DIR, "transport.csv"))
print(f"Transport Records: {len(trans_df)}")
print("Routes documented from Mumbai / Pune:")
print(trans_df[['origin', 'destination', 'mode', 'duration', 'approx_cost']].head(10))"""),
    make_markdown_cell("""## 5. Key EDA Takeaways for NLP Pipeline
1. **Rich Entity Coverage:** Across 36 major travel destinations with diverse state representation (Maharashtra, Goa, Rajasthan, Himachal Pradesh, Kerala, Karnataka, etc.).
2. **Text Richness:** Destinations and attractions feature detailed descriptions, ideal for semantic vector embeddings via `all-MiniLM-L6-v2`.
3. **Structured Constraints:** Strong variance in budget per day (from ₹1,000 budget tier to ₹4,500 luxury tier) enables multi-constraint optimization.
""")
]

# 2. model_evaluation.ipynb
eval_cells = [
    make_markdown_cell("""# 📊 TravelMate - Intent Model & NLP Evaluation
### Machine Learning & Semantic Retrieval Evaluation

This notebook presents the evaluation of:
1. **TF-IDF + Logistic Regression Intent Classifier** across 26 travel intents.
2. **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-Score.
3. **Semantic Similarity Testing:** Embedding retrieval with `all-MiniLM-L6-v2`.
4. **End-to-End Test Suite:** Evaluating unseen natural language queries.
"""),
    make_code_cell("""import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(".."))
from src.preprocessing import clean_text_for_ml, extract_tokens
from src.semantic_search import SemanticSearchEngine"""),
    make_markdown_cell("""## 1. Load Training Data and Evaluation Metrics"""),
    make_code_cell("""MODEL_DIR = os.path.join("..", "models")
metrics_path = os.path.join(MODEL_DIR, "intent_metrics.json")

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

print(f"Model: {metrics.get('model_type')}")
print(f"Overall Accuracy: {metrics.get('accuracy') * 100:.2f}%")
print(f"Macro Precision:  {metrics.get('macro_precision') * 100:.2f}%")
print(f"Macro Recall:     {metrics.get('macro_recall') * 100:.2f}%")
print(f"Macro F1-Score:   {metrics.get('macro_f1') * 100:.2f}%")
print(f"Total Intents:    {metrics.get('num_classes')}")"""),
    make_code_cell("""# Intent-wise Detailed Report
report_df = pd.DataFrame(metrics['detailed_report']).transpose()
report_df.head(15)"""),
    make_markdown_cell("""## 2. Intent Classifier Inference on Unseen Queries"""),
    make_code_cell("""classifier = joblib.load(os.path.join(MODEL_DIR, "intent_classifier.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

test_queries = [
    "I have ₹8000 and 3 days from Mumbai, where should I go?",
    "What are the best attractions to see in Jaipur?",
    "How can I reach Goa from Mumbai?",
    "What authentic food should I eat in Hyderabad?",
    "Which is better for a weekend, Lonavala or Matheran?",
    "What should I pack for a monsoon trek?",
    "Plan a 4-day itinerary for Rajasthan",
    "I want somewhere quiet and peaceful with nature"
]

results = []
for q in test_queries:
    cleaned = clean_text_for_ml(q)
    vec = vectorizer.transform([cleaned])
    pred = classifier.predict(vec)[0]
    probs = classifier.predict_proba(vec)[0]
    conf = np.max(probs)
    results.append({"Query": q, "Predicted Intent": pred, "Confidence": f"{conf:.2%}"})

pd.DataFrame(results)"""),
    make_markdown_cell("""## 3. Semantic Search Vector Similarity Evaluation"""),
    make_code_cell("""engine = SemanticSearchEngine()
engine.initialize()

print("Status:", engine.get_status())"""),
    make_code_cell("""query = "I want a quiet place away from crowds with beautiful nature and photography"
matches = engine.search_destinations(query, top_k=5)

print(f"Query: '{query}'\\n")
print(f"{'Destination':<20} | {'Score':<8} | {'Category':<15} | {'State'}")
print("-" * 65)
for m in matches:
    print(f"{m['destination']:<20} | {m['similarity_score']:.4f}   | {m.get('category',''):<15} | {m.get('state','')}")"""),
    make_markdown_cell("""## 4. Multi-Constraint Recommendation Verification"""),
    make_code_cell("""from src.recommender import RecommendationEngine

recommender = RecommendationEngine()
recommender.initialize()

# Complex unseen query: 2 days, ₹7000, nature/peaceful, low crowd, origin Mumbai
recs = recommender.recommend(
    query="I'm travelling from Mumbai for 2 days. Budget around 7000, love nature, peaceful, not crowded",
    entities={"duration": 2, "budget": 7000.0, "origin": "Mumbai"},
    preferences=["nature", "peaceful", "photography"],
    top_k=3
)

for i, r in enumerate(recs, 1):
    print(f"{i}. {r['destination']} (Score: {r['relevance_score']:.3f})")
    print(f"   Category: {r['category']} | Best Season: {r['best_season']}")
    for exp in r['explanations']:
        print(f"   {exp}")
    print()""")
]

os.makedirs("notebooks", exist_ok=True)
create_notebook(data_cells, os.path.join("notebooks", "data_exploration.ipynb"))
create_notebook(eval_cells, os.path.join("notebooks", "model_evaluation.ipynb"))
print("All notebooks created successfully!")
