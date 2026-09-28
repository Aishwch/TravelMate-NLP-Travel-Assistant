# System Architecture — TravelMate

## Architecture Overview

TravelMate follows a modular, layered software architecture designed for maintainability, offline execution, and high testability.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                              │
│                                                                        │
│   Streamlit Web Interface (app.py)                                     │
│   ├─ 🏠 Home Page (Quick-start query pills & system overview)           │
│   ├─ 💬 Travel Assistant (Chat UI with context inspector)              │
│   ├─ 🗺️ Explore Destinations (Multi-filter catalog)                    │
│   ├─ 📅 Itinerary Planner (Dynamic daily scheduler)                    │
│   ├─ 💰 Budget Planner (Interactive cost breakdowns & donut charts)    │
│   ├─ 🔍 NLP Analysis (Live Viva pipeline demonstration)                │
│   ├─ 📊 Model Performance (Actual ML metrics & confusion matrix)       │
│   └─ ℹ️ About & Documentation (Viva questions & architecture)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION & DIALOGUE LAYER                     │
│                                                                        │
│   Chatbot Engine (src/chatbot.py)                                      │
│   └─ Coordinates query parsing, dispatching, and response generation   │
│                                                                        │
│   Session Context Manager (src/context_manager.py)                     │
│   └─ Tracks active destination, duration, budget, and resolves pronouns│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          NLP PIPELINE LAYER                            │
│                                                                        │
│   1. Preprocessing (src/preprocessing.py)                              │
│      └─ Contractions, Tokenizer, Stopwords filter, Lemmatizer          │
│                                                                        │
│   2. Entity Extraction (src/entity_extractor.py)                       │
│      └─ spaCy NER + Travel Regex + Levenshtein Fuzzy String Matcher    │
│                                                                        │
│   3. Intent Classification (src/intent_classifier.py)                  │
│      └─ TF-IDF Vectorizer + Logistic Regression Classifier (26 classes)│
│                                                                        │
│   4. Preference Extraction (src/preference_extractor.py)               │
│      └─ Taxonomy matching & idiomatic phrase mapping                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL & REASONING LAYER                       │
│                                                                        │
│   Hybrid Retriever (src/retriever.py)                                  │
│   ├─ Method 1: Structured Filter (Budget, Duration, Season, Mobility)  │
│   ├─ Method 2: Keyword Lexical Overlap                                 │
│   ├─ Method 3: Document TF-IDF Similarity                              │
│   └─ Method 4: Dense Neural Cosine Similarity (src/semantic_search.py) │
│                                                                        │
│   Dynamic Recommender (src/recommender.py)                             │
│   └─ Multi-criteria weighted scoring & explainability generator        │
│                                                                        │
│   Domain Specialists                                                   │
│   ├─ Itinerary Generator (src/itinerary.py)                            │
│   ├─ Budget Planner (src/budget.py)                                    │
│   └─ Comparison, Food, Transit, Packing Advisories (src/utils.py)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           DATA & MODEL ASSETS                          │
│                                                                        │
│   data/processed/                                                      │
│   ├─ destinations.csv  (36 records)                                    │
│   ├─ attractions.csv   (78 records)                                    │
│   ├─ food.csv          (22 records)                                    │
│   ├─ budget.csv        (36 records)                                    │
│   ├─ transport.csv     (14 records)                                    │
│   └─ packing_tips.csv  (7 records)                                     │
│                                                                        │
│   models/                                                              │
│   ├─ intent_classifier.pkl  (Serialized Logistic Regression Model)     │
│   ├─ tfidf_vectorizer.pkl   (Serialized TF-IDF Vectorizer)             │
│   ├─ intent_metrics.json    (Evaluated Test Metrics & Confusion Matrix)│
│   └─ embeddings/                                                       │
│      ├─ destination_embeddings.npy  (36 x 384 Dense Embeddings)        │
│      └─ attraction_embeddings.npy   (78 x 384 Dense Embeddings)        │
└────────────────────────────────────────────────────────────────────────┘
```
