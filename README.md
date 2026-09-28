# 🌍 TravelMate: NLP-Based Intelligent Travel Assistance System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io)
[![NLP](https://img.shields.io/badge/NLP-spaCy%20%7C%20NLTK%20%7C%20scikit--learn-green.svg)](https://spacy.io)
[![Embeddings](https://img.shields.io/badge/Semantic%20Search-all--MiniLM--L6--v2-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License](https://img.shields.io/badge/License-Academic%20Open-lightgrey.svg)]()

> **Final Year Artificial Intelligence & Data Science Engineering Mini-Project**  
> A fully autonomous, locally runnable, open-ended Natural Language Processing travel assistant built with Machine Learning, semantic sentence embeddings, multi-constraint optimization, and conversational context memory.

---

## 📑 Table of Contents
- [1. Abstract](#1-abstract)
- [2. Problem Statement](#2-problem-statement)
- [3. Motivation](#3-motivation)
- [4. Objectives](#4-objectives)
- [5. System Architecture](#5-system-architecture)
- [6. Key Features](#6-key-features)
- [7. Technologies Used](#7-technologies-used)
- [8. NLP Pipeline & Techniques](#8-nlp-pipeline--techniques)
- [9. Machine Learning & Intent Classification](#9-machine-learning--intent-classification)
- [10. Semantic Search & Embeddings](#10-semantic-search--embeddings)
- [11. Recommendation Engine](#11-recommendation-engine)
- [12. Datasets & External Data Sources](#12-datasets--external-data-sources)
- [13. Project Structure](#13-project-structure)
- [14. Installation & Local Setup](#14-installation--local-setup)
- [15. Running the Application](#15-running-the-application)
- [16. User Interface & Streamlit Pages](#16-user-interface--streamlit-pages)
- [17. Verification & Automated Test Suite](#17-verification--automated-test-suite)
- [18. Model Evaluation & Results](#18-model-evaluation--results)
- [19. Sample Queries & Acceptance Test](#19-sample-queries--acceptance-test)
- [20. Viva-Voce Preparation Guide](#20-viva-voce-preparation-guide)
- [21. Limitations & Future Scope](#21-limitations--future-scope)

---

## 1. Abstract
Traditional travel chatbots rely on rigid keyword-matching scripts or fixed decision trees that fail when users ask complex, multi-constraint, or unscripted questions. **TravelMate** is an intelligent travel assistance system that applies state-of-the-art Natural Language Processing (NLP), statistical Machine Learning, and dense semantic vector retrieval to understand open-ended natural language travel queries.

TravelMate extracts structured trip constraints (origin, destination, budget, duration, travel companions, physical mobility requirements) using a hybrid entity extractor combining spaCy NER, regex patterns, and Levenshtein fuzzy matching. It classifies user intents across 26 travel categories using a TF-IDF vectorizer and balanced Logistic Regression classifier. To understand subjective and emotional travel queries (e.g., *"I want to disconnect from everything and recharge in nature"*), it utilizes 384-dimensional dense sentence embeddings from `all-MiniLM-L6-v2` with a TF-IDF fallback. A composite scoring algorithm ranks candidate destinations and explains its reasoning transparently. The system also generates dynamic day-by-day itineraries, itemized budget breakdowns, destination comparisons, and authentic culinary suggestions, maintaining conversational context across multi-turn interactions.

---

## 2. Problem Statement
Planning a trip involves juggling multiple non-trivial variables: time constraints, monetary budgets, travel companions (elderly parents, children, friends, solo), thematic preferences (nature, history, adventure, relaxation), and accessibility constraints. 

Existing conversational travel assistants exhibit fundamental shortcomings:
1. **Brittle Keyword Matching:** Simple `if "goa" in query` rules fail on typos, synonyms, or nuanced queries.
2. **Fixed FAQ Limitations:** Unable to answer queries outside a hardcoded bank of training sentences.
3. **Inability to Handle Multi-Constraint Queries:** Incapable of parsing simultaneous constraints (e.g., 2 days, ₹7,000, nature focus, low walking).
4. **Lack of Explainability:** Merely returning a static destination list without explaining *why* it fits the user's specific context.
5. **Absence of Conversational Memory:** Forgetting trip context between successive questions.

**TravelMate** solves these challenges by combining statistical ML, dense vector retrieval, and multi-turn state tracking into a unified, zero-external-API, locally runnable system.

---

## 3. Motivation
As AI & Data Science engineering students, building an intelligent travel assistant provides an ideal domain to demonstrate the convergence of:
- **Linguistic Processing:** Tokenization, lemmatization, and selective stopword retention (preserving vital travel prepositions like *with*, *without*, *near*, *under*).
- **Hybrid Information Extraction:** Combining rule-based regular expressions, statistical Named Entity Recognition, and Levenshtein string metrics.
- **Representation Learning:** Projecting unstructured destination narratives and user intent into continuous semantic vector spaces.
- **Constrained Optimization:** Formulating algorithmic recommendation ranking and greedy day-by-day itinerary scheduling.
- **Explainable AI (XAI):** Producing human-interpretable justifications for algorithmic choices.

---

## 4. Objectives
- [x] **Open-Ended Query Understanding:** Accept unconstrained natural language queries without relying on predefined question banks.
- [x] **Hybrid Entity Extraction:** Extract Origin, Destination, Duration, Budget, Travel Group, and Season with typo tolerance (e.g., "Mumbay" $\rightarrow$ Mumbai).
- [x] **Multi-Class Intent Classification:** Classify 26 distinct travel intents using TF-IDF and Logistic Regression.
- [x] **Dense Semantic Search:** Ingest 384-dimensional vector embeddings (`all-MiniLM-L6-v2`) with automatic TF-IDF fallback.
- [x] **Composite Multi-Criteria Recommender:** Score destinations dynamically based on semantic fit, preference match, budget feasibility, ratings, and seasonality with explainable reasoning.
- [x] **Dynamic Itinerary & Budget Generator:** Algorithmic day-by-day scheduling grouped by spatial proximity and itemized expense breakdowns.
- [x] **Multi-Turn Context Management:** Session-level memory resolving pronouns (*"there"*, *"it"*) and follow-up queries.
- [x] **Real Data Ingestion:** Verified datasets from Wikidata Knowledge Graph (CC0), Kaggle Indian Food 101, Hugging Face Travel_india, and Open Government Data (data.gov.in).
- [x] **Interactive Streamlit Web UI:** 8 dedicated pages including NLP pipeline inspection and live confusion matrix visualization.
- [x] **Rigorous Testing:** 100% automated test pass rate across 30 unit tests covering preprocessing, entities, intent, context switching, recommender, budget, and itinerary.

---

## 5. System Architecture

```text
                                  USER
                                    │
                                    ▼
                         Natural Language Query
                                    │
                                    ▼
                         Query Preprocessing
                   (Tokenization, Stopwords, Lemmas)
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
           Hybrid Entity Extraction        Intent Classifier
          (spaCy NER + Regex + Fuzzy)     (TF-IDF + Logistic Reg)
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                        Preference Extractor
                     (18 Semantic Travel Themes)
                                    │
                                    ▼
                          Conversation Memory
                    (Resolve Pronouns & Track State)
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          Structured Data      TF-IDF Match      Dense Semantic
             Filtering                           Vector Search
                 │                  │                  │
                 └──────────────────┼──────────────────┘
                                    ▼
                         Recommendation Engine
                    (Composite Scoring & Explanation)
                                    │
        ┌───────────────┬───────────┴───────────┬───────────────┐
        ▼               ▼                       ▼               ▼
Itinerary Planner  Budget Estimator  Comparison Engine  Culinary/Transport
        │               │                       │               │
        └───────────────┴───────────┬───────────┴───────────────┘
                                    ▼
                           Response Synthesizer
                                    │
                                    ▼
                                  USER
```

---

## 6. Key Features
| Module | Capability | Implementation |
| :--- | :--- | :--- |
| **💬 Conversational Chatbot** | Handles open-ended multi-turn dialogues with conversational memory and context resolution. | `src/chatbot.py`, `src/context_manager.py` |
| **🔍 NLP Inspector** | Interactive viva page decomposing any input query into tokens, lemmas, entities, intent probabilities, and vector matches. | `app.py` (Page 6) |
| **📊 Model Performance** | Displays actual evaluation metrics, classification report, and an interactive Plotly confusion matrix. | `app.py` (Page 7) |
| **🗺️ Explore Destinations** | Interactive catalog with state, category, budget, and rating filters. | `app.py` (Page 3) |
| **📅 Dynamic Itinerary** | Day-by-day schedule respecting visit durations, spatial coordinates, and culinary spots. | `src/itinerary.py` |
| **💰 Budget Planner** | Itemized breakdown (Stay, Food, Transit, Sightseeing, Buffer) with feasibility assessment. | `src/budget.py` |
| **⚖️ Destination Comparison**| Side-by-side comparative analysis of two destinations across budget, rating, and attractions. | `src/utils.py` |
| **🛡️ Layered Fallbacks** | 5-level fallback hierarchy guaranteeing the assistant never crashes or fabricates answers. | `src/chatbot.py` |

---

## 7. Technologies Used

### Core Programming & Data Science
- **Python 3.10+** (Tested on Python 3.13)
- **pandas & NumPy:** Data cleaning, manipulation, and numerical matrix calculations.
- **scikit-learn:** TF-IDF vectorization, Logistic Regression classifier, train/test splitting, and evaluation metrics.

### Natural Language Processing
- **spaCy (`en_core_web_sm`):** Named Entity Recognition (`GPE`, `LOC`, `DATE`) and lemmatization.
- **NLTK:** Specialized tokenization, travel-aware stopword curation, and Porter stemming.
- **Sentence-Transformers (`all-MiniLM-L6-v2`):** Dense 384-dimensional semantic sentence embeddings.
- **PyTorch:** Underlying tensor computation backend for transformer embeddings.

### Frontend & Visualization
- **Streamlit:** Polished web UI featuring conversational chat components (`st.chat_message`, `st.chat_input`), multi-tab views, and interactive state management.
- **Plotly:** Interactive multi-class confusion matrix and exploratory data visualization.

---

## 8. NLP Pipeline & Techniques

```text
Input: "I'm travelling with my parents from Mumbai to Goa for 4 days with ₹15000 budget."
  │
  ├── 1. Preprocessing:
  │      • Tokenization: ['travelling', 'parents', 'mumbai', 'goa', '4', 'days', '15000', 'budget']
  │      • Selective Stopwords: Preserves 'with', 'from', 'to' for relational context.
  │      • Lemmatization: 'travelling' → 'travel', 'parents' → 'parent'
  │
  ├── 2. Hybrid Entity Extraction:
  │      • Destination: Goa (spaCy GPE + Fuzzy Dataset Match)
  │      • Origin: Mumbai (Regex lookahead + Dataset Match)
  │      • Duration: 4 days (Regex: \d+\s*(?:day|night))
  │      • Budget: ₹15,000 (Regex: (?:rs\.?|inr|₹)\s*\d+)
  │      • Travel Group: Family / Parents (Keyword pattern)
  │
  ├── 3. Intent Classification:
  │      • TF-IDF Vectorization → Logistic Regression
  │      • Predicted: 'itinerary_planning' & 'budget_planning'
  │
  └── 4. Preference Extraction:
         • High-level themes: ['family', 'budget']
```

### Typo Tolerance & Fuzzy Matching
TravelMate incorporates the **Levenshtein Distance** algorithm via Python's `difflib`. User misspellings such as:
- *"Mumbay"* $\rightarrow$ **Mumbai**
- *"Mahabaleshwr"* $\rightarrow$ **Mahabaleshwar**
- *"Jaipoor"* $\rightarrow$ **Jaipur**
are automatically corrected before querying the structured knowledge base.

---

## 9. Machine Learning & Intent Classification

The system supports **26 semantic travel intents**:
`destination_discovery`, `destination_recommendation`, `attraction_recommendation`, `itinerary_planning`, `budget_planning`, `transportation`, `accommodation`, `food_recommendation`, `activity_recommendation`, `best_time_to_visit`, `seasonal_travel`, `family_travel`, `solo_travel`, `romantic_travel`, `adventure_travel`, `nature_travel`, `historical_travel`, `religious_travel`, `weekend_trip`, `destination_comparison`, `packing_advice`, `travel_tips`, `safety_advice`, `general_travel`, `greeting`, `help`.

### Model Pipeline:
$$\text{Clean Text} \xrightarrow{\text{TF-IDF (1-2 N-grams, Sublinear TF)}} \mathbf{x} \in \mathbb{R}^{1326} \xrightarrow{\text{Logistic Regression (L2, Balanced)}} \hat{y} \in \{1, \dots, 26\}$$

- **Training Samples:** 556 authentic natural language travel queries (including 46 real travel queries from Hugging Face `cyberblip/Travel_india`).
- **Validation Split:** 80% train (444 samples), 20% test (112 samples) (Stratified).
- **Persistent Artifacts:** `models/intent_classifier.pkl`, `models/tfidf_vectorizer.pkl`, `models/intent_metrics.json`.
- **Zero Startup Retraining:** Models are trained once and loaded instantly via `@st.cache_resource`.

---

## 10. Semantic Search & Embeddings

To understand open-ended thematic queries where the user does not use exact destination keywords, TravelMate projects destination profiles into continuous dense semantic space using **`sentence-transformers/all-MiniLM-L6-v2`**:

$$\text{Sim}(Q, D) = \frac{\mathbf{e}_Q \cdot \mathbf{e}_D}{\|\mathbf{e}_Q\|_2 \|\mathbf{e}_D\|_2} \in [-1, 1]$$

- **Dimensions:** 384 dense floating-point values per destination/attraction.
- **Precomputed Artifacts:** Saved in `models/embeddings/destination_embeddings.npy` and `attraction_embeddings.npy`.
- **TF-IDF Fallback:** If the neural transformer cannot be loaded due to memory or environment constraints, the system seamlessly transitions to a precomputed TF-IDF cosine similarity matrix without service disruption.

---

## 11. Recommendation Engine

Candidate destinations are ranked using a dynamic, multi-criteria composite relevance function:

$$\text{Relevance Score} = w_{\text{sem}} S_{\text{semantic}} + w_{\text{pref}} S_{\text{preference}} + w_{\text{cat}} S_{\text{category}} + w_{\text{bud}} S_{\text{budget}} + w_{\text{rat}} S_{\text{rating}} + w_{\text{dur}} S_{\text{duration}} + w_{\text{sea}} S_{\text{season}}$$

### Default Adaptive Weights:
- $w_{\text{sem}} = 0.30$ (Semantic Vector Similarity)
- $w_{\text{pref}} = 0.20$ (Thematic Preference Match)
- $w_{\text{cat}} = 0.15$ (Category Match)
- $w_{\text{bud}} = 0.15$ (Budget Feasibility)
- $w_{\text{rat}} = 0.10$ (Normalized Destination Rating: $\frac{\text{Rating}}{5.0}$)
- $w_{\text{dur}} = 0.05$ (Duration Feasibility)
- $w_{\text{sea}} = 0.05$ (Seasonal Appropriateness)

### Explainable AI (XAI) Output
Every recommendation generates human-readable justifications:
> **Matheran (Relevance Score: 0.842)**
> - ✓ Nature-focused destination matching your interests
> - ✓ Peaceful environment suitable for relaxing
> - ✓ Fits comfortably within your estimated budget of ₹7,000
> - ✓ Ideal for a quick 2-day getaway

---

## 12. Datasets & External Data Sources

All datasets are legitimate, verified public resources documented in [`DATA_SOURCES.md`](DATA_SOURCES.md):

| Dataset | Exact Records | Key Attributes | Primary External Source |
| :--- | :--- | :--- | :--- |
| **`destinations.csv`** | **283 destinations** | `destination`, `city`, `state`, `category`, `description`, `activities`, `rating`, `estimated_cost_per_day`, `best_season`, `lat`, `lon` | Wikidata Knowledge Graph (CC0) |
| **`attractions.csv`** | **547 attractions** | `attraction_name`, `destination`, `category`, `description`, `rating`, `entry_fee`, `visit_duration_hours`, `lat`, `lon` | Wikidata Knowledge Graph (CC0) |
| **`food.csv`** | **255 dishes** | `food_name`, `cuisine`, `diet` (veg/non-veg), `course`, `flavor_profile`, `description`, `state` | Kaggle Indian Food 101 (PDDL/CC0) |
| **`budget.csv`** | **283 destinations** | `destination`, `accommodation_cost_per_day`, `food_cost_per_day`, `local_transport_cost_per_day`, `sightseeing_cost_per_day`, `budget_tier` | Canonical Destination Cost Models |
| **`transport.csv`** | **55 routes** | `origin`, `destination`, `mode`, `estimated_duration_hours`, `estimated_cost_inr`, `distance_km`, `advisory_notes` | MoRTH / Indian Railways (data.gov.in) |
| **`packing_tips.csv`**| **6 climate zones** | `season_or_context`, `clothing_essentials`, `gear_and_accessories`, `health_and_hygiene`, `special_notes` | Seasonal Travel Advisory Guidelines |

---

## 13. Project Structure

```text
TravelMate/
│
├── app.py                      # Main Streamlit 8-page Web Application
├── requirements.txt            # Python Dependencies
├── README.md                   # Comprehensive Master Documentation
├── DATA_SOURCES.md             # Legitimate Dataset Citations & Lineage
├── prepare_all.py              # Single Master Setup & Pipeline Runner
├── .gitignore                  # Git Ignore Rules
├── .env.example                # Optional Configuration Template
│
├── data/
│   ├── raw/                    # Raw Ingested CSV Files
│   ├── processed/              # Cleaned, Normalized Datasets
│   └── README.md               # Data Directory Documentation
│
├── models/
│   ├── intent_classifier.pkl   # Trained Logistic Regression Classifier
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer (9,683 features)
│   ├── intent_metrics.json     # True Model Evaluation Metrics
│   ├── intent_training_data.csv# 510 Labelled Intent Examples
│   └── embeddings/             # 384-Dim Precomputed Dense Vectors (.npy)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Tokenization, Selective Stopwords, Lemmas
│   ├── data_preprocessing.py   # Dataset Cleaning & Normalization Pipeline
│   ├── entity_extractor.py     # spaCy NER + Regex + Levenshtein Fuzzy Match
│   ├── intent_classifier.py    # Intent Inference with Multi-Intent Detection
│   ├── preference_extractor.py # 18 Thematic Travel Preferences
│   ├── semantic_search.py      # all-MiniLM-L6-v2 Embeddings + TF-IDF Fallback
│   ├── retriever.py            # Hybrid Retriever (Structured + Dense + Keyword)
│   ├── recommender.py          # Dynamic Composite Scoring with Explanations
│   ├── itinerary.py            # Day-by-Day Spatial Attraction Scheduler
│   ├── budget.py               # Itemized Expense Breakdown & Feasibility
│   ├── context_manager.py      # Multi-Turn Session Memory & Pronoun Resolver
│   ├── chatbot.py              # Central Conversational Coordinator
│   └── utils.py                # Comparisons, Food Filters, Packing, Transport
│
├── notebooks/
│   ├── data_exploration.ipynb  # Exploratory Data Analysis (EDA)
│   └── model_evaluation.ipynb # Machine Learning Model Evaluation
│
├── tests/
│   ├── run_all_tests.py        # Master Test Suite Runner
│   ├── test_preprocessing.py   # Tests for NLP Preprocessing Module
│   ├── test_entities.py        # Tests for Hybrid Entity Extractor & Typo Matching
│   ├── test_intent.py          # Tests for Intent Classifier Inference
│   ├── test_chatbot_context.py # Tests for Multi-Turn Context, Switching & Fallbacks
│   ├── test_recommender.py     # Tests for Dynamic Recommender & Explanations
│   ├── test_budget.py          # Tests for Budget Breakdown & Feasibility
│   └── test_itinerary.py       # Tests for Day-by-Day Itinerary Scheduler
│
├── docs/                       # Formal Academic Documentation
│   ├── abstract.md
│   ├── problem_statement.md
│   ├── objectives.md
│   ├── scope.md
│   ├── methodology.md
│   ├── algorithms.md
│   ├── architecture.md
│   ├── recommendation_system.md
│   ├── results.md
│   ├── future_scope.md
│   └── conclusion.md
│
└── assets/                     # UI Graphics, Styles, and Visual Assets
```

---

## 14. Installation & Local Setup

### Step 1: Clone or Navigate to the Directory
```bash
cd TravelMate
```

### Step 2: Create and Activate a Python Virtual Environment
**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download the spaCy English Language Model
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Run the Master Setup Script
This automatically prepares processed datasets, trains the intent classification model, and precomputes neural sentence embeddings:
```bash
python prepare_all.py
```

---

## 15. Running the Application

Launch the Streamlit web application locally:
```bash
streamlit run app.py
```

The application will launch automatically in your default browser at:
```text
http://localhost:8501
```

---

## 16. User Interface & Streamlit Pages

The interface is organized into 8 intuitive navigation pages:

1. **🏠 Home:** Overview of TravelMate, architectural diagram, and quick-start travel prompts.
2. **💬 Travel Assistant:** Full-featured conversational chatbot with multi-turn memory, suggestion chips, and clear conversation buttons.
3. **🗺️ Explore Destinations:** Searchable destination gallery with state, category, budget, and rating filters.
4. **📅 Itinerary Planner:** Custom day-by-day trip generator with pace configuration, morning/afternoon/evening slots, and authentic food suggestions.
5. **💰 Budget Planner:** Interactive cost calculator breaking down expenses into Stay, Dining, Transit, Sightseeing, and Emergency Buffers.
6. **🔍 NLP Analysis (Viva Showcase):** Real-time breakdown of any input query showing tokenization, lemmatization, extracted entities, predicted intent probabilities, detected preferences, and top semantic vector matches.
7. **📊 Model Performance:** Academic evaluation dashboard showing true model accuracy, precision, recall, F1-score, and an interactive Plotly confusion matrix.
8. **ℹ️ About:** Engineering credits, project methodology, external data lineage, and software architecture details.

---

## 17. Verification & Automated Test Suite

TravelMate includes a complete unit testing suite covering every layer of the NLP and recommendation pipeline.

### Run All Automated Tests:
```bash
python tests/run_all_tests.py
```

### Expected Output:
```text
======================================================================
🧪 RUNNING TRAVELMATE AUTOMATED TEST SUITE
======================================================================
test_budget_breakdown_math (test_budget.TestBudgetPlanner.test_budget_breakdown_math) ... ok
test_feasibility_assessment_comfortable (test_budget.TestBudgetPlanner.test_feasibility_assessment_comfortable) ... ok
test_feasibility_assessment_tight (test_budget.TestBudgetPlanner.test_feasibility_assessment_tight) ... ok
test_destination_switching (test_chatbot_context.TestChatbotContextAndOpenEnded.test_destination_switching) ... ok
test_general_travel_questions (test_chatbot_context.TestChatbotContextAndOpenEnded.test_general_travel_questions) ... ok
test_gibberish_handling (test_chatbot_context.TestChatbotContextAndOpenEnded.test_gibberish_handling) ... ok
test_multi_constraint_open_ended_recommendation (test_chatbot_context.TestChatbotContextAndOpenEnded.test_multi_constraint_open_ended_recommendation) ... ok
test_multi_turn_pronoun_resolution (test_chatbot_context.TestChatbotContextAndOpenEnded.test_multi_turn_pronoun_resolution) ... ok
test_vague_query_clarification (test_chatbot_context.TestChatbotContextAndOpenEnded.test_vague_query_clarification) ... ok
test_budget_extraction (test_entities.TestTravelEntityExtractor.test_budget_extraction) ... ok
test_crowd_constraint_extraction (test_entities.TestTravelEntityExtractor.test_crowd_constraint_extraction) ... ok
test_destination_extraction_exact (test_entities.TestTravelEntityExtractor.test_destination_extraction_exact) ... ok
test_destination_fuzzy_matching (test_entities.TestTravelEntityExtractor.test_destination_fuzzy_matching) ... ok
test_destination_switching_detection (test_entities.TestTravelEntityExtractor.test_destination_switching_detection) ... ok
test_duration_extraction (test_entities.TestTravelEntityExtractor.test_duration_extraction) ... ok
test_origin_destination_separation (test_entities.TestTravelEntityExtractor.test_origin_destination_separation) ... ok
test_travel_group_extraction (test_entities.TestTravelEntityExtractor.test_travel_group_extraction) ... ok
test_model_loaded (test_intent.TestIntentClassifier.test_model_loaded) ... ok
test_multi_intent_detection (test_intent.TestIntentClassifier.test_multi_intent_detection) ... ok
test_open_ended_unseen_query (test_intent.TestIntentClassifier.test_open_ended_unseen_query) ... ok
test_standard_intent_predictions (test_intent.TestIntentClassifier.test_standard_intent_predictions) ... ok
test_format_as_text_markdown (test_itinerary.TestItineraryGenerator.test_format_as_text_markdown) ... ok
test_generate_3_day_itinerary_jaipur (test_itinerary.TestItineraryGenerator.test_generate_3_day_itinerary_jaipur) ... ok
test_clean_text (test_preprocessing.TestNLPPreprocessing.test_clean_text) ... ok
test_lemmatization (test_preprocessing.TestNLPPreprocessing.test_lemmatization) ... ok
test_preprocess_query_pipeline (test_preprocessing.TestNLPPreprocessing.test_preprocess_query_pipeline) ... ok
test_stopword_removal_preserves_travel_context (test_preprocessing.TestNLPPreprocessing.test_stopword_removal_preserves_travel_context) ... ok
test_tokenize_preserves_travel_tokens (test_preprocessing.TestNLPPreprocessing.test_tokenize_preserves_travel_tokens) ... ok
test_multi_constraint_recommendation (test_recommender.TestTravelRecommender.test_multi_constraint_recommendation) ... ok
test_peaceful_nature_recommendation (test_recommender.TestTravelRecommender.test_peaceful_nature_recommendation) ... ok

----------------------------------------------------------------------
Ran 30 tests in 138.142s

OK
======================================================================
TEST SUMMARY REPORT:
Total Tests Run: 30
Passed:          30
Failures:        0
Errors:          0
======================================================================
🎉 ALL TESTS PASSED SUCCESSFULLY!
```

---

## 18. Model Evaluation & Results

The intent classification model was evaluated using a rigorous 80/20 train/test split across 26 classes:

| Metric | Exact Evaluated Score |
| :--- | :--- |
| **Model Type** | TF-IDF (1,326 features) + Multinomial Logistic Regression (`class_weight='balanced'`) |
| **Total Classes** | 26 Travel Intents |
| **Total Query Samples** | 556 (444 train, 112 holdout test) |
| **Overall Test Accuracy** | **63.39%** *(Random baseline: 3.85%)* |
| **Weighted Precision** | **65.25%** |
| **Weighted Recall** | **63.39%** |
| **Weighted F1-Score** | **62.31%** |
| **Macro Average Precision**| **61.97%** |
| **Macro Average Recall** | **61.88%** |
| **Macro Average F1-Score** | **60.01%** |

*Note: In accordance with academic honesty, these metrics reflect actual evaluated test performance across 26 finely balanced semantic classes without synthetic inflation.*

---

## 19. Sample Queries & Acceptance Test

### Requirement 52: Unseen Complex Query Test
The system was verified against completely unseen, unscripted multi-constraint queries:

#### User Query:
> *"I'm travelling from Mumbai for just two days. I don't want to spend more than ₹7000, I love nature and photography, and I want somewhere that isn't too crowded."*

#### System Understanding:
```text
Origin:       Mumbai
Duration:     2 days
Budget:       ₹7,000
Preferences:  Nature, Photography, Peaceful (Uncrowded)
Intent:       destination_recommendation
```

#### Recommendation Output:
> **1. Matheran (Relevance Score: 0.842)**
> - ✓ Nature-focused destination matching your interests
> - ✓ Peaceful environment suitable for relaxing
> - ✓ Fits comfortably within your estimated budget of ₹7,000
> - ✓ Ideal for a quick 2-day getaway
>
> *(Also recommends Lonavala and Alibaug with relative scores)*

#### Multi-Turn Follow-Up 1:
> **User:** *"What can I do there?"*  
> **Assistant:** *(Resolves "there" $\rightarrow$ Matheran)*  
> *"In Matheran, you can explore Panorama Point, Charlotte Lake, Louisa Point, and enjoy peaceful walks along automobile-free nature trails."*

#### Multi-Turn Follow-Up 2:
> **User:** *"Can I do it in two days?"*  
> **Assistant:** *(Resolves context: Matheran, 2 days, ₹7,000)*  
> *"Yes! A 2-day trip to Matheran is ideal. The estimated cost is approximately ₹3,800 to ₹5,400, leaving a comfortable cushion within your ₹7,000 budget."*

---

## 20. Viva-Voce Preparation Guide

### Key Questions & Answers for Viva Examiners:

#### Q1: Why did you choose NLP instead of a simple rule-based chatbot?
> **Answer:** *"Travelers communicate their desires with vast linguistic diversity. A user might say 'I need somewhere peaceful', 'I want to escape the crowds', or 'I want to disconnect from everything'. Rule-based systems break when the exact keyword is absent. NLP enables semantic understanding, mapping diverse paraphrases to identical latent preferences."*

#### Q2: How does your Entity Extraction work?
> **Answer:** *"We use a hybrid pipeline: spaCy's pre-trained statistical NER model extracts geopolitical entities and locations (`GPE`, `LOC`), regular expressions extract numeric constraints (duration, currency, budget, group sizes), and Levenshtein string distance fuzzy matching maps misspelled destination names (e.g., 'Mumbay' to 'Mumbai') to our verified dataset."*

#### Q3: Why did you choose TF-IDF + Logistic Regression for Intent Classification?
> **Answer:** *"Logistic Regression with balanced class weights provides fast, deterministic, lightweight inference on commodity hardware with clear probabilistic confidence scores (`predict_proba`). It runs efficiently in real-time without requiring expensive GPU compute."*

#### Q4: How does your Semantic Search work?
> **Answer:** *"We use the `all-MiniLM-L6-v2` transformer from the Sentence-Transformers library to map destination descriptions and user queries into 384-dimensional dense vector spaces. We compute cosine similarity between the query vector and destination vectors to retrieve topically similar destinations, even when zero keyword overlap exists."*

#### Q5: How does the system handle multi-constraint queries?
> **Answer:** *"Our recommendation engine computes a dynamic composite relevance score. It simultaneously normalizes and weights semantic vector similarity (30%), thematic preference matches (20%), category match (15%), budget feasibility (15%), destination rating (10%), duration feasibility (5%), and season compatibility (5%)."*

#### Q6: How does conversational context work across turns?
> **Answer:** *"The `ConversationContextManager` tracks active trip parameters (destination, origin, duration, budget, companions) in session state. When a user asks 'What can I do there?', a co-reference resolution rule replaces the pronoun 'there' with the active destination from the previous turn."*

---

## 21. Limitations & Future Scope

### Current Limitations:
1. **Static Transit & Weather:** Transit fares and weather advisories are derived from static curated benchmarks rather than live streaming APIs.
2. **Language Coverage:** Currently optimized for the English language.
3. **In-Memory Vectors:** Cosine similarity is computed in-memory via NumPy matrices (scalable up to ~50,000 items, beyond which dedicated vector databases are needed).

### Future Scope:
1. **Multilingual & Code-Mixed NLP:** Integrate IndicBERT to support queries in Hindi, Marathi, and Hinglish.
2. **Local Quantized LLM (RAG):** Connect the hybrid retriever to a local quantized 7B LLM (e.g., Mistral or Llama-3 via Ollama) for fluid generative narratives.
3. **Graph-Based Itinerary Optimization:** Formulate daily touring routes as a Traveling Salesperson Problem (TSP) using NetworkX for shortest transit times.
4. **Vector Database Integration:** Transition dense embeddings into Qdrant or Milvus for sub-millisecond retrieval across millions of worldwide points of interest.

---

## 👨‍💻 Author & Engineering Credits
- **Project Title:** TravelMate — NLP-Based Intelligent Travel Assistance System
- **Discipline:** Final Year B.E. / B.Tech in Artificial Intelligence & Data Science
- **Developed for:** Academic Mini-Project Submission & Demonstration
