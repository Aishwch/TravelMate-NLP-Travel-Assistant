# Project Objectives — TravelMate

## Primary Objective
To design, implement, evaluate, and demonstrate a fully functioning, locally runnable, open-ended **NLP-Based Intelligent Travel Assistance System (TravelMate)** that dynamically parses natural language travel queries, performs hybrid information retrieval, and generates personalized recommendations, budgets, itineraries, and travel guidance.

---

## Specific Engineering Objectives

1. **Curate and Preprocess Authentic Tourism Knowledge Bases**:
   * Gather verified tourism datasets from Kaggle and Open Government Data (OGD) India covering 36 destinations, 78 attractions, 22 cuisines, 36 budget matrices, and 14 major interstate transit corridors.
   * Build an automated preprocessing script (`src/data_preprocessing.py`) to standardize categories, validate coordinates and numeric costs, and handle missing values.

2. **Implement a Travel-Aware NLP Preprocessing Pipeline**:
   * Develop custom tokenizers in `src/preprocessing.py` that preserve currency symbols (`₹`, `rs`, `$`), durations (`3-day`, `2d`), and numbers.
   * Implement selective stopword filtering that retains crucial travel prepositions and constraint words (`with`, `without`, `near`, `not`, `under`).
   * Implement lemmatization and stemming modules.

3. **Construct a Hybrid Named Entity Extractor**:
   * Combine spaCy NER (`en_core_web_sm`) with specialized regular expressions and Levenshtein fuzzy string distance matching in `src/entity_extractor.py`.
   * Extract destinations, cities, states, origins, trip durations (in integer days), budgets (in INR), group types, people counts, seasons, and physical mobility/crowd constraints.

4. **Train and Serialize an ML Intent Classifier**:
   * Create a balanced, diversified supplementary training corpus of 500+ natural language travel queries spanning 26 semantic travel intents in `src/train_intent_model.py`.
   * Vectorize text using sublinear TF-IDF and train a balanced Logistic Regression classifier.
   * Evaluate with formal metrics (Accuracy, Weighted & Macro Precision, Recall, F1-Score, Confusion Matrix) and serialize models without requiring retraining upon Streamlit launch.

5. **Deploy Neural Semantic Search and Embedding Caching**:
   * Integrate Sentence Transformers (`all-MiniLM-L6-v2`) in `src/semantic_search.py` to encode destination and attraction descriptions into 384-dimensional dense vectors.
   * Persist precomputed vector matrices in `models/embeddings/` to achieve sub-second query retrieval.
   * Implement a reliable, high-fidelity TF-IDF + Cosine Similarity fallback engine.

6. **Develop a Multi-Factor Dynamic Recommendation Engine**:
   * Engineer `src/recommender.py` to rank destinations using an 8-factor dynamic scoring function that adapts its weights according to active user constraints (budget, walking, duration).
   * Generate transparent, explainable natural language rationales (*"Why this was recommended"*).

7. **Build Dynamic Itinerary and Budget Generators**:
   * Implement `src/itinerary.py` to organize multi-day schedules based on attraction ratings, visit duration, and spatial coordinates.
   * Implement `src/budget.py` to compute itemized cost breakdowns (Stay, Food, Transit, Activities, Emergency buffer) and analyze budget feasibility.

8. **Ensure Session-Level Conversational Memory**:
   * Build `src/context_manager.py` to resolve pronouns (*"there"*, *"it"*) and maintain context across multi-turn dialogues.

9. **Deploy an Interactive Streamlit Dashboard with Viva Diagnostics**:
   * Create `app.py` with multi-page navigation including an interactive **NLP Analysis Page** showing live pipeline tokens, entities, intent probabilities, and cosine similarity progress bars for college viva demonstration.
