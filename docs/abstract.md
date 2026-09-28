# Project Abstract — TravelMate

## NLP-Based Intelligent Travel Assistance System

### Academic Metadata
* **Project Title**: TravelMate — NLP-Based Intelligent Travel Assistance System
* **Domain**: Natural Language Processing (NLP), Machine Learning, Semantic Information Retrieval
* **Target Audience**: Final Year B.Tech / B.E. Artificial Intelligence & Data Science Engineering

---

### Executive Summary

Contemporary travel booking and discovery platforms predominantly rely on structured drop-down menus, rigid filter facets, and keyword-matching search bars. While functional for known queries, these systems fail to accommodate natural, multi-constraint queries characteristic of human communication (e.g., *"I have 3 days, ₹8000, I'm travelling with my elderly parents who cannot walk much, and I want somewhere peaceful away from crowds"*). Furthermore, standard travel chatbots often operate as hardcoded decision trees or FAQ question-matchers, rendering them incapable of generalizing to previously unseen natural language queries.

To bridge this gap, this project develops **TravelMate**, an end-to-end, locally runnable, intelligent travel assistance system powered by modern NLP and semantic retrieval techniques. Built around authentic open-access datasets from Kaggle and Open Government Data (OGD) India, TravelMate implements a layered modular pipeline:

1. **Travel-Aware Preprocessing**: Preserves crucial numeric and alphanumeric travel entities (currencies, durations, hyphens) while applying contractions expansion, case normalization, and selective stopword removal.
2. **Hybrid Entity Extraction**: Synthesizes spaCy Named Entity Recognition (NER), regular expressions, and Levenshtein fuzzy string distance matching to recognize destinations, origins, durations, budgets, travel groups, seasons, and accessibility constraints, even in the presence of common typos (*e.g., "Mumbay" $\rightarrow$ "Mumbai"*).
3. **Machine Learning Intent Detection**: Employs a sublinear Term Frequency-Inverse Document Frequency (TF-IDF) feature extractor paired with a balanced-class Logistic Regression classifier to map open-ended user queries into 26 semantic travel intent categories with probabilistic confidence scoring.
4. **Dense Neural Semantic Retrieval**: Projects textual travel descriptions into a 384-dimensional vector space using Sentence Transformers (`all-MiniLM-L6-v2`) and computes pairwise cosine similarity against cached embedding matrices.
5. **Dynamic Recommendation Engine**: Implements an 8-factor dynamic composite relevance score (semantic similarity, preference overlap, category alignment, budget compatibility, traveler rating, duration fit, seasonal alignment, and terrain accessibility) that generates explainable natural language rationales (*"Why this was recommended"*).
6. **Multi-Turn Contextual Dialogue Memory**: Maintains session context across conversational turns to resolve pronouns (*"What can I do there?"* $\rightarrow$ references the previously recommended destination) and track evolving user constraints.

The system is deployed via an interactive Streamlit application featuring live NLP pipeline diagnostics (Tokenization, Lemmatization, Entity Extraction, Intent Probability Bar Charts, and Semantic Cosine Distance Bars) tailored for academic defense and college project viva examinations. Extensive automated unit testing confirms robust performance across multi-constraint, multi-intent, and open-ended queries.
