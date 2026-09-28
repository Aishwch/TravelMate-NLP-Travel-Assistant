# System Methodology — TravelMate

## 1. High-Level Architectural Flow

TravelMate executes a systematic multi-tier pipeline upon receiving every user query:

```text
User Query
   │
   ▼
[1] Preprocessing & Cleaning (Whitespace, Contractions, Tokenization)
   │
   ▼
[2] Hybrid Travel Entity Extraction (spaCy NER + Regex + Levenshtein Matching)
   │
   ▼
[3] Conversational Context Resolution (Pronoun Resolution & Session State)
   │
   ▼
[4] Machine Learning Intent Classification (TF-IDF + Logistic Regression)
   │
   ▼
[5] Semantic Travel Preference Extraction (Taxonomy & Idiomatic Matching)
   │
   ▼
[6] Hybrid Retrieval Engine
   ├─ Structured Filtering (State, Budget, Duration, Mobility, Crowd, Season)
   ├─ Lexical Keyword Matching
   ├─ Document TF-IDF Similarity
   └─ Neural Semantic Search (SentenceTransformer all-MiniLM-L6-v2)
   │
   ▼
[7] Dynamic Recommendation Engine (Multi-Factor Scoring & Explainability)
   │
   ▼
[8] Domain Specialists Dispatch (Itinerary Generator / Budget Planner / Utils)
   │
   ▼
[9] Natural Response Generation & Context Update
   │
   ▼
Delivered to User via Streamlit UI
```

---

## 2. Stage-by-Stage Methodology

### Stage 1: Preprocessing & Normalization
* **Contraction Expansion**: Replaces colloquial contractions (`can't` $\rightarrow$ `can not`, `it's` $\rightarrow$ `it is`).
* **Travel-Aware Tokenizer**: Employs regular expressions that preserve currency symbols (`₹`, `$`, `€`), numerical digits, durations (`3-day`, `2d`), and hyphenated words while stripping non-essential punctuation.
* **Selective Stopword Removal**: Unlike naive stopword removal that eliminates words like *"not"*, *"with"*, and *"near"*, TravelMate retains 20+ travel-critical qualifiers and constraint words to preserve negation and companionship relationships.
* **Morphological Lemmatization**: Uses spaCy lemmatization (`en_core_web_sm`) to map inflected forms back to their base vocabulary lemma (`beaches` $\rightarrow$ `beach`, `visited` $\rightarrow$ `visit`).

### Stage 2: Hybrid Entity Extraction
To address the shortcomings of generic NER models on Indian travel contexts:
1. **Rule-Based Regex Parsing**: Parses budgets across multiple syntaxes (`₹8000`, `Rs. 12,000`, `under 7k`, `10000 rupees`) and trip durations (`weekend` $\rightarrow$ 2 days, `3 nights` $\rightarrow$ 3 days).
2. **Levenshtein Fuzzy String Distance**: Computes close string matches between query tokens and the 36 known destination names using `difflib.get_close_matches` with a threshold cutoff of 0.75–0.80. This seamlessly corrects typos (*"Mumbay"* $\rightarrow$ *"Mumbai"*, *"Mahabaleshwr"* $\rightarrow$ *"Mahabaleshwar"*, *"Gokarn"* $\rightarrow$ *"Gokarna"*).
3. **Implicit Constraint Detection**: Scans for accessibility indicators (*"parents"*, *"can't walk"* $\rightarrow$ `walking_preference = Low`) and serenity indicators (*"away from crowds"*, *"peaceful"* $\rightarrow$ `crowd_preference = Low`).

### Stage 3: Conversational Context Resolution
Maintains an in-memory session object tracking:
* `active_destination`: Current destination under discussion.
* `duration_days`, `budget_inr`, `travel_group`, `preferences`: Persisted across turns.
When a follow-up query containing pronouns (*"What can I do there?"*, *"Can I do it in two days?"*) arrives, the system resolves *"there"* to the active destination and inherits known constraints.

### Stage 4: Machine Learning Intent Classification
* The user query is cleaned and vectorized using a TF-IDF Vectorizer with unigrams and bigrams (`ngram_range=(1, 2)`), sublinear term frequency scaling, and a maximum of 3,500 features.
* A Logistic Regression classifier trained with balanced class weights assigns class probability distributions across 26 discrete travel intents.
* Detects secondary intents when high competitive probabilities exist (multi-intent queries).

### Stage 5: Semantic Embedding Search
* The preprocessed query is embedded into a 384-dimensional vector space via `SentenceTransformer("all-MiniLM-L6-v2")`.
* Pairwise cosine similarity is computed against precomputed, serialized embedding matrices of destination descriptions (`models/embeddings/destination_embeddings.npy`).
* This enables conceptual matching: a query for *"I need a relaxing getaway away from crowds"* retrieves *"Matheran"* and *"Gokarna"* even if the exact words do not appear in the description.

### Stage 6: Dynamic Multi-Factor Recommendation
Candidates retrieved from the hybrid pool are evaluated against an 8-factor dynamic relevance function:
$$\text{Relevance Score} = \frac{\sum_{i=1}^8 w_i \cdot S_i}{\sum_{i=1}^8 w_i}$$
Where weights adapt dynamically based on user inputs (e.g., if a budget is specified, $w_{\text{budget}}$ increases from 0.05 to 0.15; if travelling with elderly parents, $w_{\text{accessibility}}$ increases to 0.15).

### Stage 7: Response Synthesis & Layered Fallback
* Responses are synthesized dynamically into markdown tables, bullet points, and explainable rationales.
* If a query falls outside the knowledge base, TravelMate applies a 5-level layered fallback without fabricating information:
  * Level 1: Structured dataset match.
  * Level 2: Semantic embedding search.
  * Level 3: Lexical keyword match.
  * Level 4: General travel tips advisory.
  * Level 5: Honest clarification question.
