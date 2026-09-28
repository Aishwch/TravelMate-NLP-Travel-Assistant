# Detailed Algorithmic Documentation — TravelMate

This document details the core algorithms, mathematical foundations, and NLP operations utilized in **TravelMate**.

---

## 1. Tokenization

* **Purpose**: Deconstruct continuous unstructured text strings into discrete, meaningful linguistic units (tokens) suitable for downstream feature extraction.
* **Input**: Raw query string $S$ (e.g., *"I have ₹8000 and 3 days for Goa."*).
* **Process**:
  1. Expand informal English contractions using regex mapping.
  2. Apply regex pattern `r"[₹$€]?\d+(?:,\d+)*(?:\.\d+)?k?|\b\w+(?:-\w+)*\b"` to match alphanumeric tokens while explicitly binding currency symbols and numerical values.
  3. Filter out isolated non-semantic punctuation characters.
* **Output**: List of discrete string tokens: `['I', 'have', '₹8000', 'and', '3', 'days', 'for', 'Goa']`.
* **Why Used**: Standard whitespace splitters separate currency symbols (`₹` and `8000`), breaking financial entities. Our travel-aware tokenizer preserves numbers, prices, and hyphenated words intact.

---

## 2. Selective Stop Word Removal

* **Purpose**: Filter out high-frequency functional words that carry minimal topical information while retaining relational prepositions essential for travel constraint parsing.
* **Input**: List of tokens from the tokenizer.
* **Process**:
  1. Define a standard English stopword set $\mathcal{W}_{\text{stop}}$ (179 words).
  2. Subtract 20+ travel-critical qualifiers:
     $$\mathcal{W}_{\text{travel\_stop}} = \mathcal{W}_{\text{stop}} \setminus \{\text{'not'}, \text{'no'}, \text{'with'}, \text{'without'}, \text{'near'}, \text{'from'}, \text{'to'}, \text{'under'}, \text{'between'}, \text{'more'}, \text{'less'}\}$$
  3. Filter tokens where $t \in \mathcal{W}_{\text{travel\_stop}}$, unless $t$ contains currency signs or digits.
* **Output**: Filtered token sequence preserving negations and travel relationships.
* **Why Used**: Naive stopword removal eliminates *"without"* and *"not"*, converting *"without crowds"* into *"crowds"*, completely reversing user intent. Our selective filtering preserves semantic polarity.

---

## 3. Morphological Lemmatization

* **Purpose**: Reduce inflectional and variant forms of words to their base dictionary lemma using part-of-speech context.
* **Input**: List of filtered tokens.
* **Process**:
  1. Pass the token sequence through spaCy's morphological analysis and lemmatization pipeline (`en_core_web_sm`).
  2. Retrieve the canonical root:
     $$\text{Lemma}(t) = \begin{cases} \text{token.lemma\_} & \text{if spaCy active} \\ \text{RuleLookup}(t) & \text{fallback} \end{cases}$$
  3. Transform plural nouns (*"beaches"*, *"forts"*) into singular forms (*"beach"*, *"fort"*), and past tense verbs (*"visited"*) into infinitives (*"visit"*).
* **Output**: Normalized sequence of base lemmas.
* **Why Used**: Lemmatization standardizes vocabulary dimensions across user queries and dataset records without losing semantic meaning (unlike crude stemming which truncates word endings).

---

## 4. Term Frequency-Inverse Document Frequency (TF-IDF)

* **Purpose**: Quantify the statistical significance of unigram and bigram n-grams within a query relative to a broader corpus.
* **Input**: Normalized text string of query lemmas.
* **Mathematical Formulation**:
  $$\text{TF}(t, d) = 1 + \log(f_{t, d}) \quad \text{(sublinear scaling)}$$
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
  Vectors are $L_2$-normalized:
  $$\vec{v} = \frac{\vec{v}}{\|\vec{v}\|_2}$$
* **Output**: Sparse, real-valued numerical feature vector of dimensionality $N = 3500$.
* **Why Used**: TF-IDF down-weights universally frequent words across documents while promoting distinctive keywords characteristic of specific travel intents (e.g., *"pack"*, *"itinerary"*, *"biryani"*, *"rafting"*).

---

## 5. Multinomial Logistic Regression

* **Purpose**: Multi-class supervised classification mapping TF-IDF feature vectors to probability distributions over 26 travel intents.
* **Input**: $L_2$-normalized TF-IDF vector $\vec{x} \in \mathbb{R}^{3500}$.
* **Mathematical Formulation**:
  $$P(Y = k \mid \vec{x}) = \frac{\exp(\vec{w}_k^T \vec{x} + b_k)}{\sum_{j=1}^{K} \exp(\vec{w}_j^T \vec{x} + b_j)}$$
  Optimization minimizes multinomial cross-entropy with $L_2$ regularization:
  $$\mathcal{L}(W, b) = -\sum_{i=1}^M \sum_{k=1}^K y_{ik} \log P(Y = k \mid \vec{x}_i) + \frac{1}{2C} \sum_{k=1}^K \|\vec{w}_k\|_2^2$$
* **Output**: Predicted intent $\hat{y} = \arg\max_k P(Y=k \mid \vec{x})$ along with confidence score $\max_k P(Y=k \mid \vec{x})$.
* **Why Used**: Provides fast inference ($< 5\text{ ms}$), well-calibrated probabilities for multi-intent detection, and handles sparse, high-dimensional TF-IDF matrices without overfitting.

---

## 6. Hybrid Named Entity Recognition (NER) & Levenshtein Matching

* **Purpose**: Extract domain-specific travel entities (locations, budgets, durations, groups, mobility constraints) with spelling tolerance.
* **Input**: Raw query string $S$.
* **Process**:
  1. **spaCy Statistical NER**: Identifies standard geopolitical entities (`GPE`, `LOC`, `DATE`).
  2. **Domain Regex Matcher**: Parses budget patterns (`r"(?:₹|rs\.?|inr)\s*(\d+(?:,\d+)*)\s*(k)?"`), duration patterns (`r"\b(\d+)\s*days?"`), and origin phrases.
  3. **Levenshtein Fuzzy String Matcher**: Computes minimum single-character edit operations (insertions, deletions, substitutions):
     $$\text{Ratio}(s_1, s_2) = \frac{2 \cdot M}{|s_1| + |s_2|}$$
     Matches misspelled query tokens against known destinations if $\text{Ratio} \ge 0.75$.
* **Output**: Structured dictionary containing resolved destination, origin, duration, budget, and travel group.
* **Why Used**: Standard NER models fail on non-standard Indian travel syntax (*"8k"*, *"2-day getaway"*) and misspelled locations (*"Mumbay"*). Hybrid extraction guarantees high recall.

---

## 7. Cosine Similarity

* **Purpose**: Measure the angular orientation between two high-dimensional vectors, evaluating topical similarity independent of document length.
* **Input**: Query vector $\vec{u}$ and document/embedding vector $\vec{v}$.
* **Mathematical Formulation**:
  $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} = \frac{\sum_{i=1}^D u_i v_i}{\sqrt{\sum_{i=1}^D u_i^2} \sqrt{\sum_{i=1}^D v_i^2}}$$
* **Output**: Scalar similarity value bounded in $[-1.0, 1.0]$ (normalized to $[0.0, 1.0]$ for non-negative spaces).
* **Why Used**: Scale-invariant metric well-suited for high-dimensional semantic vector spaces.

---

## 8. Sentence Embeddings via Transformer Models

* **Purpose**: Encode whole sentences and semantic paragraphs into dense, continuous metric spaces capturing conceptual meaning.
* **Input**: Destination search descriptions and user query text.
* **Model**: `all-MiniLM-L6-v2` (6-layer MiniLM architecture, 384 hidden dimensions, 22.7M parameters).
* **Process**:
  1. Tokenize query into WordPiece tokens.
  2. Pass through multi-head self-attention transformer layers.
  3. Mean pooling across output token embeddings:
     $$\vec{e} = \frac{\sum_{i=1}^L m_i \cdot \vec{h}_i}{\sum_{i=1}^L m_i}$$
  4. Normalize to unit hypersphere $\|\vec{e}\|_2 = 1$.
* **Output**: Dense vector $\vec{e} \in \mathbb{R}^{384}$.
* **Why Used**: Captures semantic synonyms and conceptual paraphrases (e.g. *"disconnect from everything"* maps close to *"peaceful mountain retreat"*).

---

## 9. Dynamic Recommendation Scoring Algorithm

* **Purpose**: Rank candidate destinations by combining 8 distinct dimensions with dynamically adaptive constraint weights.
* **Input**: Candidate destinations with component similarity metrics and user constraints.
* **Mathematical Formulation**:
  $$\text{Score}(d) = \frac{\sum_{i=1}^8 w_i \cdot S_i(d)}{\sum_{i=1}^8 w_i}$$
  Where:
  * $S_{\text{sem}}$: Dense cosine semantic similarity $[0, 1]$.
  * $S_{\text{pref}}$: Lexical preference tag overlap $[0, 1]$.
  * $S_{\text{cat}}$: Category string overlap $[0, 1]$.
  * $S_{\text{bud}}$: Financial compatibility ratio:
    $$S_{\text{bud}} = \begin{cases} 1.0 & \text{if } C_{\text{est}} \le 0.90 \cdot B \\ 0.85 & \text{if } C_{\text{est}} \le 1.15 \cdot B \\ 0.50 & \text{if } C_{\text{est}} \le 1.40 \cdot B \\ \max(0.1, 1 - (C/B - 1) \cdot 0.5) & \text{otherwise} \end{cases}$$
  * $S_{\text{rat}}$: Normalized rating $\frac{\text{Rating} - 1.0}{4.0} \in [0, 1]$.
  * $S_{\text{dur}}$: Duration compatibility score.
  * $S_{\text{sea}}$: Seasonal alignment indicator.
  * $S_{\text{acc}}$: Physical mobility score (penalizes steep treks when traveling with parents).
* **Output**: Ordered ranking of destinations with personalized explainability bullets.
* **Why Used**: Prevents rigid filtering from eliminating high-quality alternatives while ensuring that strict user limitations (e.g. elderly mobility, low budgets) are prioritized.

---

## 10. Multi-Day Itinerary Generation Algorithm

* **Purpose**: Synthesize structured day-by-day travel schedules without overloading tourists.
* **Input**: Destination name, total days $N$ ($1 \le N \le 7$), and optional preferences.
* **Process**:
  1. Retrieve all attraction records matching destination or state from `attractions.csv`.
  2. Score attractions: $\text{Rank} = \text{Rating} + 0.5 \cdot \text{PrefMatch}$.
  3. Sort spatially using coordinates $(\text{lat}, \text{lon})$ to cluster proximate points of interest.
  4. Partition attractions into daily buckets: maximum 3 attractions per day with total visit duration $\le 7.0$ hours.
  5. Assign daily time slots: Morning (09:00 - 13:00), Afternoon (14:00 - 17:00), Evening (17:30 - 19:30).
  6. Interleave regional culinary dining highlights from `food.csv` for lunch and dinner.
* **Output**: Structured JSON and formatted markdown schedule with estimated daily entrance fees.
* **Why Used**: Replaces static, hardcoded travel guides with dynamic itineraries generated directly from empirical tourism data.
