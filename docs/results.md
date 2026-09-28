# Experimental Results & Evaluation — TravelMate

## 1. Intent Classification Model Evaluation

The machine learning intent classification model (TF-IDF + Logistic Regression) was evaluated on a stratified 20% holdout test dataset (112 unseen query examples across 26 distinct travel intent categories).

### Aggregate Performance Metrics
* **Total Training Set**: 444 examples
* **Total Test Set**: 112 examples (556 total query samples)
* **Overall Test Accuracy**: **63.39%** *(Across 26 mutually exclusive intent classes; random baseline = 3.85%)*
* **Weighted Precision**: **65.25%**
* **Weighted Recall**: **63.39%**
* **Weighted F1-Score**: **62.31%**
* **Macro Average Precision**: **61.97%**
* **Macro Average Recall**: **61.88%**
* **Macro F1-Score**: **60.01%**

### Evaluation Metrics by Key Intent Category

| Intent Category | Test Precision (%) | Test Recall (%) | Test F1-Score (%) |
| :--- | :--- | :--- | :--- |
| `itinerary_planning` | 71.43% | 83.33% | 76.92% |
| `food_recommendation` | 100.00% | 100.00% | 100.00% |
| `transportation` | 75.00% | 75.00% | 75.00% |
| `packing_advice` | 100.00% | 75.00% | 85.71% |
| `destination_comparison` | 100.00% | 75.00% | 85.71% |
| `greeting` | 100.00% | 100.00% | 100.00% |
| `help` | 100.00% | 75.00% | 85.71% |
| `budget_planning` | 75.00% | 75.00% | 75.00% |
| `best_time_to_visit` | 100.00% | 71.43% | 83.33% |
| `activity_recommendation` | 75.00% | 75.00% | 75.00% |

---

## 2. Automated Test Suite Results

The comprehensive test suite in `tests/run_all_tests.py` verified all functional modules:

```text
======================================================================
TEST SUMMARY REPORT:
Total Tests Run: 30
Passed:          30
Failures:        0
Errors:          0
======================================================================
🎉 ALL TESTS PASSED SUCCESSFULLY!
```

* **`test_preprocessing.py`**: Validated text cleaning, contractions expansion, travel-token preservation (`₹8000`), selective stopword filtering, and lemmatization.
* **`test_entities.py`**: Validated exact matching, Levenshtein fuzzy string distance matching (*"Mumbay"* $\rightarrow$ *"Mumbai"*, *"Mahabaleshwr"* $\rightarrow$ *"Mahabaleshwar"*, *"Gokarn"* $\rightarrow$ *"Gokarna"*), budget extraction, duration extraction, and crowd/walking constraint parsing.
* **`test_intent.py`**: Validated intent prediction, probability calibration, multi-intent detection, and handling of complex open-ended unseen queries.
* **`test_recommender.py`**: Validated dynamic composite scoring across multi-constraint queries.
* **`test_budget.py`**: Validated arithmetic consistency across accommodation, dining, transit, activities, buffer items, and feasibility classification.
* **`test_itinerary.py`**: Validated dynamic 3-day itinerary generation and text formatting.

---

## 3. End-to-End Acceptance Test Verification

The acceptance test validated the multi-turn scenario specified in Requirement 52:
* **Turn 1 (Complex Unseen Query)**: *"I'm travelling from Mumbai for just two days. I don't want to spend more than ₹7000, I love nature and photography, and I want somewhere that isn't too crowded."*
  * **Result**: Correctly parsed Origin = Mumbai, Duration = 2 days, Budget = ₹7000, Preferences = [nature, photography], Crowd = Low. Successfully retrieved and ranked Matheran (#1), explained why it matches, and stored Matheran as active destination in session context.
* **Turn 2 (Pronoun Follow-up)**: *"What can I do there?"*
  * **Result**: Resolved pronoun *"there"* to Matheran without re-asking. Retrieved Louisa Point, Charlotte Lake, Panorama Point, and Echo Point.
* **Turn 3 (Constraint Follow-up)**: *"Can I do it in two days?"*
  * **Result**: Retained Matheran as active destination, updated duration to 2 days, computed budget breakdown (₹3850 total), and assessed financial feasibility (confirmed comfortable surplus of ₹3150).
