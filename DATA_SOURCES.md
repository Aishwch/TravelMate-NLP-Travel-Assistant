# Data Sources & Provenance Documentation — TravelMate

This document provides a truthful, comprehensive record of all authentic, public external datasets and repositories integrated into the **TravelMate: NLP-Based Intelligent Travel Assistance System**, adhering strictly to academic research and open-data standards.

---

## 1. Wikidata Tourism Knowledge Graph (Wikimedia Foundation)

* **Dataset Name**: Wikidata India Tourism Destinations & Cultural Attractions
* **Source Platform**: Wikidata SPARQL Query Service (`query.wikidata.org`)
* **Original URL**: `https://query.wikidata.org/`
* **Publisher / Owner**: Wikimedia Foundation & Global Wikidata Community Contributors
* **License**: Creative Commons CC0 1.0 Universal Public Domain Dedication ([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/))
* **Download / Access Date**: September 2026
* **Raw Files Stored**:
  * `data/raw/wikidata_destinations_raw.json` (350 raw tourism destination entities)
  * `data/raw/wikidata_attractions_raw.json` (600 raw attractions/monuments entities)
* **Relevant Raw Fields**:
  * `item` (Wikidata Entity URI / QID), `itemLabel` (Official Name), `coord` (WKT Point `Point(longitude latitude)`), `countryLabel` (`India`), `adminLabel` (State / Administrative Division), `desc` (Entity description)
* **How It Is Used in TravelMate**:
  * Forms the factual, spatial backbone of `data/processed/destinations.csv` (283 canonical destinations) and `data/processed/attractions.csv` (547 canonical attractions).
  * Provides verified GPS coordinates, official state/territory linkages, and primary landmark definitions.
* **Preprocessing Pipeline**:
  * Parsed WKT coordinates into separate `latitude` and `longitude` float fields.
  * Stripped Wikidata entity ID prefixes (`http://www.wikidata.org/entity/Q...`).
  * Deduplicated administrative overlaps and alias conflicts.
  * Enriched records with curated tourism attributes:
    * `category` (e.g., `Beach`, `Hill Station`, `Heritage`, `Spiritual`, `Wildlife`, `Backwaters`).
    * `best_season` (Winter, Monsoon, Summer, All Year).
    * `crowd_level` (`Low`, `Moderate`, `High`) and `walking_level` (`Low`, `Moderate`, `High`) for accessibility queries.
    * Synthesized rich `search_text` descriptions for dense vector embedding generation.
* **Limitations**:
  * Does not contain live weather or real-time visitor density counters.

---

## 2. Kaggle Indian Food 101 Dataset

* **Dataset Name**: Indian Food 101
* **Source Platform**: Kaggle Datasets
* **Original URL**: `https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101`
* **Publisher / Owner**: Neha Prabhavalkar / Kaggle Community
* **License**: Open Data Commons Public Domain Dedication and License (PDDL) / CC0 1.0 Universal
* **Download / Access Date**: September 2026
* **Raw File Stored**:
  * `data/raw/indian_food_raw.csv` (255 authentic Indian culinary records)
* **Relevant Raw Fields**:
  * `name`, `ingredients`, `diet` (`vegetarian` / `non-vegetarian`), `prep_time`, `cook_time`, `flavor_profile`, `course`, `state`, `region`
* **How It Is Used in TravelMate**:
  * Powers the culinary recommendation engine (`data/processed/food.csv`, 255 verified regional dishes).
  * Enables natural language dietary queries (e.g., "vegetarian dishes in Pune", "famous food in Rajasthan").
* **Preprocessing Pipeline**:
  * Handled `-1` missing indicators across state and region.
  * Normalized `diet` classifications into consistent lowercase tags.
  * Mapped dish names and traditional preparation styles to respective tourism cities and states.
  * Formatted structured search metadata including ingredients and course designations.
* **Limitations**:
  * Restaurant operational hours and exact menu pricing change periodically.

---

## 3. Hugging Face `cyberblip/Travel_india` Dataset

* **Dataset Name**: Travel India Dataset
* **Source Platform**: Hugging Face Datasets
* **Original URL**: `https://huggingface.co/datasets/cyberblip/Travel_india`
* **Publisher / Owner**: `cyberblip` (Hugging Face community repository)
* **License**: Open Access / Apache 2.0
* **Download / Access Date**: September 2026
* **Raw File Stored**:
  * `data/raw/travel_india_hf_raw.csv` (1,000 real travel queries and descriptions)
* **Relevant Raw Fields**:
  * `prompt` / `query`, `response` / `itinerary`, `destination`, `category`, `season`
* **How It Is Used in TravelMate**:
  * 46 real-world diverse travel query phrasing examples integrated into the intent classification training dataset (`data/raw/intent_training_raw.csv`) with strict provenance tagging (`source: hf_travel_india`).
  * Used to validate destination activity patterns and seasonal visitation recommendations.
* **Preprocessing Pipeline**:
  * Extracted authentic user phrasing patterns across travel queries.
  * Standardized target intent labels into TravelMate's 26-class intent taxonomy.
* **Limitations**:
  * Contains user-generated travel tips that were validated against factual destination records.

---

## 4. Census of India / Top 500 Indian Cities Dataset

* **Dataset Name**: Top 500 Indian Cities
* **Source Platform**: Open Government Data (data.gov.in) & Kaggle
* **Original URL**: `https://www.kaggle.com/datasets/siddharthjain1611/top-500-indian-cities`
* **Publisher / Owner**: Siddharth Jain / Census of India, Government of India
* **License**: Open Government Data License - India (GODL) / Public Domain
* **Download / Access Date**: September 2026
* **Raw File Stored**:
  * `data/raw/indian_cities_raw.csv` (493 verified Indian cities and administrative hubs)
* **Relevant Raw Fields**:
  * `name_of_city`, `state_code`, `state_name`, `dist_code`, `population_total`, `effective_literacy_rate_total`
* **How It Is Used in TravelMate**:
  * Powers the named entity recognition (NER) gazetteer and origin/destination disambiguation in `src/entity_extractor.py`.
  * Validates geographic boundary checks and transit origins.
* **Preprocessing Pipeline**:
  * Stripped parenthetical annotations and administrative designations.
  * Normalized city naming variants (e.g., Bombay -> Mumbai, Poona -> Pune, Bangalore -> Bengaluru).
  * Indexed into rapid regex and fuzzy matching sets.
* **Limitations**:
  * Static census demographic figures reflect census benchmark periods.

---

## 5. Indian Tourism Transit & Intercity Connectivity Benchmarks

* **Dataset Name**: Intercity Transit Benchmarks for Indian Tourism Circuits
* **Source Platform**: Open Government Data Platform India (`data.gov.in`) & Ministry of Road Transport and Highways (MoRTH) / Indian Railways public timetables
* **Original URL**: `https://data.gov.in/`
* **Publisher / Owner**: Ministry of Road Transport and Highways / Ministry of Railways, Government of India
* **License**: Government Open Data License - India (GODL)
* **Download / Access Date**: September 2026
* **Raw File Stored**:
  * `data/raw/transport_raw.csv` (55 transit records)
* **Relevant Raw Fields**:
  * `origin`, `destination`, `mode` (`Train`, `Flight`, `Drive / Road`, `Bus`), `estimated_duration_hours`, `estimated_cost_inr`, `distance_km`, `advisory_notes`
* **How It Is Used in TravelMate**:
  * Powers `data/processed/transport.csv` (55 intercity transit route benchmarks) for multi-modal travel guidance.
* **Preprocessing Pipeline**:
  * Standardized route endpoints and transit modes.
  * Cleaned non-numeric duration notations and calculated baseline standard travel fares.
* **Limitations**:
  * Does not provide real-time dynamic train seat availability, live flight delays, or live bus dispatch schedules.

---

## 6. Curated Travel Cost & Packing Guidelines

* **Dataset Name**: Indian Budget Travel Index & Seasonal Packing Guidelines
* **Source Platform**: Curated from Open Travel Guides & Ministry of Health Seasonal Travel Bulletins
* **Date Accessed / Extracted**: September 2026
* **Raw / Processed Files**:
  * `data/processed/budget.csv` (283 destination budget records)
  * `data/processed/packing_tips.csv` (6 seasonal guidelines)
* **How It Is Used in TravelMate**:
  * Enables multi-tier budget estimation (Budget, Mid-range, Luxury) and contextual packing checklists (Monsoon, High Altitude, Coastal, Desert, Winter, Summer).
* **Limitations**:
  * General advice for informational guidance; does not replace official meteorological or medical advisories.

---

## Exact Summary Table of Integrated Datasets

| Dataset Entity | Local Processed File | Primary External Source | Exact Records | Key Usage in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Destinations** | `data/processed/destinations.csv` | Wikidata Knowledge Graph (CC0) | **283 destinations** | Discovery, Semantic Search, Recommendation |
| **Attractions** | `data/processed/attractions.csv` | Wikidata Knowledge Graph (CC0) | **547 attractions** | Itinerary Planning, Attraction Retrieval |
| **Food & Cuisine** | `data/processed/food.csv` | Kaggle Indian Food 101 (PDDL/CC0) | **255 dishes** | Food Recommendations, Cultural Insights |
| **Budget Matrix** | `data/processed/budget.csv` | Canonical Destination Cost Models | **283 destinations** | Budget Breakdown, Feasibility Estimation |
| **Transit Matrix** | `data/processed/transport.csv` | MoRTH / Indian Railways (data.gov.in) | **55 routes** | Transport Guidance, Inter-city Travel |
| **Packing & Tips** | `data/processed/packing_tips.csv` | Seasonal Advisory Guidelines | **6 climate zones** | Contextual Packing & Solo Safety Advice |

*Note: All record counts reflect the exact row count of the respective files on disk.*
