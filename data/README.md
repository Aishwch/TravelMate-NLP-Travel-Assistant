# TravelMate Data Repository

This directory contains the knowledge base and datasets supporting the **TravelMate** NLP-Based Intelligent Travel Assistance System. All data is sourced from genuine public repositories (Wikidata, Kaggle Indian Food 101, Hugging Face Travel_india, Indian Cities Census) and structured into canonical schemas.

---

## Directory Structure

```text
data/
├── raw/                              # Unprocessed raw open-source datasets
│   ├── wikidata_destinations_raw.json # 350 raw Wikidata tourism entities (CC0)
│   ├── wikidata_attractions_raw.json  # 600 raw Wikidata attractions/monuments (CC0)
│   ├── indian_food_raw.csv           # 255 authentic Indian dishes from Kaggle (PDDL/CC0)
│   ├── travel_india_hf_raw.csv       # 1,000 travel QA records from Hugging Face (Apache 2.0)
│   ├── indian_cities_raw.csv         # 493 Indian cities from Indian Census (GODL)
│   └── transport_raw.csv             # 55 intercity transit route benchmarks (GODL)
│
├── processed/                        # Normalized, cleaned, validated datasets used by TravelMate
│   ├── destinations.csv              # 283 canonical destination records
│   ├── attractions.csv               # 547 canonical tourist attractions with GPS coordinates
│   ├── food.csv                      # 255 authentic regional dishes with dietary classifications
│   ├── budget.csv                    # 283 structured destination cost models
│   ├── transport.csv                 # 55 inter-city transit routes and durations
│   └── packing_tips.csv              # 6 seasonal packing and safety guidelines
│
└── README.md                         # This documentation file
```

---

## Datasets Overview & Canonical Schemas

### 1. Destinations (`data/processed/destinations.csv` — 283 records)
* **`destination_id`**: Canonical identifier (e.g., `D001`).
* **`destination`**: Destination name (e.g., `Goa`, `Jaipur`, `Matheran`, `Manali`).
* **`city`**: City / district.
* **`state`**: Indian State / Union Territory.
* **`country`**: `India`.
* **`description`**: Comprehensive multi-sentence description capturing ambiance, terrain, and highlights.
* **`category`**: Core category labels (e.g., `Beach`, `Hill Station`, `Heritage`, `Spiritual`, `Wildlife`).
* **`activities`**: Available activities (e.g., `Trekking, Photography, Sightseeing, Cafe Hopping`).
* **`rating`**: Satisfaction rating (1.0 to 5.0).
* **`estimated_cost_per_day`**: Average daily expenditure per person in INR (₹).
* **`duration`**: Recommended stay duration (e.g., `3-4 days`).
* **`best_season`**: Best visiting season (`Winter`, `Monsoon`, `Summer`, `All Year`).
* **`best_months`**: Recommended calendar months.
* **`travel_group`**: Target travel groups (`Solo, Friends, Family, Couples`).
* **`crowd_level`**: Tourist density (`Low`, `Moderate`, `High`).
* **`walking_level`**: Physical exertion / walking required (`Low`, `Moderate`, `High`) for senior-citizen and family queries.
* **`famous_for`**: Key landmark specialties.
* **`suitable_for`**: Normalized tags (`peaceful, photography, budget, romantic, adventure, nature, family, solo, weekend`).
* **`latitude` / `longitude`**: Geographic coordinates.
* **`search_text`**: Enriched dense text representation for `all-MiniLM-L6-v2` semantic vector search.

### 2. Attractions (`data/processed/attractions.csv` — 547 records)
* **`attraction_id`**: Canonical identifier (`A001`).
* **`attraction_name`**: Name of the attraction / monument.
* **`destination`**: Destination reference.
* **`city`**: Local city.
* **`state`**: State.
* **`category`**: Attraction classification (`Fort / Historical`, `Beach`, `Viewpoint / Nature`, `Spiritual`, etc.).
* **`description`**: Overview of historical and scenic significance.
* **`rating`**: Visitor rating (1.0 to 5.0).
* **`entry_fee`**: Ticket price in INR (`0` for free public access).
* **`visit_duration_hours`**: Recommended time to spend at attraction.
* **`latitude` / `longitude`**: GPS coordinates for route clustering during itinerary generation.
* **`search_text`**: Enriched text for semantic search.

### 3. Food & Cuisine (`data/processed/food.csv` — 255 records)
* **`food_id`**: Unique dish identifier (`F001`).
* **`destination`**: Closest destination reference.
* **`city` / `state`**: Origin city and state.
* **`food_name`**: Name of the dish.
* **`cuisine`**: Regional cuisine (e.g., `Maharashtrian`, `Rajasthani`, `Goan`, `Punjabi`, `South Indian`).
* **`diet`**: `vegetarian` / `non-vegetarian`.
* **`course`**: Course classification (`main course`, `snack`, `dessert`).
* **`flavor_profile`**: Flavor notes (`spicy`, `sweet`, `savory`).
* **`description`**: Narrative culinary description with key ingredients.

### 4. Travel Budget Matrix (`data/processed/budget.csv` — 283 records)
* **`destination`**: Destination name.
* **`accommodation_cost_per_day`**: Base standard room rate.
* **`food_cost_per_day`**: Daily food expenditure benchmark.
* **`local_transport_cost_per_day`**: Local cab/auto/transit daily benchmark.
* **`sightseeing_cost_per_day`**: Daily attraction entry ticket benchmark.
* **`budget_tier`**: Budget category (`Budget`, `Moderate`, `Luxury`).

### 5. Transport Matrix (`data/processed/transport.csv` — 55 records)
* **`origin`**: Departure hub (e.g., `Mumbai`, `Delhi`, `Bangalore`, `Pune`).
* **`destination`**: Arrival destination.
* **`mode`**: Transport mode (`Train`, `Flight`, `Drive / Road`, `Bus`).
* **`estimated_duration_hours`**: Typical transit time.
* **`estimated_cost_inr`**: Indicative fare in INR.
* **`distance_km`**: Approximate road/rail/air distance.
* **`advisory_notes`**: Useful route guidance (e.g., scenic ghats, eco-sensitive vehicle bans).

### 6. Packing Tips & Guidelines (`data/processed/packing_tips.csv` — 6 records)
* **`season_or_context`**: Climate / travel context (`Monsoon`, `Winter`, `Summer`, `Coastal`, `High Altitude / Mountain`, `Desert`).
* **`clothing_essentials`**: Recommended attire.
* **`gear_and_accessories`**: Essential gear (e.g., rain cover, warm layers).
* **`health_and_hygiene`**: Medical and personal care items.
* **`special_notes`**: Crucial advisories (hydration, AMS precautions, emergency contacts).
