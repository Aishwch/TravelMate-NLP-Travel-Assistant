# Problem Statement — TravelMate

## 1. Background & Context

Vacation and trip planning is inherently a multi-objective decision-making process. A typical traveler must balance budget limitations, time availability, transit logistics, seasonal weather patterns, regional cuisines, companionship dynamics, and personal leisure preferences. 

While abundant travel data exists on the web, modern online travel platforms impose severe friction on users:
* **Facet Overload**: Users must manually select multiple checkboxes, date pickers, budget sliders, and city menus across fragmented tabs.
* **Lack of Natural Query Understanding**: Traditional search engines on travel portals rely on exact lexical substring matching (e.g., searching for "peaceful beach" returns nothing if the property description only says "secluded coastal haven").
* **Superficial "Rule-Based" Chatbots**: Most student and commercial customer-service chatbots utilize static keyword dictionaries (`if "goa" in query: return ...`) or rigid predefined buttons. When an unconventional question is submitted, these bots fail catastrophically.

---

## 2. Core Problem Definition

> **"How can we engineer an intelligent, locally runnable travel assistance system that comprehends open-ended natural language travel queries, extracts multidimensional constraints (budgets, party sizes, duration, physical accessibility), performs semantic retrieval over real structured datasets without relying on paid external APIs, and generates transparent, explainable recommendations and itineraries across multi-turn conversational dialogue?"**

---

## 3. Specific Technical Challenges Addressed

1. **Typos & Entity Variations**: Travelers frequently misspell destination names (e.g., *"Mahabaleshwr"*, *"Mumbay"*, *"Gokarn"*). Standard NER models either classify these as unknown nouns or mislabel them.
2. **Polysemy and Paraphrasing**: Queries like *"I want to escape the crowds"*, *"quiet nature destination"*, and *"somewhere peaceful"* share an identical semantic intent despite completely disjoint vocabularies.
3. **Multi-Constraint Optimization**: Resolving queries containing four or more simultaneous constraints:
   $$\text{Query} = f(\text{Origin}, \text{Budget}, \text{Duration}, \text{Party}, \text{Mobility}, \text{Season})$$
4. **Conversational Contextual Ellipsis**: Resolving pronouns and dependent follow-up queries (e.g., Turn 1: *"Suggest places near Mumbai"*; Turn 2: *"What can I do there?"*; Turn 3: *"Can I do it in two days?"*).
5. **No Fake AI / No Hardcoded Response Dictionary**: The system must utilize actual NLP pipelines, vector cosine calculations, and machine learning classifiers rather than hardcoded lookup tables.
