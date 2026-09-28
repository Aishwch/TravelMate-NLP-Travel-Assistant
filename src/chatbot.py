"""
TravelMate — Intelligent NLP Travel Assistant Engine
=====================================================
The central conversational orchestrator integrating:
  - Text Normalization & Preprocessing (preprocessing.py)
  - Hybrid Entity Extraction (entity_extractor.py)
  - Machine Learning Intent Classification (intent_classifier.py)
  - Semantic Preference Extraction (preference_extractor.py)
  - Session Context & Conversational Memory (context_manager.py)
  - Semantic Embedding Retrieval (semantic_search.py)
  - Multi-Factor Dynamic Recommendation (recommender.py)
  - Day-by-Day Itinerary Generation (itinerary.py)
  - Feasibility Budget Breakdown (budget.py)
  - Comparison, Food, Transit, and Packing Advisories (utils.py)
"""

import os
import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from src.preprocessing import preprocess_query, clean_text
from src.entity_extractor import TravelEntityExtractor
from src.intent_classifier import IntentClassifier
from src.preference_extractor import PreferenceExtractor
from src.semantic_search import SemanticSearchEngine
from src.retriever import HybridRetriever
from src.recommender import TravelRecommender
from src.itinerary import ItineraryGenerator
from src.budget import BudgetPlanner
from src.context_manager import ConversationContext
from src.utils import (
    compare_destinations,
    get_food_guidance,
    get_transport_guidance,
    get_packing_guidance
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")


class TravelMateAssistant:
    """
    Main conversational agent for TravelMate.
    """

    def __init__(self, context: Optional[ConversationContext] = None):
        # 1. Load Processed Datasets
        self.destinations_df = self._load_csv("destinations.csv")
        self.attractions_df = self._load_csv("attractions.csv")
        self.food_df = self._load_csv("food.csv")
        self.budget_df = self._load_csv("budget.csv")
        self.transport_df = self._load_csv("transport.csv")
        self.packing_df = self._load_csv("packing_tips.csv")

        # 2. Initialize Core Subsystems
        self.context = context or ConversationContext()
        self.entity_extractor = TravelEntityExtractor(self.destinations_df)
        self.intent_classifier = IntentClassifier(auto_train=True)
        self.preference_extractor = PreferenceExtractor()
        self.semantic_engine = SemanticSearchEngine()
        self.retriever = HybridRetriever(self.destinations_df, self.semantic_engine)
        self.recommender = TravelRecommender(self.retriever)
        self.itinerary_gen = ItineraryGenerator(self.attractions_df, self.food_df)
        self.budget_planner = BudgetPlanner(self.budget_df)

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Loads a processed dataset from data/processed/."""
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def respond(self, user_query: str) -> Dict[str, Any]:
        """
        Processes a natural language travel query through the complete NLP pipeline
        and generates a contextual, informative response.
        """
        raw_query = str(user_query or "").strip()
        if not raw_query:
            return {
                "response": "Hello! I am **TravelMate**, your intelligent travel assistant. Tell me about your travel dreams, destination queries, budget constraints, or preferred vacation style!",
                "intent": "greeting",
                "entities": {},
                "preferences": []
            }

        # Step 1: Preprocessing & Normalization
        prep = preprocess_query(raw_query)

        # Step 2: Entity Extraction
        raw_entities = self.entity_extractor.extract_all_entities(raw_query)

        # Step 3: Context Resolution (resolves pronouns like 'there', 'it')
        resolved_entities = self.context.resolve_context(raw_query, raw_entities)

        # Step 4: Intent Classification & Confidence
        predicted_intent, confidence, top_k_intents = self.intent_classifier.predict_with_confidence(raw_query)
        multi_intents = self.intent_classifier.detect_multi_intents(raw_query)

        # Step 5: Semantic Preference Extraction
        preferences = self.preference_extractor.extract_preferences(raw_query)

        # Check for destination comparison intent
        all_dests = resolved_entities.get("all_destinations", [])
        if len(all_dests) >= 2 or ("better" in raw_query.lower() and len(all_dests) >= 2):
            predicted_intent = "destination_comparison"

        # Step 6: Dynamic Dispatch & Response Synthesis
        response_text, recommended_dests = self._dispatch_and_generate(
            query=raw_query,
            intent=predicted_intent,
            multi_intents=multi_intents,
            entities=resolved_entities,
            preferences=preferences,
            confidence=confidence
        )

        # Step 7: Update Conversation Context Memory
        self.context.update(
            query=raw_query,
            entities=resolved_entities,
            intent=predicted_intent,
            preferences=preferences,
            recommended_destinations=recommended_dests
        )
        self.context.add_history(raw_query, response_text, predicted_intent)

        return {
            "response": response_text,
            "intent": predicted_intent,
            "confidence": confidence,
            "top_k_intents": top_k_intents,
            "multi_intents": multi_intents,
            "entities": resolved_entities,
            "preferences": preferences,
            "recommended_destinations": recommended_dests,
            "preprocessed": prep,
            "context_summary": self.context.get_summary()
        }

    def _dispatch_and_generate(
        self,
        query: str,
        intent: str,
        multi_intents: List[str],
        entities: Dict[str, Any],
        preferences: List[str],
        confidence: float
    ) -> Tuple[str, List[str]]:
        """
        Dispatches request to specialized generation modules based on intent and constraints.
        """
        q_lower = query.lower()
        active_dest = entities.get("destination")
        recommended_dest_names: List[str] = []

        # 0. Gibberish / Nonsensical Input Detection
        clean_words = re.findall(r"[A-Za-z]+", q_lower)
        if clean_words and len(clean_words) <= 2:
            longest = max(clean_words, key=len)
            has_vowel = bool(re.search(r"[aeiouy]", longest))
            if not has_vowel or (len(longest) >= 7 and not any(w in longest for w in ["travel", "holiday", "itinerary", "weekend", "monsoon", "weather", "booking", "tourism", "vacation"])):
                if not active_dest and not entities.get("all_destinations") and not preferences:
                    return (
                        "I couldn't quite understand that. I am **TravelMate**, your intelligent NLP travel assistant. "
                        "You can ask me to discover destinations, plan custom day-by-day itineraries, estimate budgets, "
                        "recommend local cuisines, or explore travel routes across India!\n\n"
                        "💡 *Try asking: 'Suggest a peaceful 3-day trip from Mumbai' or 'Plan a trip to Jaipur'.*",
                        []
                    )

        # 1. Greetings
        if intent == "greeting" or q_lower in ["hi", "hello", "hey", "namaste", "greetings"]:
            return (
                "👋 **Hello and welcome to TravelMate!** I am your intelligent NLP travel assistant.\n\n"
                "I can help you discover destinations, generate custom day-by-day itineraries, estimate budgets, "
                "suggest authentic local cuisines, explore transport routes, compare places, and provide tailored packing tips.\n\n"
                "💡 *Feel free to type any question naturally — like:*  \n"
                "* 'I have ₹8000 and 3 days from Mumbai and want a peaceful trip'  \n"
                "* 'Plan a 3 day trip to Jaipur'  \n"
                "* 'Which is better for a weekend, Lonavala or Matheran?'  \n"
                "* 'What food should I try in Hyderabad?'",
                []
            )

        # 2. Help (only when user asks about system capabilities, not destination activities)
        is_activity_query = any(w in q_lower for w in ["there", "in ", "at "]) or bool(active_dest and any(w in q_lower for w in ["do", "see", "visit", "explore", "activities"]))
        if (intent == "help" or q_lower in ["help", "what can you do", "features"]) and not is_activity_query:
            return (
                "### 🌍 What TravelMate Can Do For You:\n\n"
                "1. **🎯 Destination Recommendations**: Tailored suggestions matching your budget, preferred duration, travel group, and scenery.\n"
                "2. **📅 Custom Itineraries**: Dynamic day-by-day sightseeing schedules balancing morning, afternoon, and evening sights.\n"
                "3. **💰 Budget Estimation**: Realistic breakdowns for accommodation, meals, transit, and entry fees with feasibility assessments.\n"
                "4. **⚖️ Destination Comparisons**: Side-by-side analysis of costs, crowd levels, ratings, and accessibility.\n"
                "5. **🍽️ Culinary Guides**: Iconic regional dishes and iconic local food spots.\n"
                "6. **🚆 Transit Guidance**: Realistic travel modes, approximate journey times, and route advice.\n"
                "7. **🎒 Packing & Safety Advice**: Tailored packing checklists for monsoon, winter, desert, coastal, or solo trips.\n\n"
                "Just ask whatever you need in plain, natural English!",
                []
            )

        # 3. Smart Clarification for Vague Queries
        if q_lower in ["plan a trip", "plan a trip for me", "suggest a trip", "i want to travel", "i don't know where to go", "suggest something"]:
            return (
                "I would love to help you plan an unforgettable trip! 🌍  \n\n"
                "To give you the most tailored suggestions, could you tell me a little more? For example:\n"
                "* **Where** would you like to travel, or which city are you starting from?\n"
                "* **How many days** do you have in mind?\n"
                "* What is your approximate **budget** or preferred vacation style (e.g. *peaceful nature, beaches, forts & heritage, adventure*)?",
                []
            )

        # 4. General Travel Questions (Duration, Senior Accessibility, Best Season, Solo Travel)
        if any(p in q_lower for p in ["how many days", "how much time", "duration for", "days are enough", "days required"]):
            target = active_dest or (entities.get("all_destinations", [None])[0])
            if target:
                dest_row = self.destinations_df[self.destinations_df["destination"].str.lower() == target.lower()]
                if not dest_row.empty:
                    d_days = int(dest_row.iloc[0].get("duration_days", 3))
                    d_cat = dest_row.iloc[0].get("category", "sightseeing")
                    return (
                        f"⏱️ **Ideal Duration for {target.title()}**:\n\n"
                        f"Typically, **{d_days} days** is recommended to explore {target.title()} comfortably. "
                        f"This allows enough time to visit its famous {d_cat.lower()} attractions, enjoy local dining, and explore without feeling rushed.\n\n"
                        f"💬 *Would you like me to generate a {d_days}-day itinerary for {target.title()}?*",
                        [target]
                    )

        if any(p in q_lower for p in ["suitable for parents", "good for parents", "for elderly", "senior citizens", "can parents walk", "suitable for elderly"]):
            target = active_dest or (entities.get("all_destinations", [None])[0])
            if target:
                dest_row = self.destinations_df[self.destinations_df["destination"].str.lower() == target.lower()]
                if not dest_row.empty:
                    w_lvl = dest_row.iloc[0].get("walking_level", "Moderate")
                    c_lvl = dest_row.iloc[0].get("crowd_level", "Moderate")
                    advice = "highly suitable for parents and seniors with gentle sightseeing and minimal walking" if w_lvl == "Low" else "involves moderate walking across viewpoints and historic sites; hiring local taxis or cabs is recommended for parents"
                    return (
                        f"👨‍👩‍👧 **Accessibility & Senior-Friendliness of {target.title()}**:\n\n"
                        f"• **Walking Requirement**: {w_lvl}\n"
                        f"• **Crowd Level**: {c_lvl}\n\n"
                        f"**Assessment**: {target.title()} is {advice}.\n\n"
                        f"💡 *Travel Tip: We recommend booking centrally located hotels with elevator access and arranging private transport.*",
                        [target]
                    )

        if any(p in q_lower for p in ["best season", "best time to visit", "when should i go", "when should i visit", "ideal time", "when is the best"]):
            target = active_dest or (entities.get("all_destinations", [None])[0])
            if target:
                dest_row = self.destinations_df[self.destinations_df["destination"].str.lower() == target.lower()]
                if not dest_row.empty:
                    season = dest_row.iloc[0].get("best_season", "October to March")
                    return (
                        f"🌤️ **Best Time to Visit {target.title()}**:\n\n"
                        f"The ideal period to visit **{target.title()}** is **{season}**, when weather conditions are most pleasant and optimal for outdoor sightseeing.",
                        [target]
                    )

        if "solo" in q_lower and any(w in q_lower for w in ["tips", "advice", "precaution", "safety", "keep in mind"]):
            return (
                "🎒 **Essential Safety & Guidance for Solo Travelers**:\n\n"
                "1. **Share Itinerary**: Always share your live accommodation and daily route details with a trusted family member or friend.\n"
                "2. **Keep Offline Backups**: Save offline copies of government IDs, train/bus tickets, and offline Google Maps.\n"
                "3. **Stay in Central Locations**: Choose well-reviewed hostels, homestays, or hotels in well-lit, active central neighborhoods.\n"
                "4. **Emergency Numbers**: Keep national emergency numbers saved (National Police: 112, Women Helpline: 1091).\n"
                "5. **Local Transport**: Prefer prepaid authorized taxis, public transit, or rideshare apps over unmetered private cabs after dark.",
                []
            )

        # 5. Destination Comparison
        if intent == "destination_comparison" or entities.get("is_comparison"):
            dests = entities.get("all_destinations", [])
            if len(dests) >= 2:
                resp = compare_destinations(dests[0], dests[1], self.destinations_df, self.budget_df)
                return resp, dests[:2]
            elif len(dests) == 1 and self.context.active_destination and dests[0].lower() != self.context.active_destination.lower():
                resp = compare_destinations(self.context.active_destination, dests[0], self.destinations_df, self.budget_df)
                return resp, [self.context.active_destination, dests[0]]

        # 5. Itinerary Planning
        if intent == "itinerary_planning" or any(w in q_lower for w in ["itinerary", "plan a trip to", "plan my trip"]):
            if active_dest:
                days = entities.get("duration_days") or 3
                itin_data = self.itinerary_gen.generate_itinerary(
                    destination=active_dest,
                    days=days,
                    preferences=preferences,
                    travel_group=entities.get("travel_group")
                )
                formatted = self.itinerary_gen.format_as_text(itin_data)
                return formatted, [active_dest]
            else:
                return (
                    "I would be glad to craft a customized day-by-day itinerary! Which destination would you like me to plan for? "
                    "(For example: *Jaipur, Goa, Munnar, Matheran, or Manali*)",
                    []
                )

        # 6. Budget Planning
        if intent == "budget_planning" or (entities.get("budget_inr") and active_dest and any(w in q_lower for w in ["budget", "cost", "expense", "spend"])):
            if active_dest:
                plan = self.budget_planner.plan_budget(
                    destination=active_dest,
                    total_budget=entities.get("budget_inr"),
                    duration_days=entities.get("duration_days") or 3,
                    people_count=entities.get("people_count") or 1
                )
                formatted = self.budget_planner.format_as_text(plan)
                return formatted, [active_dest]

        # 7. Transportation Guidance
        if intent == "transportation" or any(w in q_lower for w in ["how to reach", "travel from", "how can i get to", "train to", "flight to"]):
            origin = entities.get("origin") or self.context.origin or "Mumbai"
            if active_dest and active_dest.lower() != origin.lower():
                resp = get_transport_guidance(origin, active_dest, self.transport_df)
                return resp, [active_dest]
            elif entities.get("all_destinations") and len(entities["all_destinations"]) >= 1:
                target = entities["all_destinations"][0]
                resp = get_transport_guidance(origin, target, self.transport_df)
                return resp, [target]

        # 8. Food & Culinary Guidance
        if intent == "food_recommendation" or any(w in q_lower for w in ["food", "cuisine", "what should i eat", "dishes", "restaurant"]):
            target_dest = active_dest or (entities.get("all_destinations", [None])[0])
            if target_dest:
                diet = "veg" if "veg" in q_lower and "non" not in q_lower else ("non-veg" if "non" in q_lower else None)
                resp = get_food_guidance(target_dest, self.food_df, dietary_filter=diet)
                return resp, [target_dest]

        # 9. Packing Advice & Travel Safety
        if intent in ("packing_advice", "safety_advice") or any(w in q_lower for w in ["pack", "what should i wear", "carry", "safety", "precaution"]):
            resp = get_packing_guidance(query, active_dest, self.packing_df)
            return resp, [active_dest] if active_dest else []

        # 10. Attraction & Activity Recommendations for Known Destination
        if (
            (intent in ("attraction_recommendation", "activity_recommendation") or any(w in q_lower for w in ["what to do", "what can i do", "places to see", "attractions"]))
            and active_dest
        ):
            # Retrieve attractions for active destination
            attr_matches = self.semantic_engine.search_attractions(query, destination=active_dest, top_k=4)
            if attr_matches:
                lines = [f"### 📍 Top Attractions & Activities in {active_dest.title()}\n"]
                for attr, score in attr_matches:
                    fee = f"₹{attr.get('entry_fee', 0)}" if attr.get('entry_fee') else "Free Entry"
                    lines.append(f"* **{attr['attraction_name']}** ({attr.get('category', 'Sightseeing')})")
                    lines.append(f"  * {attr.get('description', '')}")
                    lines.append(f"  * *Visitor Rating*: ⭐ {attr.get('rating', 4.5)}/5.0 | *Duration*: ~{attr.get('visit_duration_hours', 2)} hrs | *Entry*: {fee}\n")
                
                # Check if user also asked about nightlife / food in multi-intent
                if "food_recommendation" in multi_intents:
                    food_part = get_food_guidance(active_dest, self.food_df)
                    lines.append("\n" + food_part)

                return "\n".join(lines), [active_dest]

        # 11. Multi-Intent Handling (e.g. Beaches + Food + Nightlife)
        if len(multi_intents) >= 2 and active_dest:
            combined_sections = [f"### 🌴 Comprehensive Guide for {active_dest.title()}\n"]
            if "attraction_recommendation" in multi_intents:
                attrs = self.semantic_engine.search_attractions("beach sightseeing attractions", destination=active_dest, top_k=3)
                if attrs:
                    combined_sections.append("#### 🏖️ Highlight Attractions:")
                    for a, _ in attrs:
                        combined_sections.append(f"* **{a['attraction_name']}**: {a.get('description', '')}")
                    combined_sections.append("")

            if "food_recommendation" in multi_intents:
                food_text = get_food_guidance(active_dest, self.food_df)
                combined_sections.append(food_text)

            if "activity_recommendation" in multi_intents:
                dest_row = self.destinations_df[self.destinations_df["destination"].str.lower() == active_dest.lower()]
                if not dest_row.empty:
                    acts = dest_row.iloc[0].get("activities", "")
                    combined_sections.append(f"#### 🎭 Activities & Nightlife:\n* {acts}\n")

            return "\n".join(combined_sections), [active_dest]

        # 12. General Destination Recommendation (The Core Open-Ended Engine)
        recs = self.recommender.recommend(
            query=query,
            entities=entities,
            preferences=preferences,
            top_n=3
        )

        if recs:
            recommended_dest_names = [r["destination"] for r in recs]
            intro_phrases = []
            if entities.get("budget_inr"):
                intro_phrases.append(f"budget of ₹{entities['budget_inr']:,}")
            if entities.get("duration_days"):
                intro_phrases.append(f"{entities['duration_days']}-day getaway")
            if entities.get("origin"):
                intro_phrases.append(f"from {entities['origin']}")
            if preferences:
                intro_phrases.append(f"focusing on {', '.join(p.title() for p in preferences[:2])}")
            if entities.get("crowd_preference") == "Low":
                intro_phrases.append("peaceful and uncrowded vibes")

            intro_str = f" based on your {', '.join(intro_phrases)}" if intro_phrases else ""
            lines = [f"### 🌟 Top Recommended Destinations{intro_str}\n"]

            for idx, r in enumerate(recs, 1):
                name = r["destination"]
                state = r.get("state", "India")
                cost = r.get("estimated_cost_per_day", 2000)
                rating = r.get("rating", 4.5)
                desc = r.get("description", "")
                reasons = r.get("reasons", [])

                lines.append(f"#### {idx}. **{name}**, {state}")
                lines.append(f"> {desc}\n")
                lines.append(f"* **⭐ Rating**: {rating}/5.0 | **💰 Est. Daily Budget**: approx. ₹{cost:,}/day per person")
                lines.append(f"* **⏱️ Ideal Duration**: {r.get('duration', '2-3 days')} | **👥 Crowd Level**: {r.get('crowd_level', 'Moderate')}")
                lines.append(f"* **🚶 Accessibility**: {r.get('walking_level', 'Moderate')} walking required")
                lines.append(f"* **🏷️ Famous For**: {r.get('famous_for', '')}")

                if reasons:
                    lines.append("\n**Why this matches your query:**")
                    for reason in reasons:
                        lines.append(f"  ✓ {reason}")
                lines.append("")

            # Actionable follow-up prompt
            first_dest = recs[0]["destination"]
            lines.append(
                f"💬 *Would you like me to generate a detailed day-by-day itinerary for **{first_dest}**, "
                f"break down the budget, or check travel routes from {entities.get('origin') or 'your city'}?*"
            )
            return "\n".join(lines), recommended_dest_names

        # 13. Layered Fallback (Level 4: General Travel Knowledge / Level 5: Honest Assistance)
        return (
            "I can help with destination recommendations, custom itineraries, budget estimation, "
            "tourist attractions, food guidance, and travel tips across India.\n\n"
            "I don't currently have enough verified information to answer that specific query accurately. "
            "Could you specify a destination name, your budget, or the kind of experience you are looking for?",
            []
        )
