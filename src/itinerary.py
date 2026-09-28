"""
TravelMate — Dynamic Itinerary Generator
=========================================
Generates structured, realistic day-by-day travel itineraries from the attractions
and culinary datasets. Organizes daily schedules (Morning, Afternoon, Evening, Dinner)
based on attraction ratings, visit duration constraints, spatial coordinates,
and user preferences, ensuring travelers are not overloaded.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class ItineraryGenerator:
    """
    Constructs multi-day itineraries dynamically from attractions and food datasets.
    """

    def __init__(self, attractions_df: pd.DataFrame, food_df: pd.DataFrame):
        self.attractions_df = attractions_df.copy()
        self.food_df = food_df.copy()

    def generate_itinerary(
        self,
        destination: str,
        days: int = 3,
        preferences: Optional[List[str]] = None,
        travel_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a complete day-by-day schedule for a given destination and duration.
        """
        dest_clean = destination.strip()
        days = max(1, min(int(days), 7))  # Clamp between 1 and 7 days
        prefs = preferences or []

        # 1. Filter attractions for this destination (or state)
        dest_attrs = self._get_destination_attractions(dest_clean, prefs)
        dest_foods = self._get_destination_foods(dest_clean)

        if dest_attrs.empty:
            return {
                "destination": dest_clean,
                "days": days,
                "success": False,
                "message": f"No structured attraction records currently available for '{dest_clean}'.",
                "daily_plans": []
            }

        # 2. Score and sort attractions by rating and preference overlap
        scored_attrs = self._score_attractions(dest_attrs, prefs)

        # 3. Partition attractions across days (2-3 attractions per day, max 6 hours sightseeing)
        daily_plans = []
        attrs_per_day = min(3, max(2, len(scored_attrs) // days))
        used_indices = set()

        for day_num in range(1, days + 1):
            day_items = []
            current_day_hours = 0.0

            # Find matching attractions for this day
            for idx, attr in scored_attrs.iterrows():
                if idx in used_indices:
                    continue
                duration = float(attr.get("visit_duration_hours", 2.0))
                if current_day_hours + duration <= 7.0 and len(day_items) < 3:
                    day_items.append(attr.to_dict())
                    used_indices.add(idx)
                    current_day_hours += duration

            # Fallback if no unused attractions left
            if not day_items and not scored_attrs.empty:
                # Cycle from top attractions
                sample_attr = scored_attrs.iloc[(day_num - 1) % len(scored_attrs)].to_dict()
                day_items.append(sample_attr)

            # Assign schedule time slots (Morning, Afternoon, Evening)
            schedule = self._organize_day_slots(day_items, day_num, dest_foods)
            daily_plans.append(schedule)

        return {
            "destination": dest_clean,
            "days": days,
            "travel_group": travel_group or "Standard",
            "preferences": prefs,
            "success": True,
            "total_attractions_visited": len(used_indices),
            "daily_plans": daily_plans
        }

    def _get_destination_attractions(self, destination: str, prefs: List[str]) -> pd.DataFrame:
        """Retrieves attractions matching destination name or state."""
        df = self.attractions_df.copy()
        if df.empty:
            return df

        # Exact destination match
        mask = df["destination"].str.lower() == destination.lower()
        if mask.any():
            return df[mask]

        # Partial substring match
        mask_sub = df["destination"].str.lower().str.contains(destination.lower(), na=False)
        if mask_sub.any():
            return df[mask_sub]

        # State match (e.g., 'Rajasthan', 'Kerala', 'Maharashtra')
        mask_state = df["state"].str.lower().str.contains(destination.lower(), na=False)
        if mask_state.any():
            return df[mask_state]

        return pd.DataFrame()

    def _get_destination_foods(self, destination: str) -> List[Dict[str, Any]]:
        """Retrieves regional cuisine and dish highlights."""
        df = self.food_df.copy()
        if df.empty:
            return []

        mask = df["destination"].str.lower() == destination.lower()
        if mask.any():
            return df[mask].to_dict(orient="records")

        mask_state = df["state"].str.lower().str.contains(destination.lower(), na=False)
        if mask_state.any():
            return df[mask_state].to_dict(orient="records")

        return []

    def _score_attractions(self, df: pd.DataFrame, prefs: List[str]) -> pd.DataFrame:
        """Ranks attractions by visitor rating and preference keywords."""
        df = df.copy()
        scores = []
        for _, row in df.iterrows():
            rating = float(row.get("rating", 4.0))
            cat = str(row.get("category", "")).lower()
            desc = str(row.get("description", "")).lower()
            
            pref_boost = 0.0
            for p in prefs:
                if p.lower() in cat or p.lower() in desc:
                    pref_boost += 0.5
            
            score = rating + pref_boost
            scores.append(score)

        df["sort_score"] = scores
        # Proximity sort if coordinates present
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.sort_values(by=["sort_score", "latitude"], ascending=[False, True])
        else:
            df = df.sort_values(by="sort_score", ascending=False)
            
        return df

    def _organize_day_slots(
        self,
        attractions: List[Dict[str, Any]],
        day_num: int,
        foods: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Arranges daily attractions into Morning, Afternoon, Evening, and Dining."""
        morning_slot = None
        afternoon_slot = None
        evening_slot = None

        for attr in attractions:
            best_time = str(attr.get("best_time", "")).lower()
            if "morning" in best_time and not morning_slot:
                morning_slot = attr
            elif "sunset" in best_time or "evening" in best_time and not evening_slot:
                evening_slot = attr
            elif not afternoon_slot:
                afternoon_slot = attr
            elif not morning_slot:
                morning_slot = attr
            elif not evening_slot:
                evening_slot = attr

        # Fill any remaining slots from attractions list
        remaining = [a for a in attractions if a not in (morning_slot, afternoon_slot, evening_slot)]
        if not morning_slot and remaining:
            morning_slot = remaining.pop(0)
        if not afternoon_slot and remaining:
            afternoon_slot = remaining.pop(0)
        if not evening_slot and remaining:
            evening_slot = remaining.pop(0)

        # Assign food recommendation for the day
        lunch_food = foods[(day_num - 1) % len(foods)] if foods else None
        dinner_food = foods[day_num % len(foods)] if len(foods) > 1 else lunch_food

        # Day theme title
        top_name = (morning_slot or evening_slot or {}).get("attraction_name", "Local Exploration")
        day_title = f"Day {day_num}: Exploring {top_name} & Heritage"

        # Calculate day entry fees
        total_entry = sum(int(a.get("entry_fee", 0)) for a in [morning_slot, afternoon_slot, evening_slot] if a)

        return {
            "day": day_num,
            "title": day_title,
            "morning": morning_slot,
            "lunch_food": lunch_food,
            "afternoon": afternoon_slot,
            "evening": evening_slot,
            "dinner_food": dinner_food,
            "estimated_entry_fees": total_entry
        }

    def format_as_text(self, itinerary_data: Dict[str, Any]) -> str:
        """Converts structured itinerary JSON to readable markdown format."""
        if not itinerary_data.get("success"):
            return itinerary_data.get("message", "Unable to generate itinerary.")

        dest = itinerary_data["destination"]
        days = itinerary_data["days"]
        output = [f"### 📅 {days}-Day Customized Itinerary for {dest}\n"]

        for plan in itinerary_data["daily_plans"]:
            day_num = plan["day"]
            output.append(f"#### 🗓️ {plan['title']}")

            if plan["morning"]:
                m = plan["morning"]
                fee = f"₹{m.get('entry_fee', 0)}" if m.get("entry_fee") else "Free"
                output.append(f"* **🌅 Morning (09:00 AM - 01:00 PM)**: **{m['attraction_name']}** ({m.get('category', 'Sightseeing')})")
                output.append(f"  * *Overview*: {m.get('description', '')}")
                output.append(f"  * *Duration*: ~{m.get('visit_duration_hours', 2)} hrs | *Entry*: {fee}")

            if plan["lunch_food"]:
                f = plan["lunch_food"]
                dish_name = f.get("food_name", f.get("food_item", f.get("name", "Regional Dish")))
                spot = f.get("famous_at", f.get("famous_spots", "Local traditional eateries"))
                output.append(f"* **🍽️ Lunch Recommendation**: Try authentic **{dish_name}** ({f.get('type', 'Specialty')})")
                output.append(f"  * *Where*: {spot} | *Course*: {f.get('course', 'Main Course')}")

            if plan["afternoon"]:
                a = plan["afternoon"]
                fee = f"₹{a.get('entry_fee', 0)}" if a.get("entry_fee") else "Free"
                output.append(f"* **☀️ Afternoon (02:00 PM - 05:00 PM)**: **{a['attraction_name']}**")
                output.append(f"  * *Overview*: {a.get('description', '')}")
                output.append(f"  * *Duration*: ~{a.get('visit_duration_hours', 2)} hrs | *Entry*: {fee}")

            if plan["evening"]:
                e = plan["evening"]
                fee = f"₹{e.get('entry_fee', 0)}" if e.get("entry_fee") else "Free"
                output.append(f"* **🌆 Evening & Sunset (05:30 PM - 07:30 PM)**: **{e['attraction_name']}**")
                output.append(f"  * *Overview*: {e.get('description', '')}")
                output.append(f"  * *Best Time*: Golden hour sunset | *Entry*: {fee}")

            if plan["dinner_food"]:
                df = plan["dinner_food"]
                d_dish = df.get("food_name", df.get("food_item", df.get("name", "Local Cuisine")))
                output.append(f"* **🌙 Dinner & Evening Leisure**: Savor **{d_dish}** — *{df.get('description', '')}*\n")

        return "\n".join(output)
