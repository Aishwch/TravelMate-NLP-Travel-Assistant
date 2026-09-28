"""
TravelMate — Utility Functions & Domain Specialists
===================================================
Specialized assistance modules for:
  - Dynamic Destination Comparison
  - Regional & Dietary Food Assistance
  - Multi-Modal Transit Guidance
  - Contextual Packing Checklists & Safety Precautions
"""

from typing import Dict, List, Any, Optional
import pandas as pd


def compare_destinations(
    dest1_name: str,
    dest2_name: str,
    destinations_df: pd.DataFrame,
    budget_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Performs dynamic multi-attribute comparison between two destinations:
    costs, ratings, crowd levels, walking accessibility, activities, and vibe.
    """
    if destinations_df.empty:
        return "Destination dataset is not available for comparison."

    def find_row(name):
        mask = destinations_df["destination"].str.lower() == name.strip().lower()
        if mask.any():
            return destinations_df[mask].iloc[0].to_dict()
        mask_sub = destinations_df["destination"].str.lower().str.contains(name.strip().lower(), na=False)
        if mask_sub.any():
            return destinations_df[mask_sub].iloc[0].to_dict()
        return None

    row1 = find_row(dest1_name)
    row2 = find_row(dest2_name)

    if not row1 or not row2:
        missing = []
        if not row1:
            missing.append(dest1_name)
        if not row2:
            missing.append(dest2_name)
        return (
            f"I could not locate detailed records for: {', '.join(missing)}. "
            f"Please verify destination names."
        )

    d1, d2 = row1["destination"], row2["destination"]
    r1, r2 = row1.get("rating", 4.5), row2.get("rating", 4.5)
    c1, c2 = row1.get("estimated_cost_per_day", 2000), row2.get("estimated_cost_per_day", 2000)
    dur1, dur2 = row1.get("duration", "2-3 days"), row2.get("duration", "2-3 days")
    crowd1, crowd2 = row1.get("crowd_level", "Moderate"), row2.get("crowd_level", "Moderate")
    walk1, walk2 = row1.get("walking_level", "Moderate"), row2.get("walking_level", "Moderate")
    season1, season2 = row1.get("best_season", "Winter"), row2.get("best_season", "Winter")
    vibe1, vibe2 = row1.get("category", "Tourism"), row2.get("category", "Tourism")

    lines = [
        f"### ⚖️ Destination Comparison: {d1} vs {d2}\n",
        f"| Feature / Parameter | **{d1}** | **{d2}** |",
        f"| :--- | :--- | :--- |",
        f"| **⭐ Traveler Rating** | {r1}/5.0 | {r2}/5.0 |",
        f"| **💰 Est. Daily Budget** | approx. ₹{c1:,}/day | approx. ₹{c2:,}/day |",
        f"| **⏱️ Recommended Duration** | {dur1} | {dur2} |",
        f"| **👥 Tourist Crowd Density** | {crowd1} | {crowd2} |",
        f"| **🚶 Walking Exertion** | {walk1} | {walk2} |",
        f"| **🌦️ Best Season** | {season1} | {season2} |",
        f"| **🏷️ Primary Vibe** | {vibe1} | {vibe2} |\n",
        f"#### 🔍 Key Takeaways & Decision Guidance:",
        f"* **Cost Comparison**: {'**' + d1 + '** is more budget-friendly' if c1 < c2 else ('**' + d2 + '** is more budget-friendly' if c2 < c1 else 'Both destinations share a similar daily budget tier')}."
    ]

    # Walking comparison
    if walk1 == "Low" and walk2 != "Low":
        lines.append(f"* **Accessibility**: **{d1}** involves significantly less walking, making it superior for travelling with elderly parents.")
    elif walk2 == "Low" and walk1 != "Low":
        lines.append(f"* **Accessibility**: **{d2}** has lower walking requirements and is easier for seniors.")

    # Crowd comparison
    if crowd1 == "Low" and crowd2 != "Low":
        lines.append(f"* **Peace & Serenity**: **{d1}** is noticeably less crowded and ideal for peaceful relaxation.")
    elif crowd2 == "Low" and crowd1 != "Low":
        lines.append(f"* **Peace & Serenity**: **{d2}** offers an uncrowded, serene environment away from tourist commercialization.")

    lines.append(f"* **Highlights of {d1}**: {row1.get('famous_for', '')}")
    lines.append(f"* **Highlights of {d2}**: {row2.get('famous_for', '')}\n")
    return "\n".join(lines)


def get_food_guidance(
    destination: str,
    food_df: pd.DataFrame,
    dietary_filter: Optional[str] = None
) -> str:
    """
    Retrieves authentic culinary specialties, famous spots, and dietary matches.
    """
    if food_df.empty:
        return "Food database currently unavailable."

    dest_clean = destination.strip()
    mask = food_df["destination"].str.lower() == dest_clean.lower()
    if not mask.any():
        mask = food_df["destination"].str.lower().str.contains(dest_clean.lower(), na=False)
    if not mask.any() and "state" in food_df.columns:
        mask = food_df["state"].str.lower().str.contains(dest_clean.lower(), na=False)

    matching = food_df[mask]
    if matching.empty:
        # Fallback to general regional Indian dishes if destination not found
        matching = food_df.head(4)

    # Dietary filtering
    if dietary_filter:
        diet_lower = dietary_filter.lower()
        if "non" in diet_lower:
            matching = matching[matching["type"].str.lower().str.contains("non", na=False)]
        elif "veg" in diet_lower:
            matching = matching[matching["type"].str.lower().str.contains("veg", na=False) & ~matching["type"].str.lower().str.contains("non", na=False)]

    lines = [f"### 🍽️ Culinary Guide for {dest_clean.title()}\n"]
    if dietary_filter:
        lines.append(f"*Filtered for: {dietary_filter.capitalize()} specialties*\n")

    for _, row in matching.head(5).iterrows():
        item = row.get("food_name", row.get("food_item", row.get("name", "Regional Dish")))
        cuisine = row.get("cuisine", "Traditional Indian Cuisine")
        desc = row.get("description", "")
        spots = row.get("famous_at", row.get("famous_spots", "Local traditional eateries"))
        ftype = str(row.get("type", "Vegetarian")).capitalize()

        lines.append(f"* **{item}** `[{ftype}]` — *{cuisine}*")
        if desc:
            lines.append(f"  * {desc}")
        lines.append(f"  * **Famous At**: {spots}\n")

    return "\n".join(lines)


def get_transport_guidance(
    origin: str,
    destination: str,
    transport_df: pd.DataFrame
) -> str:
    """
    Retrieves realistic route transit options, travel times, cost ranges, and routing tips.
    """
    if transport_df.empty:
        return "Transport connectivity database currently unavailable."

    orig_clean = origin.strip().lower()
    dest_clean = destination.strip().lower()

    # Search for direct route match
    mask = (
        (transport_df["origin"].str.lower() == orig_clean) &
        (transport_df["destination"].str.lower() == dest_clean)
    )
    if not mask.any():
        # Reverse or partial search
        mask = (
            transport_df["destination"].str.lower().str.contains(dest_clean, na=False) &
            transport_df["origin"].str.lower().str.contains(orig_clean, na=False)
        )

    matched = transport_df[mask]
    if matched.empty:
        # General route fallback advice
        return (
            f"### 🚗 Travel Route: {origin.title()} to {destination.title()}\n\n"
            f"* **Distance**: Approximately varies depending on selected highway/railway corridor.\n"
            f"* **Transport Modes**: Typically connected via State/Interstate AC Volvo buses, express trains, or flight to nearest airport.\n"
            f"* **Guidance**: For long-distance inter-state travel, booking express train tickets (Vande Bharat / Rajdhani / Superfast) in advance on IRCTC or direct flights offers the most seamless connectivity.\n\n"
            f"> [!NOTE]\n"
            f"> *Live transit timings fluctuate. Please verify real-time seat availability on official booking portals (IRCTC / RedBus / MakeMyTrip).* "
        )

    row = matched.iloc[0]
    lines = [
        f"### 🚆 How to Reach {row['destination']} from {row['origin']}\n",
        f"* **Distance**: Approx. **{row['distance_km']} km**",
        f"* **Available Modes**: {row['modes']}",
        f"* **Estimated Travel Times**: {row['travel_time']}",
        f"* **Indicative Fare Brackets**: {row['estimated_cost_range']}\n",
        f"#### 💡 Route & Transit Advice:",
        f"{row['general_route_info']}\n\n",
        f"> [!NOTE]\n",
        f"> *Transit fares and durations are representative historical estimates. Schedules are subject to weather conditions and rail updates.*"
    ]
    return "\n".join(lines)


def get_packing_guidance(
    query: str,
    destination: Optional[str],
    packing_df: pd.DataFrame
) -> str:
    """
    Retrieves tailored packing checklists and health & safety precautions.
    """
    if packing_df.empty:
        return "Packing tips database currently unavailable."

    q_lower = query.lower()
    category = "General Travel"

    if any(w in q_lower for w in ["monsoon", "rain", "rainy", "rains"]):
        category = "Monsoon Travel"
    elif any(w in q_lower for w in ["ladakh", "leh", "snow", "winter", "cold", "manali", "shimla"]):
        category = "Winter / Hill Station"
    elif any(w in q_lower for w in ["beach", "goa", "gokarna", "andaman", "coastal"]):
        category = "Beach / Coastal"
    elif any(w in q_lower for w in ["rajasthan", "desert", "jaisalmer", "jaipur"]):
        category = "Desert / Rajasthan"
    elif any(w in q_lower for w in ["solo", "alone", "safety"]):
        category = "Solo Travel Safety"
    elif any(w in q_lower for w in ["temple", "shrine", "varanasi", "spiritual", "shirdi"]):
        category = "Spiritual / Heritage"
    elif any(w in q_lower for w in ["trek", "trekking", "hike", "hiking", "adventure"]):
        category = "Trekking / Adventure"
    else:
        # Check destination type if given
        if destination:
            d_lower = destination.lower()
            if any(w in d_lower for w in ["ladakh", "manali", "shimla", "nainital", "darjeeling"]):
                category = "Winter / Hill Station"
            elif any(w in d_lower for w in ["goa", "gokarna", "andaman", "alibaug"]):
                category = "Beach / Coastal"
            elif any(w in d_lower for w in ["jaisalmer", "jaipur", "udaipur"]):
                category = "Desert / Rajasthan"

    mask = packing_df["category"].str.lower() == category.lower()
    if not mask.any():
        mask = packing_df["category"].str.contains(category.split()[0], case=False, na=False)

    row = packing_df[mask].iloc[0] if mask.any() else packing_df.iloc[0]

    lines = [
        f"### 🎒 Packing Checklist & Travel Advisory: {row['category']}\n",
        f"#### 🧳 Essential Items to Pack:",
        f"{row['packing_items']}\n",
        f"#### 👕 Recommended Wardrobe & Clothing:",
        f"{row['clothing']}\n",
        f"#### 🩹 Health, First-Aid & Safety Precautions:",
        f"{row['health_safety']}\n",
        f"#### 💡 Practical Travel Advice:",
        f"{row['travel_advice']}\n"
    ]
    return "\n".join(lines)
