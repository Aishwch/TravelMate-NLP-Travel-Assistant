"""
TravelMate — Semantic Travel Preference Extractor
==================================================
Identifies explicit and implicit traveler preferences from natural language
queries. Maps expressive and open-ended queries (e.g., 'disconnect from everything',
'take amazing photos without spending too much') to core travel attributes
(peaceful, photography, budget, nature).
"""

import re
from typing import List, Dict, Set, Tuple

# Comprehensive taxonomy of travel preference concepts
PREFERENCE_KEYWORDS: Dict[str, List[str]] = {
    "beach": [
        "beach", "beaches", "coast", "coastal", "sea", "ocean", "sand", "sandy",
        "shore", "shores", "waves", "shack", "shacks", "sunbathing", "coastal walk",
        "scuba", "snorkel", "water sports"
    ],
    "nature": [
        "nature", "greenery", "forest", "forests", "flora", "fauna", "green",
        "valley", "valleys", "trees", "natural", "eco", "wilderness", "lush",
        "botanical", "wooded", "scenic beauty"
    ],
    "waterfalls": [
        "waterfall", "waterfalls", "falls", "cascade", "cascades", "cataract",
        "stream", "streams", "torrent", "gushing water"
    ],
    "peaceful": [
        "peaceful", "peace", "calm", "relaxing", "relaxation", "quiet", "serene",
        "serenity", "tranquil", "tranquility", "disconnect", "unwind", "chill",
        "away from crowds", "escape the crowds", "escape crowds", "no crowd",
        "not crowded", "uncrowded", "silent", "solitude", "offbeat", "slow travel",
        "soothing", "stress free"
    ],
    "romantic": [
        "romantic", "romance", "couple", "couples", "honeymoon", "candlelight",
        "partner", "sunset", "love", "intimate", "cozy", "getaway for two"
    ],
    "photography": [
        "photography", "photo", "photos", "pictures", "photographer", "camera",
        "scenic", "viewpoint", "viewpoints", "breathtaking", "panoramic",
        "instagram", "instagrammable", "landscape", "landscapes", "sunset view",
        "scenery", "photogenic", "visuals"
    ],
    "adventure": [
        "adventure", "thrill", "thrilling", "exciting", "adrenaline", "rafting",
        "paragliding", "bungee", "scuba", "diving", "rock climbing", "bouldering",
        "zipline", "sports", "action"
    ],
    "trekking": [
        "trekking", "trek", "treks", "hike", "hiking", "hikes", "trails", "trail",
        "mountaineering", "climb", "climbing", "summit", "peak"
    ],
    "historical": [
        "historical", "history", "heritage", "monument", "monuments", "fort", "forts",
        "palace", "palaces", "ancient", "ruins", "unesco", "castle", "colonial",
        "architecture", "dynasty", "museum", "museums"
    ],
    "religious": [
        "religious", "spiritual", "temple", "temples", "shrine", "shrines",
        "pilgrimage", "ashram", "ashrams", "ghat", "ghats", "aarti", "samadhi",
        "darshan", "sacred", "holy", "mosque", "church", "monastery", "meditation"
    ],
    "food": [
        "food", "foodie", "cuisine", "eat", "dining", "street food", "dishes",
        "dish", "culinary", "delicacy", "delicacies", "taste", "tasting",
        "restaurant", "restaurants", "biryani", "thali", "cafe", "cafes",
        "local food", "vegetarian", "non-vegetarian", "breakfast"
    ],
    "nightlife": [
        "nightlife", "party", "parties", "club", "clubs", "pub", "pubs", "bars",
        "bar", "music", "dj", "night market", "evening life", "cocktails"
    ],
    "budget": [
        "budget", "cheap", "affordable", "economical", "inexpensive", "low cost",
        "save money", "shoestring", "pocket friendly", "without spending too much",
        "not spend much", "small budget", "backpacking", "hostel"
    ],
    "luxury": [
        "luxury", "luxurious", "expensive", "premium", "5 star", "five star",
        "resort", "resorts", "high end", "plush", "opulent", "exclusive", "royal stay"
    ],
    "family": [
        "family", "parents", "elderly", "kids", "children", "relatives", "mom and dad",
        "grandparents", "all ages", "safe for family"
    ],
    "solo": [
        "solo", "alone", "myself", "by myself", "individual", "solo trip", "solo traveler"
    ],
    "hill_station": [
        "hill station", "hill stations", "hills", "mountain", "mountains", "himalayas",
        "cool weather", "mist", "fog", "cloudy", "high altitude", "tea gardens"
    ],
    "weekend": [
        "weekend", "weekend trip", "weekend getaway", "short trip", "quick trip",
        "2 days trip", "two days", "quick getaway"
    ]
}


class PreferenceExtractor:
    """
    Extracts high-level and granular travel preferences from natural text.
    """

    def __init__(self):
        self.preference_map = PREFERENCE_KEYWORDS

    def extract_preferences(self, query: str) -> List[str]:
        """
        Scans normalized tokens and idioms to detect applicable travel preferences.
        Returns a sorted list of matched preference categories.
        """
        if not query or not isinstance(query, str):
            return []

        text = query.lower()
        matched_prefs: Dict[str, int] = {}

        # 1. Check idiomatic expressions
        idioms = [
            ("disconnect from everything", ["peaceful", "nature", "relaxation"]),
            ("away from the crowds", ["peaceful"]),
            ("escape the crowds", ["peaceful"]),
            ("without spending too much", ["budget"]),
            ("take amazing photos", ["photography", "scenic"]),
            ("breathtaking views", ["scenic", "photography"]),
            ("soak in the sun", ["beach", "relaxation"]),
            ("adrenaline rush", ["adventure"]),
            ("travelling with parents", ["family"]),
            ("can't walk too much", ["peaceful", "family"]),
            ("royal treatment", ["luxury", "historical"]),
            ("night out", ["nightlife"]),
            ("local food", ["food", "culture"]),
        ]
        for phrase, prefs in idioms:
            if phrase in text:
                for p in prefs:
                    matched_prefs[p] = matched_prefs.get(p, 0) + 3

        # 2. Check keyword lexicon
        for pref, keywords in self.preference_map.items():
            for kw in keywords:
                # Use word boundary regex for single words, direct substring for multi-word phrases
                if " " in kw:
                    if kw in text:
                        matched_prefs[pref] = matched_prefs.get(pref, 0) + 2
                else:
                    pattern = rf"\b{re.escape(kw)}\b"
                    if re.search(pattern, text):
                        matched_prefs[pref] = matched_prefs.get(pref, 0) + 1

        # Sort preferences by frequency / weight
        sorted_prefs = sorted(matched_prefs.keys(), key=lambda k: matched_prefs[k], reverse=True)
        return sorted_prefs

    def get_preference_description(self, pref: str) -> str:
        """Returns human-readable explanation of an extracted preference."""
        descriptions = {
            "beach": "Coastal vistas, sun, sand, and beachside relaxation",
            "nature": "Lush greenery, natural landscapes, and wildlife",
            "waterfalls": "Cascading mountain waterfalls and rivers",
            "peaceful": "Tranquil, quiet environments free from heavy tourist crowds",
            "romantic": "Intimate scenic ambiance ideal for couples and honeymooners",
            "photography": "Vivid viewpoints, architecture, and photogenic panoramas",
            "adventure": "Active thrills such as rafting, paragliding, and extreme sports",
            "trekking": "Hiking trails, mountain summits, and scenic walks",
            "historical": "Centuries-old forts, palaces, ruins, and UNESCO heritage",
            "religious": "Sacred temples, shrines, pilgrimage routes, and spiritual peace",
            "food": "Authentic regional cuisines, iconic street specialties, and culinary walks",
            "nightlife": "Vibrant evening entertainment, beach shacks, pubs, and live music",
            "budget": "Economical stays, transit, and affordable activities",
            "luxury": "High-end resorts, fine dining, and premium comfort",
            "family": "Accessible, safe, and comfortable for all generations",
            "solo": "Safe, social, and convenient for independent exploration",
            "hill_station": "Cool alpine climate, mist-covered valleys, and mountain views",
            "weekend": "Compact destinations ideal for 1 to 2 day quick getaways"
        }
        return descriptions.get(pref, "Travel preference")
