"""
TravelMate — Hybrid Travel Entity Extractor
============================================
Combines spaCy NER, regular expressions, and custom travel entity matching
against destination knowledge bases, complete with fuzzy spelling correction
(e.g., 'Mumbay' -> 'Mumbai', 'Mahabaleshwr' -> 'Mahabaleshwar').
"""

import os
import re
import difflib
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from src.preprocessing import clean_text, get_spacy_nlp

# Default Destination Catalog fallback if CSV not yet loaded
DEFAULT_DESTINATIONS = [
    "Goa", "Jaipur", "Lonavala", "Matheran", "Mumbai", "Mahabaleshwar", "Gokarna",
    "Udaipur", "Munnar", "Hyderabad", "Manali", "Hampi", "Pondicherry", "Leh Ladakh",
    "Rishikesh", "Varanasi", "Agra", "Coorg", "Ooty", "Darjeeling", "Shillong",
    "Alleppey", "Amritsar", "Jaisalmer", "Andaman Islands", "Pune", "Alibaug",
    "Wayanad", "Kochi", "Dharamshala", "Nainital", "Gangtok", "Shimla",
    "Kanyakumari", "Mysore", "Shirdi"
]

DEFAULT_STATES = [
    "Maharashtra", "Goa", "Rajasthan", "Karnataka", "Kerala", "Telangana",
    "Himachal Pradesh", "Uttarakhand", "Uttar Pradesh", "Tamil Nadu", "West Bengal",
    "Meghalaya", "Punjab", "Ladakh", "Sikkim", "Puducherry", "Andaman and Nicobar Islands"
]

DEFAULT_ORIGINS = [
    "Mumbai", "Pune", "Delhi", "Bangalore", "Bengaluru", "Hyderabad",
    "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Surat", "Goa"
]


class TravelEntityExtractor:
    """
    Extracts structured travel entities from natural language queries.
    """

    def __init__(self, destinations_df: Optional[pd.DataFrame] = None):
        self.destinations = set(DEFAULT_DESTINATIONS)
        self.states = set(DEFAULT_STATES)
        self.cities = set(DEFAULT_ORIGINS)
        
        if destinations_df is not None and not destinations_df.empty:
            self._update_from_df(destinations_df)
        else:
            self._try_load_from_csv()

    def _update_from_df(self, df: pd.DataFrame):
        """Populates known entity sets from processed DataFrame."""
        if "destination" in df.columns:
            self.destinations.update(df["destination"].dropna().unique())
        if "state" in df.columns:
            self.states.update(df["state"].dropna().unique())
        if "city" in df.columns:
            self.cities.update(df["city"].dropna().unique())

    def _try_load_from_csv(self):
        """Attempts to load processed destination list from disk."""
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "destinations.csv"
        )
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                self._update_from_df(df)
            except Exception:
                pass

    def fuzzy_match_destination(self, token: str, threshold: float = 0.80) -> Optional[str]:
        """
        Fuzzy matches a candidate word against known destinations.
        Handles misspellings like 'Mumbay' -> 'Mumbai', 'Mahabaleshwr' -> 'Mahabaleshwar'.
        """
        if not token or len(token) < 4:
            return None
        token_clean = token.strip().title()
        
        # Exact match
        for dest in self.destinations:
            if token_clean.lower() == dest.lower():
                return dest
                
        # Substring / multi-word match (e.g. 'Ladakh' -> 'Leh Ladakh', 'Andaman' -> 'Andaman Islands')
        for dest in self.destinations:
            if token_clean.lower() in dest.lower() or dest.lower() in token_clean.lower():
                if len(token_clean) >= 4:
                    return dest

        # Difflib close match
        matches = difflib.get_close_matches(token_clean, list(self.destinations), n=1, cutoff=threshold)
        if matches:
            return matches[0]
            
        return None

    def extract_budget(self, text: str) -> Optional[int]:
        """
        Extracts budget figure in INR using multiple regex patterns.
        Handles ₹8000, Rs. 15000, 10k, 8000 rupees, under 7000, etc.
        """
        cleaned = text.lower()
        
        # 1. Matches with currency symbol: ₹8000, ₹ 12,000, rs 5000, inr 10000
        pattern_curr = r"(?:₹|rs\.?|inr|rupees?)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|lakh)?"
        m = re.search(pattern_curr, cleaned)
        if m:
            val_str = m.group(1).replace(",", "")
            multiplier = m.group(2)
            try:
                val = float(val_str)
                if multiplier in ("k", "thousand"):
                    val *= 1000
                elif multiplier == "lakh":
                    val *= 100000
                return int(val)
            except ValueError:
                pass

        # 2. Matches with trailing currency: 8000 rs, 15000 rupees, 10k budget
        pattern_trail = r"\b(\d+(?:,\d+)*)\s*(k)?\s*(?:rs|rupees|inr|bucks)\b"
        m = re.search(pattern_trail, cleaned)
        if m:
            val_str = m.group(1).replace(",", "")
            is_k = m.group(2)
            try:
                val = float(val_str)
                if is_k:
                    val *= 1000
                return int(val)
            except ValueError:
                pass

        # 3. Matches with constraint keywords: 'budget of 10000', 'under 8000', 'around 7000'
        pattern_kw = r"\b(?:budget(?:\s*of|\s*is|\s*around)?|under|within|around|spend(?:\s*more\s*than)?|not\s*more\s*than)\s*(?:₹|rs\.?)?\s*(\d+(?:,\d+)*)\s*(k)?\b"
        m = re.search(pattern_kw, cleaned)
        if m:
            val_str = m.group(1).replace(",", "")
            is_k = m.group(2)
            try:
                val = float(val_str)
                if is_k:
                    val *= 1000
                # Filter out obvious duration numbers (e.g. under 3 days)
                if val >= 100:
                    return int(val)
            except ValueError:
                pass

        # 4. Matches shorthand 8k / 10k budget
        pattern_k = r"\b(\d+)\s*k\b"
        m = re.search(pattern_k, cleaned)
        if m:
            try:
                return int(m.group(1)) * 1000
            except ValueError:
                pass

        return None

    def extract_duration(self, text: str) -> Optional[int]:
        """
        Extracts trip duration in days.
        Handles '3 days', '4 nights', 'weekend', '2-day trip', 'a week'.
        """
        cleaned = text.lower()
        
        # Weekend = 2 days
        if "weekend" in cleaned or "weekend trip" in cleaned or "weekend getaway" in cleaned:
            return 2
            
        # 'a week' / 'one week' = 7 days
        if re.search(r"\b(?:a|one|1)\s*week\b", cleaned):
            return 7

        # 'a day' / 'one day' = 1 day
        if re.search(r"\b(?:a|one|1)\s*day\b", cleaned):
            return 1

        # Digit + days/nights: '3 days', '4 nights', '2 days trip'
        pattern_days = r"\b(\d+)\s*(?:days?|nights?|d\b)"
        m = re.search(pattern_days, cleaned)
        if m:
            try:
                days = int(m.group(1))
                return min(max(days, 1), 30)  # Bound to reasonable trip range
            except ValueError:
                pass

        # Word numbers: 'two days', 'three days', 'four days', 'five days'
        word_num_map = {
            "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for word, num in word_num_map.items():
            if re.search(rf"\b{word}\s*(?:days?|nights?)\b", cleaned):
                return num

        return None

    def extract_origin(self, text: str) -> Optional[str]:
        """
        Extracts departure city or origin hub.
        Handles 'from Mumbai', 'leaving from Pune', 'starting out of Delhi'.
        """
        cleaned = text.strip()
        
        # Regex pattern for origin indicator: stop at boundary words, punctuation, or end of string
        pattern = r"\b(?:from|leaving\s+from|starting\s+from|out\s+of)\s+([A-Za-z\s]+?)(?=\s+(?:for|to|with|and|in|within|on|at|i\b)|$|,|\.|\?)"
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Check against known cities or fuzzy match
            fuzzy_match = self.fuzzy_match_destination(candidate, threshold=0.75)
            if fuzzy_match:
                return fuzzy_match
            for city in self.cities:
                if candidate.lower() == city.lower():
                    return city
            if len(candidate) > 2 and len(candidate.split()) <= 2:
                return candidate.title()
                
        return None

    def extract_travel_group(self, text: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Extracts travel party classification (Solo, Friends, Family, Parents, Couple)
        and approximate number of people.
        """
        cleaned = text.lower()
        group_type = None
        people_count = None

        # 1. Travel Group Type
        if any(w in cleaned for w in ["parent", "parents", "elderly", "mom and dad", "father", "mother"]):
            group_type = "Parents / Family"
        elif any(w in cleaned for w in ["family", "kids", "children"]):
            group_type = "Family"
        elif any(w in cleaned for w in ["friend", "friends", "buddies", "colleagues", "gang"]):
            group_type = "Friends"
        elif any(w in cleaned for w in ["couple", "partner", "wife", "husband", "girlfriend", "boyfriend", "romantic"]):
            group_type = "Couple"
        elif any(w in cleaned for w in ["solo", "alone", "myself", "by myself"]):
            group_type = "Solo"
            people_count = 1

        # 2. People Count
        m_count = re.search(r"\b(\d+)\s*(?:people|persons?|friends?|adults?|members?|travelers?)\b", cleaned)
        if m_count:
            try:
                people_count = int(m_count.group(1))
                if group_type == "Friends":
                    # E.g., 'with 3 friends' = 3 + 1 (the user) = 4
                    if "with" in cleaned:
                        people_count += 1
            except ValueError:
                pass
        elif re.search(r"\bwith\s*(?:my\s*)?(?:friend|partner|wife|husband|girlfriend|boyfriend)\b", cleaned):
            people_count = 2
            if not group_type:
                group_type = "Couple"

        return group_type, people_count

    def extract_season(self, text: str) -> Optional[str]:
        """Extracts season or temporal climate reference."""
        cleaned = text.lower()
        if "monsoon" in cleaned or "rain" in cleaned or "rainy" in cleaned:
            return "Monsoon"
        elif "winter" in cleaned or "cold" in cleaned or "snow" in cleaned:
            return "Winter"
        elif "summer" in cleaned or "hot" in cleaned:
            return "Summer"
        elif "spring" in cleaned:
            return "Spring"
        elif "autumn" in cleaned:
            return "Autumn"
        
        # Check specific months
        months = {
            "january": "Winter", "february": "Winter", "march": "Spring",
            "april": "Summer", "may": "Summer", "june": "Monsoon",
            "july": "Monsoon", "august": "Monsoon", "september": "Monsoon",
            "october": "Autumn", "november": "Winter", "december": "Winter"
        }
        for month, ssn in months.items():
            if re.search(rf"\b{month}\b", cleaned):
                return ssn
                
        return None

    def extract_constraints(self, text: str) -> Dict[str, Any]:
        """
        Extracts implicit travel constraints:
        - crowd_preference: 'Low' (peaceful, uncrowded, offbeat)
        - walking_preference: 'Low' (elderly, parents can't walk, less physical exertion)
        """
        cleaned = text.lower()
        constraints = {}

        # Crowd constraint
        crowd_low_keywords = [
            "peaceful", "quiet", "crowd", "crowded", "escape the crowds",
            "away from crowds", "not crowded", "isn't too crowded", "less crowd",
            "untouristy", "offbeat", "secluded", "relaxing", "unwind", "calm"
        ]
        if any(kw in cleaned for kw in crowd_low_keywords):
            constraints["crowd_preference"] = "Low"

        # Walking / physical constraint
        walking_low_keywords = [
            "can't walk", "cannot walk", "not walk too much", "less walking",
            "low walking", "elderly", "senior", "parents", "wheelchair",
            "easy walking", "not strenuous", "gentle"
        ]
        if any(kw in cleaned for kw in walking_low_keywords):
            constraints["walking_preference"] = "Low"

        return constraints

    def extract_destinations(self, text: str, origin: Optional[str] = None) -> List[str]:
        """
        Extracts all destination mentions in the query, handling multi-destination
        comparisons ('Which is better, Lonavala or Matheran?'), fuzzy matching,
        and distinguishing destination from origin.
        """
        cleaned = text.strip()
        extracted_dests = []
        
        # 1. Exact match pass against all known destinations
        for dest in self.destinations:
            # Word boundary regex for exact matching
            pattern = rf"\b{re.escape(dest)}\b"
            if re.search(pattern, cleaned, re.IGNORECASE):
                # Don't add if it's the specified origin
                if origin and dest.lower() == origin.lower():
                    continue
                if dest not in extracted_dests:
                    extracted_dests.append(dest)

        # 2. Check for state names (e.g., 'visit Maharashtra', 'trip to Rajasthan')
        for state in self.states:
            pattern = rf"\b{re.escape(state)}\b"
            if re.search(pattern, cleaned, re.IGNORECASE):
                if state not in extracted_dests:
                    extracted_dests.append(state)

        # 3. Fuzzy matching pass on individual words if no destination found yet
        if not extracted_dests:
            # Tokenize words
            words = re.findall(r"\b[A-Za-z]{4,}\b", cleaned)
            for w in words:
                # Skip common function words
                if w.lower() in {"want", "visit", "trip", "days", "budget", "somewhere", "place", "travel", "travelling"}:
                    continue
                match = self.fuzzy_match_destination(w, threshold=0.82)
                if match and match not in extracted_dests:
                    if origin and match.lower() == origin.lower():
                        continue
                    extracted_dests.append(match)

        # 4. SpaCy NER pass as a complementary check
        nlp = get_spacy_nlp()
        if nlp is not None:
            doc = nlp(cleaned)
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC"):
                    ent_text = ent.text.strip()
                    # Check if ent_text is not origin and matches a destination
                    if origin and ent_text.lower() == origin.lower():
                        continue
                    fuzzy_dest = self.fuzzy_match_destination(ent_text, threshold=0.80)
                    if fuzzy_dest and fuzzy_dest not in extracted_dests:
                        extracted_dests.append(fuzzy_dest)

        return extracted_dests

    def extract_all_entities(self, query: str) -> Dict[str, Any]:
        """
        Runs full hybrid entity extraction pipeline on query.
        """
        cleaned = clean_text(query)
        origin = self.extract_origin(cleaned)
        destinations = self.extract_destinations(cleaned, origin=origin)
        budget = self.extract_budget(cleaned)
        duration = self.extract_duration(cleaned)
        group_type, people_count = self.extract_travel_group(cleaned)
        season = self.extract_season(cleaned)
        constraints = self.extract_constraints(cleaned)

        # Check for destination switching and negation ('Actually, forget Goa. What about Jaipur?')
        negated_dests = []
        target_dests = []
        for d in destinations:
            neg_pat = rf"\b(forget|cancel|not|skip|instead of|leaving out|rather than)\s+{re.escape(d)}\b"
            if re.search(neg_pat, cleaned, re.IGNORECASE):
                negated_dests.append(d)
            else:
                target_dests.append(d)

        if target_dests:
            primary_destination = target_dests[0]
            valid_destinations = target_dests
        else:
            primary_destination = destinations[0] if destinations else None
            valid_destinations = destinations

        is_destination_switch = bool(negated_dests and target_dests) or (
            bool(target_dests) and any(kw in cleaned.lower() for kw in ["instead", "what about", "how about", "change to", "switch to"])
        )

        is_comparison = (len(valid_destinations) >= 2 or ("better" in cleaned.lower() and len(valid_destinations) >= 1)) and not is_destination_switch

        entities = {
            "destination": primary_destination,
            "all_destinations": valid_destinations,
            "negated_destination": negated_dests[0] if negated_dests else None,
            "is_destination_switch": is_destination_switch,
            "is_comparison": is_comparison,
            "origin": origin,
            "duration_days": duration,
            "budget_inr": budget,
            "travel_group": group_type,
            "people_count": people_count,
            "season": season,
            "crowd_preference": constraints.get("crowd_preference"),
            "walking_preference": constraints.get("walking_preference"),
            "has_constraints": bool(constraints or budget or duration or group_type)
        }
        return entities
