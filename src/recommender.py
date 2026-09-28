"""
TravelMate — Dynamic Recommendation Engine
===========================================
Calculates multi-dimensional composite relevance scores across semantic similarity,
explicit user preferences, category alignment, budget compatibility, rating,
duration match, seasonal alignment, and physical accessibility.
Generates personalized, explainable justifications ('Why this was recommended').
"""

from typing import List, Dict, Any, Optional
import numpy as np


class TravelRecommender:
    """
    Ranks destination candidates using dynamic constraint-weighted scoring
    and produces explainable recommendation summaries.
    """

    def __init__(self, hybrid_retriever):
        self.retriever = hybrid_retriever

    def recommend(
        self,
        query: str,
        entities: Dict[str, Any],
        preferences: List[str],
        top_n: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval, dynamically re-scores candidates against all constraints,
        and attaches transparent explainability tags.
        """
        # Retrieve candidate pool
        candidates = self.retriever.hybrid_retrieve(
            query=query,
            entities=entities,
            preferences=preferences,
            top_k=15
        )
        if not candidates:
            return []

        # Score and explain all candidates
        scored_candidates = self._score_and_explain_candidates(candidates, entities, preferences)

        # If user explicitly specified one destination, prioritize it at the top
        named_dest = entities.get("destination")
        if named_dest:
            exact_matches = [
                c for c in scored_candidates
                if c["destination"].lower() == named_dest.lower()
            ]
            other_matches = [
                c for c in scored_candidates
                if c["destination"].lower() != named_dest.lower()
            ]
            if exact_matches:
                exact_matches[0]["composite_score"] = 1.0
                return (exact_matches + other_matches)[:top_n]

        return scored_candidates[:top_n]

    def _score_and_explain_candidates(
        self,
        candidates: List[Dict[str, Any]],
        entities: Dict[str, Any],
        preferences: List[str]
    ) -> List[Dict[str, Any]]:
        """Computes composite scores and explainability for all candidates."""
        budget = entities.get("budget_inr")
        duration = entities.get("duration_days")
        travel_group = entities.get("travel_group")
        walking_pref = entities.get("walking_preference")
        crowd_pref = entities.get("crowd_preference")
        season = entities.get("season")
        people = entities.get("people_count") or 1

        # Dynamic weights configuration based on active constraints
        w_sem = 0.30
        w_pref = 0.20
        w_cat = 0.10
        w_bud = 0.15 if budget else 0.05
        w_rat = 0.10
        w_dur = 0.10 if duration else 0.05
        w_sea = 0.10 if season else 0.05
        w_acc = 0.15 if (walking_pref or travel_group == "Parents / Family") else 0.05

        total_weight = w_sem + w_pref + w_cat + w_bud + w_rat + w_dur + w_sea + w_acc

        scored = []
        for cand in candidates:
            dest = cand["destination_data"]
            sem_score = cand["semantic_score"]

            # 1. Preference score
            pref_score = self._compute_preference_overlap(dest, preferences)

            # 2. Category score
            cat_score = self._compute_category_overlap(dest, preferences)

            # 3. Budget compatibility
            bud_score = self._compute_budget_compatibility(dest, budget, duration, people)

            # 4. Rating score (normalized 1.0 - 5.0 to 0.0 - 1.0)
            rat_score = (float(dest.get("rating", 4.0)) - 1.0) / 4.0

            # 5. Duration compatibility
            dur_score = self._compute_duration_compatibility(dest, duration)

            # 6. Season compatibility
            sea_score = self._compute_season_compatibility(dest, season)

            # 7. Accessibility compatibility (walking & crowd)
            acc_score = self._compute_accessibility_score(dest, walking_pref, crowd_pref)

            # Composite weighted calculation
            composite_score = (
                (w_sem * sem_score) +
                (w_pref * pref_score) +
                (w_cat * cat_score) +
                (w_bud * bud_score) +
                (w_rat * rat_score) +
                (w_dur * dur_score) +
                (w_sea * sea_score) +
                (w_acc * acc_score)
            ) / total_weight

            reasons = self._generate_explanations(
                dest, entities, preferences,
                bud_score=bud_score, dur_score=dur_score,
                acc_score=acc_score, pref_score=pref_score
            )

            result_item = {
                "destination": dest["destination"],
                "city": dest.get("city", dest["destination"]),
                "state": dest.get("state", ""),
                "description": dest.get("description", ""),
                "category": dest.get("category", ""),
                "rating": dest.get("rating", 4.5),
                "estimated_cost_per_day": dest.get("estimated_cost_per_day", 2000),
                "duration": dest.get("duration", "2-3 days"),
                "best_season": dest.get("best_season", ""),
                "crowd_level": dest.get("crowd_level", "Moderate"),
                "walking_level": dest.get("walking_level", "Moderate"),
                "famous_for": dest.get("famous_for", ""),
                "activities": dest.get("activities", ""),
                "attractions": dest.get("attractions", ""),
                "composite_score": round(float(composite_score), 4),
                "reasons": reasons,
                "raw_data": dest
            }
            scored.append(result_item)

        # Rank descending by composite score
        scored.sort(key=lambda s: s["composite_score"], reverse=True)
        return scored

    def _compute_preference_overlap(self, dest: Dict[str, Any], preferences: List[str]) -> float:
        """Calculates overlap between user preferences and destination tags."""
        if not preferences:
            return 0.5
        tags = (
            str(dest.get("suitable_for", "")).lower() + " " +
            str(dest.get("famous_for", "")).lower() + " " +
            str(dest.get("activities", "")).lower()
        )
        matches = sum(1 for p in preferences if p.lower() in tags)
        return min(matches / len(preferences), 1.0)

    def _compute_category_overlap(self, dest: Dict[str, Any], preferences: List[str]) -> float:
        """Calculates overlap between user preferences and category string."""
        if not preferences:
            return 0.5
        cat = str(dest.get("category", "")).lower()
        matches = sum(1 for p in preferences if p.lower() in cat)
        return min(matches / max(len(preferences), 1), 1.0)

    def _compute_budget_compatibility(
        self, dest: Dict[str, Any], budget: Optional[int], duration: Optional[int], people: int
    ) -> float:
        """Evaluates whether destination costs fit user budget."""
        if not budget or budget <= 0:
            return 0.8
        dur = duration if duration else 2
        daily_cost = dest.get("estimated_cost_per_day", 2000)
        total_est = daily_cost * dur * max(people, 1)

        ratio = total_est / budget
        if ratio <= 0.90:
            return 1.0       # Comfortably within budget
        elif ratio <= 1.15:
            return 0.85      # Fits budget closely
        elif ratio <= 1.40:
            return 0.50      # Slightly over budget
        else:
            return max(0.1, 1.0 - (ratio - 1.0) * 0.5)

    def _compute_duration_compatibility(self, dest: Dict[str, Any], duration: Optional[int]) -> float:
        """Checks if destination duration suits user timeframe."""
        if not duration:
            return 0.8
        dur_str = str(dest.get("duration", "2-3 days")).lower()
        if "1-2" in dur_str and duration <= 2:
            return 1.0
        elif "2-3" in dur_str and 2 <= duration <= 3:
            return 1.0
        elif "3-4" in dur_str and 3 <= duration <= 5:
            return 1.0
        elif "5-7" in dur_str and duration >= 5:
            return 1.0
        return 0.65

    def _compute_season_compatibility(self, dest: Dict[str, Any], season: Optional[str]) -> float:
        """Scores match against destination's best visiting season."""
        if not season:
            return 0.8
        best_season = str(dest.get("best_season", "")).lower()
        if season.lower() in best_season:
            return 1.0
        return 0.5

    def _compute_accessibility_score(
        self, dest: Dict[str, Any], walking_pref: Optional[str], crowd_pref: Optional[str]
    ) -> float:
        """Scores physical accessibility and crowd density."""
        score = 0.7
        w_level = str(dest.get("walking_level", "Moderate")).lower()
        c_level = str(dest.get("crowd_level", "Moderate")).lower()

        if walking_pref == "Low":
            if w_level == "low":
                score += 0.25
            elif w_level == "moderate":
                score += 0.05
            elif w_level == "high":
                score -= 0.35

        if crowd_pref == "Low":
            if c_level == "low":
                score += 0.20
            elif c_level == "moderate":
                score += 0.05
            elif c_level == "high":
                score -= 0.20

        return max(min(score, 1.0), 0.1)

    def _generate_explanations(
        self,
        dest: Dict[str, Any],
        entities: Dict[str, Any],
        preferences: List[str],
        bud_score: float = 0.8,
        dur_score: float = 0.8,
        acc_score: float = 0.8,
        pref_score: float = 0.8
    ) -> List[str]:
        """Generates clear, human-understandable explanation bullet points."""
        reasons = []
        name = dest.get("destination", "This destination")
        daily_cost = dest.get("estimated_cost_per_day", 2000)
        budget = entities.get("budget_inr")
        duration = entities.get("duration_days")
        group = entities.get("travel_group")
        walking_pref = entities.get("walking_preference")
        crowd_pref = entities.get("crowd_preference")
        w_level = str(dest.get("walking_level", "Moderate")).capitalize()
        c_level = str(dest.get("crowd_level", "Moderate")).capitalize()

        # 1. Budget rationale
        if budget:
            if bud_score >= 0.8:
                reasons.append(f"Fits your ₹{budget:,} budget comfortably (approx. ₹{daily_cost:,}/day per person)")
            else:
                reasons.append(f"Slightly above standard budget (approx. ₹{daily_cost:,}/day) but offers great value")
        else:
            reasons.append(f"Estimated daily cost: ₹{daily_cost:,} per person")

        # 2. Duration / trip length rationale
        if duration:
            reasons.append(f"Ideal duration ({dest.get('duration', '2-3 days')}) matches your {duration}-day timeframe")
        else:
            reasons.append(f"Recommended duration: {dest.get('duration', '2-3 days')}")

        # 3. Accessibility / Parents rationale
        if walking_pref == "Low" or group == "Parents / Family":
            if w_level == "Low":
                reasons.append("Low walking & flat terrain — exceptionally comfortable for elderly parents")
            elif w_level == "Moderate":
                reasons.append("Moderate walking with easy vehicle and taxi accessibility")

        # 4. Crowd rationale
        if crowd_pref == "Low" or c_level == "Low":
            reasons.append("Peaceful, uncrowded atmosphere free from overwhelming tourist rush")

        # 5. Preference matches
        matched_prefs = [p for p in preferences if p.lower() in str(dest.get("suitable_for", "")).lower() or p.lower() in str(dest.get("famous_for", "")).lower()]
        if matched_prefs:
            top_prefs = ", ".join(p.title() for p in matched_prefs[:3])
            reasons.append(f"Matches your interests in {top_prefs}")
        else:
            reasons.append(f"Renowned for: {dest.get('famous_for', 'scenic spots and culture')}")

        # 6. Rating endorsement
        rating = dest.get("rating", 4.5)
        reasons.append(f"Highly rated by travelers ({rating}/5.0)")

        return reasons[:4]
