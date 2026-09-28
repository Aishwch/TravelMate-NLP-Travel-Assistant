"""
TravelMate — Conversational Context & Session Memory
=====================================================
Maintains multi-turn conversational memory, tracks active entities
(destination, duration, budget, preferences, party size), resolves pronouns
('there', 'it', 'that place'), and supports natural follow-up queries.
"""

from typing import Dict, List, Any, Optional
import re


class ConversationContext:
    """
    Manages session context state across turns.
    """

    def __init__(self):
        self.active_destination: Optional[str] = None
        self.last_recommended: List[str] = []
        self.origin: Optional[str] = None
        self.duration_days: Optional[int] = None
        self.budget_inr: Optional[int] = None
        self.travel_group: Optional[str] = None
        self.people_count: Optional[int] = None
        self.preferences: List[str] = []
        self.last_intent: Optional[str] = None
        self.turn_count: int = 0
        self.history: List[Dict[str, Any]] = []

    def update(
        self,
        query: str,
        entities: Dict[str, Any],
        intent: str,
        preferences: List[str],
        recommended_destinations: Optional[List[str]] = None
    ):
        """
        Updates session context with latest turn information.
        """
        self.turn_count += 1
        self.last_intent = intent

        # Handle explicit cancellation of active destination
        if entities.get("negated_destination"):
            if self.active_destination and entities["negated_destination"].lower() == self.active_destination.lower():
                self.active_destination = None

        # Update destination if newly provided
        if entities.get("destination"):
            self.active_destination = entities["destination"]
        elif recommended_destinations and len(recommended_destinations) > 0 and not self.active_destination:
            self.active_destination = recommended_destinations[0]

        # Update last recommended destinations
        if recommended_destinations:
            self.last_recommended = recommended_destinations

        # Update duration if specified
        if entities.get("duration_days"):
            self.duration_days = entities["duration_days"]

        # Update budget if specified
        if entities.get("budget_inr"):
            self.budget_inr = entities["budget_inr"]

        # Update origin if specified
        if entities.get("origin"):
            self.origin = entities["origin"]

        # Update travel group if specified
        if entities.get("travel_group"):
            self.travel_group = entities["travel_group"]

        # Update people count if specified
        if entities.get("people_count"):
            self.people_count = entities["people_count"]

        # Accumulate preferences without duplicates
        for p in preferences:
            if p not in self.preferences:
                self.preferences.append(p)

    def resolve_context(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves implicit pronouns ('there', 'that place', 'it') and missing entities
        using the active conversation memory.
        """
        resolved = dict(entities)
        q_lower = query.lower()

        # Pronoun resolution: 'there', 'that place', 'it', 'this place'
        has_pronoun = bool(
            re.search(r"\b(there|that place|this place|in it|about it)\b", q_lower)
            or q_lower.startswith(("what can i do", "what should i eat", "how can i reach", "can i do it", "how much will it cost"))
        )

        # 1. Resolve Destination from context
        if not resolved.get("destination"):
            if has_pronoun or not resolved.get("destination"):
                if self.active_destination:
                    resolved["destination"] = self.active_destination
                elif self.last_recommended:
                    resolved["destination"] = self.last_recommended[0]

        # 2. Resolve Duration from context if not in current query
        if not resolved.get("duration_days") and self.duration_days:
            resolved["duration_days"] = self.duration_days

        # 3. Resolve Budget from context if not in current query
        if not resolved.get("budget_inr") and self.budget_inr:
            resolved["budget_inr"] = self.budget_inr

        # 4. Resolve Origin from context if not in current query
        if not resolved.get("origin") and self.origin:
            resolved["origin"] = self.origin

        # 5. Resolve Travel Group from context
        if not resolved.get("travel_group") and self.travel_group:
            resolved["travel_group"] = self.travel_group

        # 6. Resolve People Count
        if not resolved.get("people_count") and self.people_count:
            resolved["people_count"] = self.people_count

        # 7. Comparison resolution: 'which is better?' when no destinations in current query
        if resolved.get("is_comparison") and not resolved.get("all_destinations"):
            if len(self.last_recommended) >= 2:
                resolved["all_destinations"] = self.last_recommended[:2]

        return resolved

    def add_history(self, user_msg: str, bot_response: str, intent: str):
        """Records turn history."""
        self.history.append({
            "turn": self.turn_count,
            "user": user_msg,
            "assistant": bot_response,
            "intent": intent
        })

    def get_summary(self) -> Dict[str, Any]:
        """Returns clean snapshot of active context state."""
        return {
            "active_destination": self.active_destination,
            "last_recommended": self.last_recommended,
            "origin": self.origin,
            "duration_days": self.duration_days,
            "budget_inr": self.budget_inr,
            "travel_group": self.travel_group,
            "people_count": self.people_count,
            "preferences": self.preferences,
            "turn_count": self.turn_count
        }

    def reset(self):
        """Clears all session memory."""
        self.active_destination = None
        self.last_recommended = []
        self.origin = None
        self.duration_days = None
        self.budget_inr = None
        self.travel_group = None
        self.people_count = None
        self.preferences = []
        self.last_intent = None
        self.turn_count = 0
        self.history = []
