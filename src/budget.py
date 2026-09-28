"""
TravelMate — Intelligent Budget Planner
========================================
Parses natural language budget inputs and computes realistic cost breakdowns
(Accommodation, Meals, Local Transit, Activities, Buffer) using data/processed/budget.csv.
Assesses trip financial feasibility and suggests actionable money-saving adjustments.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd


class BudgetPlanner:
    """
    Computes itemized travel cost estimates and evaluates financial feasibility.
    """

    def __init__(self, budget_df: pd.DataFrame):
        self.budget_df = budget_df.copy()

    def plan_budget(
        self,
        destination: str,
        total_budget: Optional[int] = None,
        duration_days: int = 3,
        people_count: int = 1,
        tier: str = "midrange"
    ) -> Dict[str, Any]:
        """
        Calculates itemized budget breakdown and feasibility analysis.
        """
        dest_clean = destination.strip()
        days = max(1, duration_days or 2)
        people = max(1, people_count or 1)

        # Lookup destination in budget matrix
        row = self._find_budget_record(dest_clean)
        if row is None:
            # Fallback average figures
            row = {
                "destination": dest_clean,
                "accommodation_budget": 700,
                "accommodation_midrange": 1800,
                "accommodation_luxury": 5000,
                "food_cost_per_day": 550,
                "local_transport_cost_per_day": 400,
                "activity_cost_per_day": 350,
                "daily_budget_min": 1400,
                "daily_budget_avg": 2200,
                "budget_tier": "Moderate"
            }

        # Select accommodation unit cost based on tier or user's stated total budget
        base_acc = row.get("accommodation_cost_per_day", 1200)
        acc_budget = row.get("accommodation_budget", int(base_acc * 0.7))
        acc_mid = row.get("accommodation_midrange", base_acc)
        acc_lux = row.get("accommodation_luxury", int(base_acc * 2.2))

        if tier.lower() == "budget" or (total_budget and total_budget / (days * people) < 1800):
            stay_per_night = acc_budget
            chosen_tier = "Budget (Hostels / Guesthouses)"
        elif tier.lower() == "luxury" or (total_budget and total_budget / (days * people) > 4000):
            stay_per_night = acc_lux
            chosen_tier = "Luxury (4/5-Star Resorts)"
        else:
            stay_per_night = acc_mid
            chosen_tier = "Mid-range (3-Star Boutique Hotels)"

        # Number of hotel rooms needed (assume 2 adults per room)
        rooms_needed = max(1, (people + 1) // 2)
        total_stay = stay_per_night * (days - 1 if days > 1 else 1) * rooms_needed
        food_per_day = row.get("food_cost_per_day", 550)
        transit_per_day = row.get("local_transport_cost_per_day", 350)
        activity_per_day = row.get("activity_cost_per_day", row.get("sightseeing_cost_per_day", 250))
        total_food = food_per_day * days * people
        total_transit = transit_per_day * days * rooms_needed
        total_activity = activity_per_day * days * people
        subtotal = total_stay + total_food + total_transit + total_activity
        buffer_emergency = int(subtotal * 0.10)  # 10% contingency buffer
        total_estimated = subtotal + buffer_emergency

        # Feasibility check against user's stated budget
        feasibility_status = "Estimated"
        feasibility_message = ""

        if total_budget and total_budget > 0:
            diff = total_budget - total_estimated
            min_daily = row.get("daily_budget_min", acc_budget + int(food_per_day * 0.7) + int(transit_per_day * 0.5))
            if total_budget >= total_estimated:
                feasibility_status = "Comfortable"
                surplus = total_budget - total_estimated
                feasibility_message = (
                    f"✓ Your budget of ₹{total_budget:,} is sufficient and comfortable for {days} days in {dest_clean}. "
                    f"You have an estimated cushion of ₹{surplus:,}."
                )
            elif total_budget >= (min_daily * days * people):
                feasibility_status = "Shoestring / Manageable"
                feasibility_message = (
                    f"⚠️ Your budget of ₹{total_budget:,} is feasible on a budget/backpacker style. "
                    f"Opt for budget hostels, public transport, and street eateries to stay within limit."
                )
            else:
                feasibility_status = "Tight / Under-Budget"
                shortfall = total_estimated - total_budget
                feasibility_message = (
                    f"⚠️ Estimated minimum cost is approximately ₹{total_estimated:,}, which exceeds your ₹{total_budget:,} budget by ₹{shortfall:,}. "
                    f"Consider reducing duration to {max(1, days - 1)} days, sharing accommodation, or exploring closer destinations."
                )

        return {
            "destination": dest_clean,
            "duration_days": days,
            "people_count": people,
            "tier": chosen_tier,
            "user_budget": total_budget,
            "total_estimated": total_estimated,
            "breakdown": {
                "accommodation": total_stay,
                "food_dining": total_food,
                "local_transport": total_transit,
                "activities_sightseeing": total_activity,
                "contingency_buffer": buffer_emergency
            },
            "per_person_per_day": round(total_estimated / (days * people)),
            "feasibility_status": feasibility_status,
            "feasibility_message": feasibility_message,
            "is_estimate": True
        }

    def _find_budget_record(self, destination: str) -> Optional[Dict[str, Any]]:
        """Finds matching row in budget dataframe."""
        if self.budget_df.empty:
            return None
        mask = self.budget_df["destination"].str.lower() == destination.strip().lower()
        if mask.any():
            return self.budget_df[mask].iloc[0].to_dict()
        # Partial match
        mask_sub = self.budget_df["destination"].str.lower().str.contains(destination.strip().lower(), na=False)
        if mask_sub.any():
            return self.budget_df[mask_sub].iloc[0].to_dict()
        return None

    def format_as_text(self, plan: Dict[str, Any]) -> str:
        """Formats budget plan as markdown text."""
        dest = plan["destination"]
        days = plan["duration_days"]
        people = plan["people_count"]
        bd = plan["breakdown"]

        lines = [
            f"### 💰 Estimated Travel Budget for {dest} ({days} Days, {people} Person{'s' if people > 1 else ''})\n",
            f"*Tier: {plan['tier']}*\n",
            f"| Expenditure Category | Estimated Amount (INR) |",
            f"| :--- | :--- |",
            f"| **🏨 Accommodation** | ₹{bd['accommodation']:,} |",
            f"| **🍽️ Meals & Dining** | ₹{bd['food_dining']:,} |",
            f"| **🚖 Local Transit** | ₹{bd['local_transport']:,} |",
            f"| **🎟️ Activities & Entry** | ₹{bd['activities_sightseeing']:,} |",
            f"| **🛡️ Buffer & Miscellaneous (10%)** | ₹{bd['contingency_buffer']:,} |",
            f"| **💳 Total Estimated Cost** | **₹{plan['total_estimated']:,}** |\n",
            f"*Approx. ₹{plan['per_person_per_day']:,} per person per day.*\n"
        ]

        if plan.get("feasibility_message"):
            lines.append(f"> **Feasibility Assessment**: {plan['feasibility_message']}\n")

        lines.append(
            "> [!NOTE]\n"
            "> *All figures are realistic indicative estimates based on historical travel data. "
            "Actual tariffs may fluctuate based on peak holiday seasons, booking lead times, and personal preferences.*"
        )
        return "\n".join(lines)
