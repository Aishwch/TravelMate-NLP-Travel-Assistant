"""
Unit tests for src/budget.py
"""

import unittest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.budget import BudgetPlanner


class TestBudgetPlanner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.budget_df = pd.read_csv("data/processed/budget.csv")
        cls.planner = BudgetPlanner(cls.budget_df)

    def test_budget_breakdown_math(self):
        plan = self.planner.plan_budget(
            destination="Goa",
            total_budget=10000,
            duration_days=3,
            people_count=1
        )
        self.assertEqual(plan["destination"], "Goa")
        self.assertEqual(plan["duration_days"], 3)
        bd = plan["breakdown"]

        # Ensure all key subcomponents exist
        self.assertIn("accommodation", bd)
        self.assertIn("food_dining", bd)
        self.assertIn("local_transport", bd)
        self.assertIn("activities_sightseeing", bd)
        self.assertIn("contingency_buffer", bd)

        # Check total estimated sum matches components
        calculated_sum = sum(bd.values())
        self.assertEqual(plan["total_estimated"], calculated_sum)

    def test_feasibility_assessment_comfortable(self):
        # 30,000 for 2 days in Matheran should be very comfortable
        plan = self.planner.plan_budget(
            destination="Matheran",
            total_budget=30000,
            duration_days=2,
            people_count=1
        )
        self.assertEqual(plan["feasibility_status"], "Comfortable")

    def test_feasibility_assessment_tight(self):
        # 1,000 total for 5 days in Mumbai is under-budget
        plan = self.planner.plan_budget(
            destination="Mumbai",
            total_budget=1000,
            duration_days=5,
            people_count=1
        )
        self.assertEqual(plan["feasibility_status"], "Tight / Under-Budget")


if __name__ == "__main__":
    unittest.main()
