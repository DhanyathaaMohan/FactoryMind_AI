"""
test_condition_mapping.py
--------------------------
Unit tests for the tool-condition threshold mapping in
predictive_maintenance.map_tool_condition().

Thresholds
----------
>= 0.75        -> Healthy
0.50 – 0.74    -> Degrading
0.25 – 0.49    -> Worn
< 0.25         -> Critical
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from predictive_maintenance import map_tool_condition


class TestConditionMapping:

    # -- Healthy boundary --------------------------------------------------

    def test_exactly_0_75_is_healthy(self):
        assert map_tool_condition(0.75) == "Healthy"

    def test_above_0_75_is_healthy(self):
        assert map_tool_condition(0.90) == "Healthy"

    def test_max_rul_is_healthy(self):
        assert map_tool_condition(1.0) == "Healthy"

    # -- Degrading boundary ------------------------------------------------

    def test_exactly_0_50_is_degrading(self):
        assert map_tool_condition(0.50) == "Degrading"

    def test_midpoint_degrading(self):
        assert map_tool_condition(0.625) == "Degrading"

    def test_just_below_healthy_is_degrading(self):
        assert map_tool_condition(0.7499) == "Degrading"

    # -- Worn boundary -----------------------------------------------------

    def test_exactly_0_25_is_worn(self):
        assert map_tool_condition(0.25) == "Worn"

    def test_midpoint_worn(self):
        assert map_tool_condition(0.375) == "Worn"

    def test_just_below_degrading_is_worn(self):
        assert map_tool_condition(0.4999) == "Worn"

    # -- Critical boundary -------------------------------------------------

    def test_just_below_worn_is_critical(self):
        assert map_tool_condition(0.2499) == "Critical"

    def test_zero_rul_is_critical(self):
        assert map_tool_condition(0.0) == "Critical"

    def test_very_low_rul_is_critical(self):
        assert map_tool_condition(0.01) == "Critical"

    # -- Return type -------------------------------------------------------

    def test_returns_string(self):
        assert isinstance(map_tool_condition(0.5), str)

    # -- All four conditions reachable -------------------------------------

    def test_all_four_conditions_reachable(self):
        conditions = {
            map_tool_condition(0.9),
            map_tool_condition(0.6),
            map_tool_condition(0.35),
            map_tool_condition(0.1),
        }
        assert conditions == {"Healthy", "Degrading", "Worn", "Critical"}
