"""
Tests for the traffic model (src/model.py).

Verifies against the 4 official test vectors for scenario R,
plus additional edge-case and property tests.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import Scenario, evaluate_schedule, score, decode_schedule, enumerate_all


SCENARIO_R = Path(__file__).resolve().parent.parent / "scenario_R.json"


class TestDecodeSchedule(unittest.TestCase):
    def test_string_input(self):
        a, b = decode_schedule("10101010")
        self.assertEqual(a, [1, 0, 1, 0])
        self.assertEqual(b, [1, 0, 1, 0])

    def test_list_input(self):
        a, b = decode_schedule([0, 1, 1, 1, 1, 0, 0, 1])
        self.assertEqual(a, [0, 1, 1, 1])
        self.assertEqual(b, [1, 0, 0, 1])

    def test_wrong_length_raises(self):
        with self.assertRaises(AssertionError):
            decode_schedule("1010")


class TestOfficialVectors(unittest.TestCase):
    """Verify against the 4 official test vectors from the problem statement."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = Scenario.from_json(SCENARIO_R)

    def test_1010_1010(self):
        cost = score(self.scenario, "10101010")
        self.assertAlmostEqual(cost, 116.0, places=1)

    def test_1111_1111(self):
        cost = score(self.scenario, "11111111")
        self.assertAlmostEqual(cost, 146.0, places=1)

    def test_0000_0000(self):
        cost = score(self.scenario, "00000000")
        self.assertAlmostEqual(cost, 237.0, places=1)

    def test_0111_1001(self):
        cost = score(self.scenario, "01111001")
        self.assertAlmostEqual(cost, 92.5, places=1)


class TestModelProperties(unittest.TestCase):
    """Property-based tests for the model."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = Scenario.from_json(SCENARIO_R)

    def test_all_256_schedules_produce_finite_cost(self):
        """Every schedule must produce a finite, non-negative cost."""
        for i in range(256):
            bits = format(i, "08b")
            c = score(self.scenario, bits)
            self.assertIsInstance(c, float)
            self.assertGreaterEqual(c, 0.0)
            self.assertTrue(c < float("inf"))

    def test_enumerate_returns_256(self):
        results = enumerate_all(self.scenario)
        self.assertEqual(len(results), 256)

    def test_enumerate_sorted(self):
        results = enumerate_all(self.scenario)
        costs = [c for _, c in results]
        self.assertEqual(costs, sorted(costs))

    def test_cost_components_nonneg(self):
        """Delay, overflow, and switches must all be non-negative."""
        for i in range(256):
            bits = format(i, "08b")
            result = evaluate_schedule(self.scenario, bits)
            self.assertGreaterEqual(result["delay"], 0)
            self.assertGreaterEqual(result["overflow"], 0)
            self.assertGreaterEqual(result["switches"], 0)

    def test_switches_range(self):
        """Switches per intersection: 0 to 3. Total: 0 to 6."""
        for i in range(256):
            bits = format(i, "08b")
            result = evaluate_schedule(self.scenario, bits)
            self.assertLessEqual(result["switches"], 6)

    def test_no_arrivals_no_delay(self):
        """With zero arrivals and no transfers, cost should be zero."""
        import json
        import tempfile
        zero_scenario = {
            "name": "zero",
            "num_slots": 4,
            "capacity": 5,
            "storage": 7,
            "overflow_weight": 4,
            "switch_weight": 0,
            "link": {"from": "A", "to": "B", "approach": "EW", "fraction": 0.6},
            "arrivals": {
                "A": {"NS": [0, 0, 0, 0], "EW": [0, 0, 0, 0]},
                "B": {"NS": [0, 0, 0, 0], "EW": [0, 0, 0, 0]},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(zero_scenario, f)
            tmppath = f.name

        s = Scenario.from_json(tmppath)
        for i in range(256):
            bits = format(i, "08b")
            c = score(s, bits)
            # Cost should be 0 when switch_weight is 0 and no arrivals
            self.assertAlmostEqual(c, 0.0, places=5)

        import os
        os.unlink(tmppath)


class TestEvaluateScheduleDetails(unittest.TestCase):
    """Test specific details of the evaluate_schedule output."""

    @classmethod
    def setUpClass(cls):
        cls.scenario = Scenario.from_json(SCENARIO_R)

    def test_1111_no_switches(self):
        result = evaluate_schedule(self.scenario, "11111111")
        self.assertEqual(result["switches"], 0)

    def test_0000_no_switches(self):
        result = evaluate_schedule(self.scenario, "00000000")
        self.assertEqual(result["switches"], 0)

    def test_1010_max_switches(self):
        result = evaluate_schedule(self.scenario, "10101010")
        self.assertEqual(result["switches"], 6)

    def test_0111_1001_transfers(self):
        """0111|1001: A gives EW green at slot 1, so transfers should go to B."""
        result = evaluate_schedule(self.scenario, "01111001")
        # Slot 1: A gives EW green (bit=0), A.EW arrivals=5, serve min(5,5)=5
        # Transfer = floor(0.6 * 5 + 0.5) = floor(3.5) = 3
        self.assertEqual(result["transfers_to_B"][1], 3)


if __name__ == "__main__":
    unittest.main()
