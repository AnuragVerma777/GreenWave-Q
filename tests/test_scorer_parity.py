#!/usr/bin/env python3
"""
CHECKPOINT 0 — Parity test: src/model.py vs official scorer.py

Compare cost, delay, overflow, and switches on:
  1. Four official reference vectors for scenario R.
  2. All 256 schedules on the official scenario_R.json.
  3. All 256 schedules on 100 random hidden-range scenarios.

All four quantities must match exactly.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the official scorer from its directory
# ---------------------------------------------------------------------------
_SCORER_PATH = Path(__file__).resolve().parent.parent / "scorer.py"
_spec = importlib.util.spec_from_file_location("official_scorer", _SCORER_PATH)
_official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_official)
evaluate_official = _official.evaluate

# ---------------------------------------------------------------------------
# Import our model
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import Scenario, evaluate_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_scenario(rng: random.Random, name: str) -> Scenario:
    """Generate a random scenario within the published hidden-range bounds.

    Arrivals 0-8, capacity 4-6, storage 5-8, overflow weight 2-6,
    switch weight 0.25-1.5, link phase NS or EW, fraction 0.4-0.8
    rounded to 2 decimal places.
    """
    approach = rng.choice(("NS", "EW"))
    arrivals = {
        intersection: {
            phase: [rng.randint(0, 8) for _ in range(4)]
            for phase in ("NS", "EW")
        }
        for intersection in ("A", "B")
    }
    return Scenario(
        name=name,
        num_slots=4,
        capacity=rng.randint(4, 6),
        storage=rng.randint(5, 8),
        overflow_weight=round(rng.uniform(2, 6), 2),
        switch_weight=round(rng.uniform(0.25, 1.5), 2),
        link_from="A",
        link_to="B",
        link_approach=approach,
        link_fraction=round(rng.uniform(0.4, 0.8), 2),
        arrivals=arrivals,
    )


def _compare_all_256(scenario: Scenario) -> list[dict]:
    """Compare all 256 schedules between our model and the official scorer.

    Returns a list of mismatch dicts (empty = pass).
    """
    official_dict = scenario.to_scorer_dict()
    mismatches = []

    for i in range(256):
        bits_str = format(i, "08b")
        bits_list = [int(b) for b in bits_str]

        # Our model
        ours = evaluate_schedule(scenario, bits_str)

        # Official scorer
        theirs = evaluate_official(official_dict, bits_list)

        # Compare all four fields
        if (ours["delay"] != theirs["delay"]
                or ours["overflow"] != theirs["overflow"]
                or ours["switches"] != theirs["switches"]
                or round(ours["cost"], 6) != theirs["cost"]):
            mismatches.append({
                "schedule": bits_str,
                "ours": {
                    "cost": ours["cost"],
                    "delay": ours["delay"],
                    "overflow": ours["overflow"],
                    "switches": ours["switches"],
                },
                "theirs": {
                    "cost": theirs["cost"],
                    "delay": theirs["delay"],
                    "overflow": theirs["overflow"],
                    "switches": theirs["switches"],
                },
            })
    return mismatches


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReferenceVectors:
    """Verify the 4 official test vectors from the problem statement."""

    VECTORS = [
        ("10101010", 116.0),
        ("11111111", 146.0),
        ("00000000", 237.0),
        ("01111001", 92.5),
    ]

    def test_our_model(self):
        """Our fast model matches the reference costs."""
        s = Scenario.from_json(
            Path(__file__).resolve().parent.parent / "scenario_R.json"
        )
        for bits, expected in self.VECTORS:
            result = evaluate_schedule(s, bits)
            assert result["cost"] == expected, (
                f"Schedule {bits}: expected cost {expected}, got {result['cost']}"
            )

    def test_official_scorer(self):
        """The official scorer matches the reference costs."""
        import json
        p = Path(__file__).resolve().parent.parent / "scenario_R.json"
        with open(p) as f:
            scenario_dict = json.load(f)
        for bits, expected in self.VECTORS:
            result = evaluate_official(scenario_dict, bits)
            assert result["cost"] == expected, (
                f"Schedule {bits}: expected cost {expected}, got {result['cost']}"
            )

    def test_both_agree(self):
        """Our model and the official scorer give the same results for all 4."""
        s = Scenario.from_json(
            Path(__file__).resolve().parent.parent / "scenario_R.json"
        )
        official_dict = s.to_scorer_dict()
        for bits, _ in self.VECTORS:
            ours = evaluate_schedule(s, bits)
            theirs = evaluate_official(official_dict, [int(b) for b in bits])
            assert ours["delay"] == theirs["delay"], f"{bits}: delay mismatch"
            assert ours["overflow"] == theirs["overflow"], f"{bits}: overflow mismatch"
            assert ours["switches"] == theirs["switches"], f"{bits}: switches mismatch"
            assert round(ours["cost"], 6) == theirs["cost"], f"{bits}: cost mismatch"


class TestScenarioRParity:
    """All 256 schedules on scenario R must match exactly."""

    def test_all_256_official_format(self):
        """Using the official scenario_R.json (scorer schema)."""
        s = Scenario.from_json(
            Path(__file__).resolve().parent.parent / "scenario_R.json"
        )
        mismatches = _compare_all_256(s)
        assert mismatches == [], (
            f"{len(mismatches)} mismatches on scenario R (official format).\n"
            f"First: {mismatches[0]}"
        )


class TestRandomScenariosParity:
    """100 random hidden-range scenarios, all 256 schedules each."""

    SEED = 20261016
    N_SCENARIOS = 100

    def test_100_random_all_256(self):
        """Cost, delay, overflow, and switches must match exactly."""
        rng = random.Random(self.SEED)
        total_compared = 0
        all_mismatches: list[tuple[str, dict]] = []

        for idx in range(self.N_SCENARIOS):
            s = _random_scenario(rng, f"V{idx:03d}")
            mismatches = _compare_all_256(s)
            total_compared += 256
            for m in mismatches:
                all_mismatches.append((s.name, m))

        # Summary
        print(
            f"\nCompared {total_compared} evaluations "
            f"({self.N_SCENARIOS} scenarios × 256 schedules)."
        )

        if all_mismatches:
            # Report first 5
            lines = [
                f"{len(all_mismatches)} total mismatches "
                f"across {self.N_SCENARIOS} scenarios."
            ]
            for name, m in all_mismatches[:5]:
                lines.append(
                    f"  {name} {m['schedule']}: "
                    f"ours={m['ours']} theirs={m['theirs']}"
                )
            assert False, "\n".join(lines)

        print(f"All {total_compared} evaluations match exactly.")


# ---------------------------------------------------------------------------
# Also runnable as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("CHECKPOINT 0 — Scorer Parity Test")
    print("=" * 70)

    # Reference vectors
    print("\n--- Reference Vectors ---")
    s = Scenario.from_json(
        Path(__file__).resolve().parent.parent / "scenario_R.json"
    )
    for bits, expected in TestReferenceVectors.VECTORS:
        cost = evaluate_schedule(s, bits)["cost"]
        status = "PASS" if cost == expected else "FAIL"
        print(f"  {status} {bits} -> {cost} (expected {expected})")

    # All 256 on scenario R
    print("\n--- Scenario R: all 256 schedules ---")
    t0 = time.perf_counter()
    mismatches = _compare_all_256(s)
    dt = time.perf_counter() - t0
    print(f"  {'PASS' if not mismatches else 'FAIL'} "
          f"{256 - len(mismatches)}/256 match ({dt:.2f}s)")
    if mismatches:
        for m in mismatches[:3]:
            print(f"    MISMATCH: {m}")

    # Also test official format
    s_off = Scenario.from_json(
        Path(__file__).resolve().parent.parent / "scenario_R.json"
    )
    mismatches_off = _compare_all_256(s_off)
    print(f"  {'PASS' if not mismatches_off else 'FAIL'} "
          f"Official format: {256 - len(mismatches_off)}/256 match")

    # 100 random scenarios
    print(f"\n--- 100 Random Hidden-Range Scenarios (seed {TestRandomScenariosParity.SEED}) ---")
    rng = random.Random(TestRandomScenariosParity.SEED)
    total, fail_scenarios = 0, 0
    all_mismatches: list[tuple[str, dict]] = []
    t0 = time.perf_counter()
    for idx in range(100):
        sc = _random_scenario(rng, f"V{idx:03d}")
        mm = _compare_all_256(sc)
        total += 256
        if mm:
            fail_scenarios += 1
            for m in mm:
                all_mismatches.append((sc.name, m))
    dt = time.perf_counter() - t0

    if all_mismatches:
        print(f"  FAIL: {len(all_mismatches)} mismatches in "
              f"{fail_scenarios} scenarios ({dt:.1f}s)")
        for name, m in all_mismatches[:5]:
            print(f"    {name} {m['schedule']}: "
                  f"ours={m['ours']}  theirs={m['theirs']}")
    else:
        print(f"  PASS: All {total} evaluations match exactly ({dt:.1f}s)")

    # Final verdict
    all_pass = (
        not mismatches and not mismatches_off and not all_mismatches
    )
    print("\n" + "=" * 70)
    print(f"CHECKPOINT 0: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)

