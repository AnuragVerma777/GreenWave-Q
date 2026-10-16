#!/usr/bin/env python3
"""
Quantum Traffic Gridlock — Main Pipeline

Usage:
    python run.py --scenario <scenario.json> --seed <int> --out <dir>

QC@IITI Qiskit Fall Fest 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import Scenario, evaluate_schedule, score, enumerate_all
from src.baselines import run_all_baselines
from src.metrics import (
    best_cost,
    uniform_random_expected_cost,
    lift,
    elite_set,
    elite_enrichment,
    mean_cost_from_counts,
    surrogate_regret,
)
from src.qaoa import run_qaoa


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Traffic Gridlock — Signal Optimization Pipeline"
    )
    parser.add_argument(
        "--scenario", required=True, help="Path to scenario JSON file"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out", required=True, help="Output directory"
    )
    args = parser.parse_args()

    # Load scenario
    scenario = Scenario.from_json(args.scenario)
    print(f"Loaded scenario: {scenario.name}")
    print(f"  Capacity={scenario.capacity}, Storage={scenario.storage}")
    print(f"  Overflow weight={scenario.overflow_weight}, Switch weight={scenario.switch_weight}")
    print(f"  Link: {scenario.link_from}->{scenario.link_to} on {scenario.link_approach}, fraction={scenario.link_fraction}")
    print()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run baselines
    print("=" * 60)
    print("BASELINES")
    print("=" * 60)
    baselines = run_all_baselines(scenario)

    alt = baselines["alternating"]
    print(f"  Best alternating:  {alt['schedule']} -> cost={alt['cost']:.1f}")

    gr = baselines["greedy"]
    print(f"  Greedy:            {gr['schedule']} -> cost={gr['cost']:.1f}")

    enum = baselines["enumeration"]
    print(f"  Best (enumeration):{enum['best_schedule']} -> cost={enum['best_cost']:.1f}")

    rand = baselines["uniform_random"]
    print(f"  Uniform random:    expected cost={rand['expected_cost']:.2f}")

    # Show top-10 schedules
    print()
    print("Top 10 schedules by cost:")
    for i, (bs, c) in enumerate(enum["all_results"][:10]):
        print(f"  {i+1:2d}. {bs[:4]}|{bs[4:]}  cost={c:.1f}")

    # Save baselines summary
    baselines_out = {
        "alternating": alt,
        "greedy": gr,
        "best_enumeration": {
            "schedule": enum["best_schedule"],
            "cost": enum["best_cost"],
        },
        "uniform_random_expected_cost": rand["expected_cost"],
        "top_10": [
            {"schedule": bs, "cost": c}
            for bs, c in enum["all_results"][:10]
        ],
    }
    with open(out_dir / "baselines.json", "w") as f:
        json.dump(baselines_out, f, indent=2)
    print(f"\nBaselines saved to {out_dir / 'baselines.json'}")

    print()
    print("=" * 60)
    print("QAOA")
    print("=" * 60)
    qaoa = run_qaoa(scenario, args.seed, out_dir)
    print(f"  Depth p: {qaoa['qaoa']['p']}")
    print(f"  Optimizer evaluations: {qaoa['optimizer']['evaluations']}/{qaoa['optimizer']['evaluation_cap']}")
    print(f"  Shots: {qaoa['qaoa']['shots']}")
    print(f"  Raw most frequent: {qaoa['bits']}")
    print(f"  Hybrid official cost: {qaoa['hybrid']['official_cost']:.3f}")
    print(f"  Runtime: {qaoa['qaoa']['runtime_seconds']:.3f}s")

    print("=" * 60)
    print("Pipeline complete (model + baselines + QAOA).")
    print("=" * 60)


if __name__ == "__main__":
    main()
