"""Generate Block 6 robustness and method-comparison reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.baselines import best_alternating, greedy_schedule
from src.metrics import elite_set, mean_cost_from_counts, uniform_random_expected_cost
from src.model import Scenario, enumerate_all, score
from src.qaoa import run_qaoa
from src.qubo import build_qubo, evaluate_qubo, qubo_ranking

DEFAULT_SEEDS = [1, 7, 21, 42, 101]
N_SHOTS = 4096
N_ELITE = 13


def calculate_report(scenario_path: Path, seeds: list[int], output_root: Path) -> dict:
    scenario = Scenario.from_json(scenario_path)
    results = enumerate_all(scenario)
    c_star = results[0][1]
    c_rand = sum(cost for _, cost in results) / 256
    elite = {bits for bits, _ in results[:N_ELITE]}  # sorted by (official cost, bits)
    Q, offset, _ = build_qubo(scenario)
    qubo_bits, qubo_energy = qubo_ranking(Q, offset)[0]
    qubo_official = score(scenario, qubo_bits)

    seed_rows = []
    sampled_hybrids = []
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        run = run_qaoa(scenario, seed, seed_dir)
        counts = json.loads((seed_dir / "counts.json").read_text())
        shots = sum(counts.values())
        if shots != N_SHOTS:
            raise AssertionError(f"seed {seed}: expected {N_SHOTS} shots, got {shots}")
        e_q = mean_cost_from_counts(scenario, counts)
        raw_lift = ((c_rand - e_q) / (c_rand - c_star)) if c_rand > c_star else 0.0
        clipped_lift = max(0.0, min(1.0, raw_lift))
        elite_shots = sum(counts.get(bits, 0) for bits in elite)
        enrichment = (elite_shots / shots) / (N_ELITE / 256)
        raw_mode = run["raw_qaoa"]["most_frequent"]["bits"]
        hybrid_bits = run["hybrid"]["bits"]
        sampled_hybrids.append((hybrid_bits, score(scenario, hybrid_bits), seed))
        seed_rows.append({
            "seed": seed,
            "shots": shots,
            "E_Q": e_q,
            "lift": raw_lift,
            "lift_clipped": clipped_lift,
            "elite_shots": elite_shots,
            "elite_fraction": elite_shots / shots,
            "elite_enrichment": enrichment,
            "raw_most_frequent_bits": raw_mode,
            "raw_most_frequent_official_cost": score(scenario, raw_mode),
            "hybrid_bits": hybrid_bits,
            "hybrid_official_cost": score(scenario, hybrid_bits),
            "optimizer_evaluations": run["optimizer"]["evaluations"],
            "runtime_seconds": run["qaoa"]["runtime_seconds"],
        })

    e_q_mean = sum(row["E_Q"] for row in seed_rows) / len(seed_rows)
    e_q_min, e_q_max = min(row["E_Q"] for row in seed_rows), max(row["E_Q"] for row in seed_rows)
    best_hybrid_bits, best_hybrid_cost, best_hybrid_seed = min(
        sampled_hybrids, key=lambda item: (item[1], item[0], item[2]))
    altern_bits, altern_cost = best_alternating(scenario)
    greedy_bits, greedy_cost = greedy_schedule(scenario)
    table = [
        {"method": "Best alternating", "schedule": altern_bits, "official_cost": altern_cost},
        {"method": "Greedy (ties to NS)", "schedule": greedy_bits, "official_cost": greedy_cost},
        {"method": "Exact enumeration", "schedule": results[0][0], "official_cost": c_star},
        {"method": "Uniform-random mean", "schedule": "—", "official_cost": c_rand},
        {"method": "Exact QUBO minimizer (officially scored)", "schedule": qubo_bits,
         "official_cost": qubo_official, "qubo_energy": qubo_energy},
        {"method": f"QAOA raw samples (mean E_Q, {len(seeds)} seeds)", "schedule": "4,096 samples/seed",
         "official_cost": e_q_mean, "min_seed_E_Q": e_q_min, "max_seed_E_Q": e_q_max},
        {"method": "Hybrid (best official cost among all sampled bitstrings)",
         "schedule": best_hybrid_bits, "official_cost": best_hybrid_cost,
         "seed": best_hybrid_seed},
    ]
    lifts = [row["lift_clipped"] for row in seed_rows]
    payload = {
        "scenario": scenario.name,
        "scenario_file": str(scenario_path),
        "seeds": seeds,
        "definitions": {
            "C_star": "minimum official cost over all 256 schedules",
            "C_rand": "uniform mean official cost over all 256 schedules",
            "E_Q": "mean official cost weighted by raw QAOA counts (4,096 shots)",
            "elite": "13 lowest official-cost schedules, ties by ascending bitstring",
            "elite_enrichment": "elite shot fraction divided by 13/256",
            "lift_clipped": "per-seed lift clipped to [0,1]",
            "hybrid": "best official-cost schedule among sampled bitstrings, across listed seeds",
        },
        "C_star": c_star,
        "C_rand": c_rand,
        "qubo_exact_minimizer": {"bits": qubo_bits, "qubo_energy": qubo_energy,
                                  "official_cost": qubo_official},
        "per_seed": seed_rows,
        "minimum_lift_clipped": min(lifts),
        "comparison": table,
        "hybrid_best": {"bits": best_hybrid_bits, "official_cost": best_hybrid_cost,
                        "seed": best_hybrid_seed},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "block6_metrics.json").write_text(json.dumps(payload, indent=2))
    lines = [f"# Block 6 metrics — {scenario.name}", "",
             f"Seeds: {', '.join(map(str, seeds))}. Each raw sample set contains {N_SHOTS:,} shots.",
             f"C* = {c_star:.3f}; C_rand = {c_rand:.3f}.",
             "", "## Method comparison", "",
             "| Method | Schedule | Official cost |", "|---|---:|---:|"]
    for row in table:
        lines.append(f"| {row['method']} | {row['schedule']} | {row['official_cost']:.3f} |")
    lines += ["", "## Per-seed raw-QAOA metrics", "",
              "| Seed | E_Q | Lift | Lift clipped | Elite shots | Elite enrichment | Raw mode | Raw mode cost | Hybrid schedule | Hybrid cost | Evaluations | Runtime (s) |",
              "|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|"]
    for row in seed_rows:
        lines.append(f"| {row['seed']} | {row['E_Q']:.3f} | {row['lift']:.5f} | {row['lift_clipped']:.5f} | {row['elite_shots']} | {row['elite_enrichment']:.4f} | {row['raw_most_frequent_bits']} | {row['raw_most_frequent_official_cost']:.3f} | {row['hybrid_bits']} | {row['hybrid_official_cost']:.3f} | {row['optimizer_evaluations']}/300 | {row['runtime_seconds']:.3f} |")
    lines += ["", f"**Minimum lift clipped to [0,1]: {min(lifts):.5f}.**",
              f"Elite enrichment is each seed's fraction of shots in the 13-schedule elite divided by 13/256. The hybrid row reports the best official-cost schedule found among all sampled bitstrings across these seeds; it is distinct from the raw QAOA mean.", ""]
    (output_root / "block6_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="scenario_R.json")
    parser.add_argument("--seed", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--out", help="Report root (default: outputs/<scenario name>)")
    args = parser.parse_args()
    scenario_path = Path(args.scenario).resolve()
    scenario = Scenario.from_json(scenario_path)
    out = Path(args.out).resolve() if args.out else ROOT / "outputs" / scenario.name
    report = calculate_report(scenario_path, args.seed, out)
    print((out / "block6_metrics.md").read_text())
    if not (ROOT / "scenario_H.json").exists() and scenario.name.upper() == "R":
        print("Scenario H is not present yet; this report covers R only.")


if __name__ == "__main__":
    main()

