"""Block 7 QAOA ablations and untouched-pipeline generalization runs."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from calibrate_surrogate import sample_scenario
from src.model import Scenario, enumerate_all, score
from src.qaoa import _build_circuit, qubo_to_ising
from src.qubo import build_qubo, evaluate_qubo

FIXED_SEEDS = [1, 7, 21, 42, 101]
DEPTHS = [1, 2, 3, 4]
PENALTY_SCALES = [0.5, 1.0, 2.0]
INITIALIZATIONS = ["linear-ramp", "random", "fixed"]
RANDOM_BANK_SEED = 20261017
DEFAULT_BANK_SIZE = 4
DEFAULT_EVAL_CAP = 80


def scenario_dict(s: Scenario) -> dict:
    return {
        "name": s.name, "num_slots": s.num_slots, "capacity": s.capacity,
        "storage": s.storage, "overflow_weight": s.overflow_weight,
        "switch_weight": s.switch_weight,
        "link": {"from": s.link_from, "to": s.link_to,
                 "approach": s.link_approach, "fraction": s.link_fraction},
        "arrivals": s.arrivals,
    }


def initial_parameters(p: int, kind: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "linear-ramp":
        values = np.r_[np.linspace(0.4, 0.8, p), np.linspace(0.15, 0.45, p)]
        return values + rng.uniform(-0.01, 0.01, len(values))
    if kind == "random":
        return np.r_[rng.uniform(0, math.pi, p), rng.uniform(0, math.pi / 2, p)]
    if kind == "fixed":
        return np.r_[np.full(p, 0.45), np.full(p, 0.25)]
    raise ValueError(kind)


def optimize_case(scenario: Scenario, seed: int, p: int, penalty_scale: float,
                  initialization: str, evaluation_cap: int) -> dict:
    gamma_default = 2.0 * scenario.overflow_weight / (scenario.storage ** 2)
    Q, offset, _ = build_qubo(scenario, gamma=gamma_default * penalty_scale)
    _, h, J = qubo_to_ising(Q, offset)
    scale = float(np.sqrt(np.sum(h ** 2) + np.sum(J ** 2))) or 1.0
    circuit, gammas, betas = _build_circuit(p, h / scale, J / scale)
    energies = np.array([evaluate_qubo(Q, offset, [(i >> q) & 1 for q in range(8)])
                         for i in range(256)])
    calls = 0
    best_value = float("inf")
    best_theta = None
    cached = float(np.mean(energies))

    def objective(theta):
        nonlocal calls, best_value, best_theta, cached
        if calls >= evaluation_cap:
            return cached
        bindings = {gammas[k]: float(theta[k]) for k in range(p)}
        bindings.update({betas[k]: float(theta[p + k]) for k in range(p)})
        probabilities = Statevector.from_instruction(
            circuit.assign_parameters(bindings)).probabilities()
        cached = float(probabilities @ energies)
        calls += 1
        if cached < best_value:
            best_value, best_theta = cached, np.array(theta, dtype=float).copy()
        return cached

    initial = initial_parameters(p, initialization, seed)
    started = time.perf_counter()
    result = minimize(objective, initial, method="COBYLA",
                      options={"maxiter": evaluation_cap, "rhobeg": 0.3, "tol": 1e-4})
    elapsed = time.perf_counter() - started
    if best_theta is None:
        best_theta = result.x
    bindings = {gammas[k]: float(best_theta[k]) for k in range(p)}
    bindings.update({betas[k]: float(best_theta[p + k]) for k in range(p)})
    probabilities = Statevector.from_instruction(
        circuit.assign_parameters(bindings)).probabilities()

    official_costs = np.array([score(scenario, "".join(str((i >> q) & 1) for q in range(8)))
                               for i in range(256)])
    def schedule_bits(i: int) -> str:
        return "".join(str((i >> q) & 1) for q in range(8))
    ranked = sorted((score(scenario, schedule_bits(i)), schedule_bits(i)) for i in range(256))
    elite_bits = {bits for _, bits in ranked[:13]}
    elite_prob = sum(probabilities[i] for i in range(256)
                     if schedule_bits(i) in elite_bits)
    c_star = min(official_costs)
    c_rand = float(np.mean(official_costs))
    e_official = float(probabilities @ official_costs)
    lift = (c_rand - e_official) / (c_rand - c_star) if c_rand > c_star else 0.0
    enrichment = elite_prob / (13 / 256)
    return {"scenario": scenario.name, "seed": seed, "p": p,
            "penalty_scale": penalty_scale, "initialization": initialization,
            "evaluations": calls, "evaluation_cap": evaluation_cap,
            "optimizer_success": bool(result.success), "optimizer_seconds": elapsed,
            "expected_official_cost": e_official, "C_star": c_star,
            "C_rand": c_rand, "lift": lift, "elite_probability": elite_prob,
            "elite_enrichment": enrichment,
            "expected_qubo_energy": float(probabilities @ energies)}


def markdown_report(rows: list[dict], scenarios: list[Scenario], bank_seed: int,
                    eval_cap: int, generalization: list[dict]) -> str:
    def mean(items, key):
        return float(np.mean([item[key] for item in items]))

    lines = ["# Block 7 ablations and generalization", "",
             f"Five seeds: {', '.join(map(str, FIXED_SEEDS))}. Random bank: {len(scenarios)-1} scenarios generated with new seed `{bank_seed}` (R excluded from generation). QAOA expectation values are computed from the optimized ideal statevector; they are not finite-shot estimates.",
             f"Each ablation optimizer has an explicit {eval_cap}-evaluation cap. Factors: p=1..4; gamma multiplier x0.5/x1/x2; initialization linear-ramp/random/fixed.",
             "", "## Depth ablation (averaged over penalty, initialization, scenarios, and seeds)", "",
             "| p | Expected official cost | Lift | Elite enrichment | Mean optimizer seconds |",
             "|---:|---:|---:|---:|---:|"]
    for p in DEPTHS:
        subset = [r for r in rows if r["p"] == p]
        lines.append(f"| {p} | {mean(subset, 'expected_official_cost'):.3f} | {mean(subset, 'lift'):.5f} | {mean(subset, 'elite_enrichment'):.4f} | {mean(subset, 'optimizer_seconds'):.3f} |")
    lines += ["", "## Penalty-scale ablation", "",
              "| Gamma multiplier | Expected official cost | Lift | Elite enrichment |", "|---:|---:|---:|---:|"]
    for scale in PENALTY_SCALES:
        subset = [r for r in rows if r["penalty_scale"] == scale]
        lines.append(f"| x{scale:g} | {mean(subset, 'expected_official_cost'):.3f} | {mean(subset, 'lift'):.5f} | {mean(subset, 'elite_enrichment'):.4f} |")
    lines += ["", "## Initialization ablation", "",
              "| Initialization | Expected official cost | Lift | Elite enrichment |", "|---|---:|---:|---:|"]
    for kind in INITIALIZATIONS:
        subset = [r for r in rows if r["initialization"] == kind]
        lines.append(f"| {kind} | {mean(subset, 'expected_official_cost'):.3f} | {mean(subset, 'lift'):.5f} | {mean(subset, 'elite_enrichment'):.4f} |")
    lines += ["", "## Scenario-level default-pipeline generalization", "",
              "These results come from the normal `run.py` CLI at its unchanged defaults (p=1, gamma x1). Each scenario uses all five seeds and the usual 4,096 shots. Runtime shown is the QAOA stage runtime printed by the CLI.",
              "", "| Scenario | Type | C* | C_rand | Mean E_Q | Mean lift | Minimum lift clipped [0,1] | Mean elite enrichment | Max evaluations | Max runtime/seed (s) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in generalization:
        lines.append(f"| {r['scenario']} | unmodified default pipeline | {r['C_star']:.3f} | {r['C_rand']:.3f} | {r['mean_E_Q']:.3f} | {r['mean_lift']:.5f} | {r['minimum_lift_clipped']:.5f} | {r['mean_elite_enrichment']:.4f} | {r['max_evaluations']}/300 | {r['max_runtime_per_seed_seconds']:.3f} |")
    lines += ["", "### Default-pipeline results by scenario and seed", "",
              "| Scenario | Seed | E_Q | Lift | Lift clipped | Elite enrichment | Evaluations | QAOA runtime (s) | CLI wall time (s) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    per_seed_totals = {seed: 0.0 for seed in FIXED_SEEDS}
    for r in generalization:
        for item in r["per_seed"]:
            per_seed_totals[item["seed"]] += item["wall_seconds"]
            lines.append(f"| {r['scenario']} | {item['seed']} | {item['E_Q']:.3f} | {item['lift']:.5f} | {item['lift_clipped']:.5f} | {item['elite_enrichment']:.4f} | {item['evaluations']}/300 | {item['qaoa_runtime_seconds']:.3f} | {item['wall_seconds']:.3f} |")
    if generalization:
        lines += ["", "Cumulative CLI wall time across all scenarios per seed: " + ", ".join(f"seed {s}: {t:.2f}s" for s, t in per_seed_totals.items()) + f" (maximum {max(per_seed_totals.values()):.2f}s)."]
    lines += ["", "Scenario H was not present at run time. The random bank is persisted alongside this report so runs can be reproduced. No scenario-specific parameters are selected.", ""]
    return "\n".join(lines)


def run_generalization(scenario_files: list[Path], seeds: list[int], out_root: Path) -> list[dict]:
    records = []
    for path in scenario_files:
        scenario = Scenario.from_json(path)
        dest_base = out_root / "generalization" / scenario.name
        all_costs = enumerate_all(scenario)
        c_star = all_costs[0][1]
        c_rand = float(np.mean([v for _, v in all_costs]))
        elite_bits = {bits for bits, _ in all_costs[:13]}
        per_seed = []
        for seed in seeds:
            dest = dest_base / f"seed_{seed}"
            started = time.perf_counter()
            proc = subprocess.run([sys.executable, str(ROOT / "run.py"),
                                   "--scenario", str(path), "--seed", str(seed),
                                   "--out", str(dest)], cwd=ROOT, capture_output=True,
                                  text=True, timeout=600)
            wall = time.perf_counter() - started
            if proc.returncode:
                raise RuntimeError(f"pipeline failed for {scenario.name} seed {seed}: {proc.stderr}")
            schedule = json.loads((dest / "schedule.json").read_text())
            counts = json.loads((dest / "counts.json").read_text())
            shots = sum(counts.values())
            if shots != 4096:
                raise AssertionError(f"{scenario.name}/{seed}: got {shots} shots")
            e_q = sum(score(scenario, bits) * count for bits, count in counts.items()) / shots
            raw_lift = (c_rand - e_q) / (c_rand - c_star) if c_rand > c_star else 0.0
            elite_frac = sum(counts.get(bits, 0) for bits in elite_bits) / shots
            match = re.search(r"Runtime:\s*([0-9.]+)s", proc.stdout)
            if not match:
                raise RuntimeError(f"pipeline did not report runtime for {scenario.name}/{seed}")
            qaoa_runtime = float(match.group(1))
            evals = schedule["optimizer"]["evaluations"]
            if wall >= 600 or evals > 300:
                raise AssertionError(f"runtime/evaluation cap exceeded at {scenario.name}/{seed}")
            per_seed.append({"seed": seed, "E_Q": e_q, "lift": raw_lift,
                             "lift_clipped": max(0.0, min(1.0, raw_lift)),
                             "elite_enrichment": elite_frac / (13 / 256),
                             "evaluations": evals, "qaoa_runtime_seconds": qaoa_runtime,
                             "wall_seconds": wall, "most_frequent_bits": schedule["bits"],
                             "hybrid_cost": schedule["hybrid"]["official_cost"]})
        records.append({"scenario": scenario.name, "scenario_file": str(path),
                        "C_star": c_star, "C_rand": c_rand, "per_seed": per_seed,
                        "mean_E_Q": float(np.mean([r["E_Q"] for r in per_seed])),
                        "mean_lift": float(np.mean([r["lift"] for r in per_seed])),
                        "minimum_lift_clipped": min(r["lift_clipped"] for r in per_seed),
                        "mean_elite_enrichment": float(np.mean([r["elite_enrichment"] for r in per_seed])),
                        "max_evaluations": max(r["evaluations"] for r in per_seed),
                        "max_runtime_per_seed_seconds": max(r["qaoa_runtime_seconds"] for r in per_seed),
                        "max_cli_wall_seconds": max(r["wall_seconds"] for r in per_seed)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-size", type=int, default=DEFAULT_BANK_SIZE,
                        help="Number of fresh random scenarios (default: 4)")
    parser.add_argument("--eval-cap", type=int, default=DEFAULT_EVAL_CAP)
    parser.add_argument("--skip-generalization", action="store_true")
    parser.add_argument("--reuse-results", action="store_true",
                        help="Regenerate the report from existing result JSON without rerunning the sweep")
    args = parser.parse_args()
    if args.bank_size < 1 or args.eval_cap < 1 or args.eval_cap > 300:
        parser.error("bank-size must be positive and eval-cap must be 1..300")

    out_root = ROOT / "outputs" / "block7"
    bank_dir = out_root / "random_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_results:
        ablation_payload = json.loads((out_root / "ablation_results.json").read_text())
        manifest = json.loads((out_root / "random_bank_manifest.json").read_text())
        scenario_paths = [ROOT / "scenario_R.json", *[ROOT / p for p in manifest["scenario_files"]]]
        scenarios = [Scenario.from_json(p) for p in scenario_paths]
        rows = ablation_payload["runs"]
        eval_cap = ablation_payload["evaluation_cap"]
        bank_seed = ablation_payload["random_bank_seed"]
        generalization_path = out_root / "generalization.json"
        generalization = json.loads(generalization_path.read_text()) if generalization_path.exists() else []
    else:
        rng = __import__("random").Random(RANDOM_BANK_SEED)
        scenarios = [Scenario.from_json(ROOT / "scenario_R.json")]
        scenario_paths = [ROOT / "scenario_R.json"]
        for i in range(args.bank_size):
            scenario = sample_scenario(rng, f"G{(i+1):03d}")
            path = bank_dir / f"{scenario.name}.json"
            path.write_text(json.dumps(scenario_dict(scenario), indent=2), encoding="utf-8")
            scenarios.append(scenario)
            scenario_paths.append(path)
        bank_manifest = {"seed": RANDOM_BANK_SEED, "count": args.bank_size,
                         "scenario_files": [str(p.relative_to(ROOT)) for p in scenario_paths[1:]],
                         "generation": "calibrate_surrogate.sample_scenario; arrivals 0..8, capacity 4..6, storage 5..8, weights and link values per that generator"}
        (out_root / "random_bank_manifest.json").write_text(json.dumps(bank_manifest, indent=2), encoding="utf-8")
        rows = []
        configs = list(itertools.product(DEPTHS, PENALTY_SCALES, INITIALIZATIONS))
        total = len(configs) * len(scenarios) * len(FIXED_SEEDS)
        done = 0
        for scenario in scenarios:
            for seed in FIXED_SEEDS:
                for p, scale, init in configs:
                    result = optimize_case(scenario, seed, p, scale, init, args.eval_cap)
                    rows.append(result)
                    done += 1
                    if done % 30 == 0 or done == total:
                        print(f"Ablation progress: {done}/{total}", flush=True)
        ablation_payload = {"random_bank_seed": RANDOM_BANK_SEED, "scenario_names": [s.name for s in scenarios],
                            "seeds": FIXED_SEEDS, "depths": DEPTHS, "penalty_scales": PENALTY_SCALES,
                            "initializations": INITIALIZATIONS, "evaluation_cap": args.eval_cap,
                            "runs": rows}
        (out_root / "ablation_results.json").write_text(json.dumps(ablation_payload, indent=2), encoding="utf-8")
        generalization = []
        if not args.skip_generalization:
            paths = [ROOT / "scenario_R.json", *scenario_paths[1:]]
            if (ROOT / "scenario_H.json").exists():
                paths.insert(1, ROOT / "scenario_H.json")
            generalization = run_generalization(paths, FIXED_SEEDS, out_root)
            (out_root / "generalization.json").write_text(json.dumps(generalization, indent=2), encoding="utf-8")
        eval_cap = args.eval_cap
        bank_seed = RANDOM_BANK_SEED

    report = markdown_report(rows, scenarios, bank_seed, eval_cap, generalization)
    (out_root / "block7_ablation.md").write_text(report, encoding="utf-8")
    print(report)
    if generalization:
        print(f"Generalization run count: {sum(len(r['per_seed']) for r in generalization)}; all runs completed below 600 s per CLI invocation.")
    if not (ROOT / "scenario_H.json").exists():
        print("Scenario H is not present; R and the fresh random bank were used.")


if __name__ == "__main__":
    main()
