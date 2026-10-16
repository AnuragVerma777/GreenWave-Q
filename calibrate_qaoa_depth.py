"""Small, reproducible p=1 vs p=2 training-bank ablation for QAOA depth."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector

from calibrate_surrogate import sample_scenario
from src.qubo import build_qubo, evaluate_qubo
from src.qaoa import _build_circuit, qubo_to_ising

SEED = 20261016
N_SCENARIOS = 16
P_VALUES = (1, 2)
EVALS = 80


def main():
    rng = random.Random(SEED)
    results = {p: [] for p in P_VALUES}
    runtimes = {p: [] for p in P_VALUES}
    import time
    for si in range(N_SCENARIOS):
        scenario = sample_scenario(rng, f"train-{si:02d}")
        Q, offset, _ = build_qubo(scenario)
        _, h, J = qubo_to_ising(Q, offset)
        energies = np.array([evaluate_qubo(Q, offset, [(i >> q) & 1 for q in range(8)])
                             for i in range(256)])
        ground = float(energies.min())
        energy_range = max(float(energies.max() - energies.min()), 1.0)
        for p in P_VALUES:
            hamiltonian_scale = float(np.sqrt(np.sum(h ** 2) + np.sum(J ** 2))) or 1.0
            circuit, gammas, betas = _build_circuit(
                p, h / hamiltonian_scale, J / hamiltonian_scale)
            local_rng = np.random.default_rng(SEED + si * 10 + p)
            initial = np.r_[np.linspace(.4, .8, p), np.linspace(.2, .4, p)]
            initial += local_rng.uniform(-.01, .01, len(initial))
            calls = 0
            def objective(theta):
                nonlocal calls
                calls += 1
                bind = {gammas[k]: theta[k] for k in range(p)}
                bind.update({betas[k]: theta[p+k] for k in range(p)})
                probs = Statevector.from_instruction(circuit.assign_parameters(bind)).probabilities()
                return float(probs @ energies)
            start = time.perf_counter()
            opt = minimize(objective, initial, method="COBYLA", options={"maxiter": EVALS, "rhobeg": .3})
            runtime = time.perf_counter() - start
            # The optimized expectation, rather than the best sampled bitstring, measures depth ablation.
            value = float(opt.fun)
            results[p].append((value - ground) / energy_range)
            runtimes[p].append(runtime)
    report = ["# QAOA depth ablation", "", f"Training bank: {N_SCENARIOS} independently generated scenarios (seed {SEED}); random generator and ranges match `calibrate_surrogate.py`. Neither R nor H is part of this bank.",
              f"Both depths use seeded COBYLA, linear-ramp initialization, and at most {EVALS} evaluations per scenario. Metric is normalized expected-QUBO-energy gap `(E_QAOA - E_min)/(E_max-E_min)` after optimization. Lower is better.", "", "| p | Mean normalized gap | Median normalized gap | Mean optimizer time (s) | Maximum ZZ gates |", "|---:|---:|---:|---:|---:|"]
    for p in P_VALUES:
        report.append(f"| {p} | {np.mean(results[p]):.5f} | {np.median(results[p]):.5f} | {np.mean(runtimes[p]):.4f} | {p * 28} |")
    gap_delta = np.mean(results[1]) - np.mean(results[2])
    time_ratio = np.mean(runtimes[2]) / np.mean(runtimes[1])
    report += ["", f"Default selection: p=1. On this training bank, p=2 reduced normalized gap by {gap_delta:.5f} while taking {time_ratio:.2f} times the mean optimizer time and doubling the maximum two-qubit gate count. We choose the smaller layer count for the production default; it is fixed and is not selected per scenario.", ""]
    target = Path(__file__).resolve().parent / "docs" / "qaoa_ablation.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
