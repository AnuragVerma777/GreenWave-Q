"""
Calibrate the overflow-surrogate coefficient gamma.

Tests multiple scenario-independent gamma formulas on scenario R and
reports surrogate regret, QUBO-minimiser true cost, rank of true optimum
in the QUBO ranking, and Spearman rank correlation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

from src.model import Scenario, score, enumerate_all
from src.qubo import build_qubo, evaluate_qubo, qubo_ranking

s = Scenario.from_json("scenario_R.json")

# True ranking
true_ranking = enumerate_all(s)  # sorted by (cost, bitstring)
c_star = true_ranking[0][1]
c_star_bs = true_ranking[0][0]
c_rand = sum(c for _, c in true_ranking) / 256

true_cost_map = {bs: c for bs, c in true_ranking}
true_rank_map = {bs: i for i, (bs, _) in enumerate(true_ranking)}

# Elite set (13 lowest-cost)
elite = {bs for bs, _ in true_ranking[:13]}

print(f"True optimum: {c_star_bs} -> C* = {c_star:.1f}")
print(f"C_rand = {c_rand:.2f}")
print(f"Regret denom = {c_rand - c_star:.2f}")
print()

w = s.overflow_weight
S = s.storage
C = s.capacity

# Gamma formulas to test (all are scenario-independent functions of w_ovf, S, C)
formulas = {
    "w/(4*S^2)":      w / (4 * S**2),
    "w/(2*S^2)":      w / (2 * S**2),
    "w/S^2":          w / S**2,
    "2w/S^2":         2 * w / S**2,
    "3w/S^2":         3 * w / S**2,
    "w/(4*S)":        w / (4 * S),
    "w/(2*S)":        w / (2 * S),
    "w/(S*C)":        w / (S * C),
    "w/(2*S*C)":      w / (2 * S * C),
    "w/((S+C)*S)":    w / ((S + C) * S),
}

header = f"{'Formula':<16} {'gamma':>8} {'QUBO best':>10} {'True cost':>10} {'Regret%':>8} {'TrueOpt rank':>13} {'Spearman':>9} {'Elite in top13':>15}"
print(header)
print("-" * len(header))

for name, gamma in sorted(formulas.items(), key=lambda x: x[1]):
    Q, offset, meta = build_qubo(s, gamma=gamma)
    qubo_rank = qubo_ranking(Q, offset)

    # QUBO minimiser
    qubo_best_bs = qubo_rank[0][0]
    qubo_best_true_cost = true_cost_map[qubo_best_bs]

    # Surrogate regret
    regret = (qubo_best_true_cost - c_star) / (c_rand - c_star) * 100

    # Where does the true optimum land in the QUBO ranking?
    qubo_rank_map = {bs: i for i, (bs, _) in enumerate(qubo_rank)}
    true_opt_qubo_rank = qubo_rank_map[c_star_bs]

    # Spearman rank correlation
    all_bs = [bs for bs, _ in true_ranking]
    true_costs = [true_cost_map[bs] for bs in all_bs]
    qubo_vals  = [evaluate_qubo(Q, offset, [int(b) for b in bs]) for bs in all_bs]
    rho, _ = spearmanr(true_costs, qubo_vals)

    # How many of the elite schedules are in the QUBO's top 13?
    qubo_top13 = {bs for bs, _ in qubo_rank[:13]}
    elite_overlap = len(elite & qubo_top13)

    print(f"{name:<16} {gamma:>8.5f} {qubo_best_bs:>10} {qubo_best_true_cost:>10.1f} {regret:>7.1f}% {true_opt_qubo_rank:>13d} {rho:>9.4f} {elite_overlap:>10d}/13")

# Also show what happens with gamma = 0 (no overflow penalty, just delay + switches)
print()
Q0, off0, _ = build_qubo(s, gamma=0.0)
qr0 = qubo_ranking(Q0, off0)
qb0_bs = qr0[0][0]
qb0_tc = true_cost_map[qb0_bs]
regret0 = (qb0_tc - c_star) / (c_rand - c_star) * 100
rho0, _ = spearmanr(
    [true_cost_map[bs] for bs in all_bs],
    [evaluate_qubo(Q0, off0, [int(b) for b in bs]) for bs in all_bs],
)
print(f"{'gamma=0 (none)':<16} {'0.0':>8} {qb0_bs:>10} {qb0_tc:>10.1f} {regret0:>7.1f}% {'':>13} {rho0:>9.4f}")
