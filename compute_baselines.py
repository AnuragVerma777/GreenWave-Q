"""Compute and display all Block 2 baselines for scenario R."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import Scenario, score, enumerate_all
from src.baselines import best_alternating, greedy_schedule, alternating_schedules
from src.metrics import elite_set, uniform_random_expected_cost

s = Scenario.from_json("scenario_R.json")

# 1. All four alternating schedules
print("=" * 65)
print("1. ALTERNATING SCHEDULES")
print("=" * 65)
for bs in alternating_schedules():
    c = score(s, bs)
    print(f"   {bs[:4]}|{bs[4:]}  cost = {c:.1f}")
best_alt_bs, best_alt_cost = best_alternating(s)
print(f"   >>> Best alternating: {best_alt_bs[:4]}|{best_alt_bs[4:]} = {best_alt_cost:.1f}")

# 2. Per-slot greedy
print()
print("=" * 65)
print("2. PER-SLOT GREEDY (ties -> NS)")
print("=" * 65)
greedy_bs, greedy_cost = greedy_schedule(s)
print(f"   Schedule: {greedy_bs[:4]}|{greedy_bs[4:]}  cost = {greedy_cost:.1f}")

# 3. Exact enumeration
print()
print("=" * 65)
print("3. EXACT ENUMERATION (256 schedules)")
print("=" * 65)
all_results = enumerate_all(s)
c_star_bs, c_star = all_results[0]
print(f"   C* = {c_star:.1f}  at  {c_star_bs[:4]}|{c_star_bs[4:]}")

# 4. Uniform-random mean
print()
print("=" * 65)
print("4. UNIFORM-RANDOM MEAN COST")
print("=" * 65)
c_rand = uniform_random_expected_cost(s)
print(f"   C_rand = {c_rand:.4f}")

# 5. Elite set: 13 lowest-cost schedules, ties by ascending bitstring
print()
print("=" * 65)
print("5. ELITE SET (13 lowest-cost schedules)")
print("=" * 65)
# enumerate_all already sorts by (cost, bitstring)
elite_list = all_results[:13]
elite_bs_set = {bs for bs, _ in elite_list}
# verify with metrics.elite_set
elite_from_metrics = elite_set(s, k=13)
assert elite_bs_set == elite_from_metrics, "Elite set mismatch!"

print(f"   {'Rank':>4}  {'A|B':>9}  {'Cost':>8}")
print(f"   {'----':>4}  {'---------':>9}  {'--------':>8}")
for i, (bs, c) in enumerate(elite_list, 1):
    print(f"   {i:4d}  {bs[:4]}|{bs[4:]}  {c:8.1f}")

# Summary table
print()
print("=" * 65)
print("SUMMARY TABLE")
print("=" * 65)
print(f"   {'Baseline':<35} {'Schedule':<12} {'Cost':>8}")
print(f"   {'-'*35} {'-'*12} {'-'*8}")
print(f"   {'Best alternating':<35} {best_alt_bs[:4]+'|'+best_alt_bs[4:]:<12} {best_alt_cost:>8.1f}")
print(f"   {'Per-slot greedy (ties->NS)':<35} {greedy_bs[:4]+'|'+greedy_bs[4:]:<12} {greedy_cost:>8.1f}")
print(f"   {'C* (exact enumeration)':<35} {c_star_bs[:4]+'|'+c_star_bs[4:]:<12} {c_star:>8.1f}")
print(f"   {'C_rand (uniform random mean)':<35} {'—':<12} {c_rand:>8.2f}")
