"""
Metrics computation for the Quantum Traffic Gridlock challenge.

Provides functions to compute lift, elite enrichment, and other
evaluation metrics as defined in the problem statement.
"""

from __future__ import annotations

from collections import Counter

from src.model import Scenario, score, enumerate_all


def uniform_random_expected_cost(scenario: Scenario) -> float:
    """Mean cost over all 256 schedules (uniform random baseline)."""
    all_schedules = enumerate_all(scenario)
    return sum(c for _, c in all_schedules) / len(all_schedules)


def best_cost(scenario: Scenario) -> tuple[str, float]:
    """Return (bitstring, cost) of the optimal schedule by enumeration."""
    all_schedules = enumerate_all(scenario)
    return all_schedules[0]


def lift(c_rand: float, e_q: float, c_star: float) -> float:
    """Compute lift = (C_rand - E_Q) / (C_rand - C*).

    Clipped to [0, 1].
    """
    denom = c_rand - c_star
    if denom <= 0:
        return 0.0
    raw = (c_rand - e_q) / denom
    return max(0.0, min(1.0, raw))


def elite_set(scenario: Scenario, k: int = 13) -> set[str]:
    """Return the set of exactly k lowest-cost schedules.

    Ties broken by canonical ascending bitstring order.
    """
    all_schedules = enumerate_all(scenario)
    # Already sorted by (cost, bitstring) from enumerate_all
    return {bs for bs, _ in all_schedules[:k]}


def elite_enrichment(
    counts: dict[str, int],
    elite: set[str],
    total_shots: int = 4096,
    n_schedules: int = 256,
    k: int = 13,
) -> float:
    """Compute elite enrichment.

    enrichment = (fraction of shots in elite set) / (k / n_schedules)
    Random sampling has expected enrichment 1.
    """
    hits = sum(counts.get(bs, 0) for bs in elite)
    frac = hits / total_shots if total_shots > 0 else 0.0
    expected_frac = k / n_schedules
    return frac / expected_frac if expected_frac > 0 else 0.0


def mean_cost_from_counts(scenario: Scenario, counts: dict[str, int]) -> float:
    """Compute mean cost from a counts dictionary {bitstring: count}."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    weighted = sum(score(scenario, bs) * cnt for bs, cnt in counts.items())
    return weighted / total


def surrogate_regret(
    qubo_best_cost: float,
    true_best_cost: float,
    c_rand: float,
) -> float:
    """Surrogate regret = (QUBO_best_scorer_cost - C*) / (C_rand - C*).

    Lower is better. Full marks at <= 5%, zero at >= 25%.
    """
    denom = c_rand - true_best_cost
    if denom <= 0:
        return 0.0
    return (qubo_best_cost - true_best_cost) / denom
