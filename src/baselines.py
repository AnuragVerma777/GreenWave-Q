"""
Baseline methods for the Quantum Traffic Gridlock challenge.

Required baselines per the problem statement:
  1. Best of four alternating schedules
  2. Per-slot greedy (serve the larger queue after arrivals, ties to NS)
  3. Exact enumeration of all 256 schedules
  4. Uniform-random expected cost
"""

from __future__ import annotations

from src.model import Scenario, score, enumerate_all


def alternating_schedules() -> list[str]:
    """Return the four alternating 8-bit schedules.

    These are schedules where each intersection alternates phases:
      0101|0101, 1010|1010, 0101|1010, 1010|0101
    """
    return ["01010101", "10101010", "01011010", "10100101"]


def best_alternating(scenario: Scenario) -> tuple[str, float]:
    """Return (bitstring, cost) of the best alternating schedule."""
    results = [(bs, score(scenario, bs)) for bs in alternating_schedules()]
    results.sort(key=lambda x: (x[1], x[0]))
    return results[0]


def greedy_schedule(scenario: Scenario) -> tuple[str, float]:
    """Per-slot greedy: serve the larger queue after arrivals.

    At each slot, independently for each intersection, choose NS (1) if
    NS queue >= EW queue after arrivals (ties to NS), else EW (0).

    For B, this greedy doesn't account for transfers from A since those
    depend on A's schedule which is being decided simultaneously. We
    simulate A first (greedy), compute transfers, then simulate B (greedy).
    """
    import math

    ns = scenario.num_slots
    cap = scenario.capacity
    arr = scenario.arrivals
    link_approach = scenario.link_approach
    link_fraction = scenario.link_fraction

    # Greedily decide A's schedule
    sched_a = []
    queue_a_ns = 0
    queue_a_ew = 0
    transfers_to_b = [0] * ns

    for t in range(ns):
        queue_a_ns += arr["A"]["NS"][t]
        queue_a_ew += arr["A"]["EW"][t]

        # Greedy: serve larger queue, ties to NS
        if queue_a_ns >= queue_a_ew:
            green = 1  # NS
        else:
            green = 0  # EW

        sched_a.append(green)

        # Serve
        if green == 1:
            dep = min(queue_a_ns, cap)
            queue_a_ns -= dep
            dep_ns, dep_ew = dep, 0
        else:
            dep = min(queue_a_ew, cap)
            queue_a_ew -= dep
            dep_ns, dep_ew = 0, dep

        # Transfer
        if link_approach == "NS":
            linked_dep = dep_ns
        else:
            linked_dep = dep_ew
        transfer = math.floor(link_fraction * linked_dep + 0.5)
        if t + 1 < ns:
            transfers_to_b[t + 1] = transfer

    # Greedily decide B's schedule
    sched_b = []
    queue_b_ns = 0
    queue_b_ew = 0

    for t in range(ns):
        queue_b_ns += arr["B"]["NS"][t]
        queue_b_ew += arr["B"]["EW"][t]

        # Add incoming transfers
        if link_approach == "NS":
            queue_b_ns += transfers_to_b[t]
        else:
            queue_b_ew += transfers_to_b[t]

        # Greedy: serve larger queue, ties to NS
        if queue_b_ns >= queue_b_ew:
            green = 1
        else:
            green = 0

        sched_b.append(green)

        # Serve
        if green == 1:
            dep = min(queue_b_ns, cap)
            queue_b_ns -= dep
        else:
            dep = min(queue_b_ew, cap)
            queue_b_ew -= dep

    bits = "".join(str(b) for b in sched_a) + "".join(str(b) for b in sched_b)

    # Score using the full model (not the greedy simulation)
    # to get the authoritative cost
    return bits, score(scenario, bits)


def exact_enumeration(scenario: Scenario) -> tuple[str, float, list[tuple[str, float]]]:
    """Enumerate all 256 schedules.

    Returns (best_bitstring, best_cost, sorted_list).
    """
    results = enumerate_all(scenario)
    return results[0][0], results[0][1], results


def uniform_random_cost(scenario: Scenario) -> float:
    """Expected cost under uniform random schedule selection."""
    results = enumerate_all(scenario)
    return sum(c for _, c in results) / len(results)


def run_all_baselines(scenario: Scenario) -> dict:
    """Run all baselines and return a summary dict."""
    # 1. Best alternating
    alt_bs, alt_cost = best_alternating(scenario)

    # 2. Greedy
    greedy_bs, greedy_cost = greedy_schedule(scenario)

    # 3. Exact enumeration
    best_bs, best_cost, all_results = exact_enumeration(scenario)

    # 4. Uniform random
    rand_cost = uniform_random_cost(scenario)

    return {
        "alternating": {"schedule": alt_bs, "cost": alt_cost},
        "greedy": {"schedule": greedy_bs, "cost": greedy_cost},
        "enumeration": {
            "best_schedule": best_bs,
            "best_cost": best_cost,
            "all_results": all_results,
        },
        "uniform_random": {"expected_cost": rand_cost},
    }
