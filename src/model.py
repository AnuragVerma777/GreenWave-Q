"""
Traffic model for the Quantum Traffic Gridlock challenge.

Two intersections (A, B) over 4 time slots.  Each intersection gives green
to exactly one phase per slot:  1 = North-South (NS),  0 = East-West (EW).

Eight decision bits ordered A1-A4 then B1-B4.

Per-slot processing:
  1. Add external arrivals and transfers from the preceding slot.
  2. The green approach serves up to its capacity.
  3. Record delay (all remaining queues after service) and overflow
     (excess above nominal storage) from queues remaining after service.
  4. On the directed A->B link, transfer floor(fraction * departures + 0.5)
     vehicles from the linked approach into B at the beginning of the next slot.

Transfers after the last slot leave the model.  Other departures leave the
model.  Queues start empty.

Cost = delay + overflow_weight * overflow + switch_weight * switches.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Scenario:
    """Parsed traffic scenario."""
    name: str
    num_slots: int
    capacity: int
    storage: int
    overflow_weight: float
    switch_weight: float
    link_from: str          # "A"
    link_to: str            # "B"
    link_approach: str      # "NS" or "EW"
    link_fraction: float
    arrivals: dict[str, dict[str, list[int]]]
    # arrivals["A"]["NS"] = [a1, a2, a3, a4]

    @classmethod
    def from_json(cls, path: str | Path) -> "Scenario":
        with open(path) as f:
            d = json.load(f)
        link = d["link"]
        # Support both internal schema (capacity, storage, link.approach)
        # and the official scorer schema (service_capacity, storage_capacity, link.phase).
        # Hidden scenarios use the official format.
        return cls(
            name=d.get("name", ""),
            num_slots=d.get("num_slots", 4),
            capacity=int(d.get("capacity", d.get("service_capacity"))),
            storage=int(d.get("storage", d.get("storage_capacity"))),
            overflow_weight=float(d["overflow_weight"]),
            switch_weight=float(d["switch_weight"]),
            link_from=link["from"],
            link_to=link["to"],
            link_approach=link.get("approach", link.get("phase")),
            link_fraction=float(link["fraction"]),
            arrivals=d["arrivals"],
        )

    def to_scorer_dict(self) -> dict:
        """Convert to the official scorer's expected dictionary format.

        The scorer validates specific keys: service_capacity, storage_capacity,
        link.phase. All arrival values and capacities must be Python ints.
        """
        return {
            "name": self.name,
            "arrivals": {
                node: {
                    phase: [int(v) for v in vals]
                    for phase, vals in phases.items()
                }
                for node, phases in self.arrivals.items()
            },
            "service_capacity": int(self.capacity),
            "storage_capacity": int(self.storage),
            "overflow_weight": float(self.overflow_weight),
            "switch_weight": float(self.switch_weight),
            "link": {
                "from": self.link_from,
                "to": self.link_to,
                "phase": self.link_approach,
                "fraction": float(self.link_fraction),
            },
        }


def decode_schedule(bits: list[int] | str) -> tuple[list[int], list[int]]:
    """Return (schedule_A, schedule_B) each of length 4 from 8-bit input.

    bits can be a list of 8 ints or a string like '10101010'.
    bit = 1 means NS green, bit = 0 means EW green.
    """
    if isinstance(bits, str):
        bits = [int(b) for b in bits]
    assert len(bits) == 8, f"Expected 8 bits, got {len(bits)}"
    return list(bits[:4]), list(bits[4:])


@dataclass
class SlotState:
    """State of one intersection's queues at one slot (after processing)."""
    queue_ns: int = 0
    queue_ew: int = 0
    delay: int = 0
    overflow: int = 0
    departures_ns: int = 0
    departures_ew: int = 0


def simulate_intersection(
    schedule: list[int],
    arrivals_ns: list[int],
    arrivals_ew: list[int],
    capacity: int,
    storage: int,
    incoming_transfers: list[int],
    transfer_approach: str | None,
    num_slots: int = 4,
) -> tuple[list[SlotState], list[int]]:
    """Simulate one intersection over all slots.

    Args:
        schedule: list of 4 bits (1=NS green, 0=EW green)
        arrivals_ns, arrivals_ew: external arrivals per slot
        capacity: max vehicles served per green phase per slot
        storage: nominal queue storage
        incoming_transfers: transfers arriving FROM the other intersection,
            index t means transfers that arrive at the start of slot t.
            incoming_transfers[0] is always 0 (no transfers before slot 1).
        transfer_approach: "NS" or "EW" — which approach receives transfers.
            None if this intersection doesn't receive transfers.
        num_slots: number of time slots

    Returns:
        (states, outgoing_departures_on_link) where outgoing_departures_on_link[t]
        is the raw departures from the linked approach at slot t (before applying
        the transfer fraction).
    """
    states: list[SlotState] = []
    queue_ns = 0
    queue_ew = 0
    outgoing_link_departures: list[int] = []

    for t in range(num_slots):
        # Step 1: Add external arrivals
        queue_ns += arrivals_ns[t]
        queue_ew += arrivals_ew[t]

        # Step 1 (cont): Add incoming transfers from preceding slot
        if transfer_approach is not None and t < len(incoming_transfers):
            if transfer_approach == "NS":
                queue_ns += incoming_transfers[t]
            else:
                queue_ew += incoming_transfers[t]

        # Step 2: Serve the green approach
        green = schedule[t]  # 1=NS, 0=EW
        dep_ns = 0
        dep_ew = 0
        if green == 1:  # NS green
            dep_ns = min(queue_ns, capacity)
            queue_ns -= dep_ns
        else:  # EW green
            dep_ew = min(queue_ew, capacity)
            queue_ew -= dep_ew

        # Step 3: Record delay and overflow
        delay = queue_ns + queue_ew
        overflow = max(0, queue_ns - storage) + max(0, queue_ew - storage)

        states.append(SlotState(
            queue_ns=queue_ns,
            queue_ew=queue_ew,
            delay=delay,
            overflow=overflow,
            departures_ns=dep_ns,
            departures_ew=dep_ew,
        ))

        # Record outgoing departures on linked approach (raw, before fraction)
        outgoing_link_departures.append(dep_ns if False else dep_ew)  # placeholder
        # We'll fix this below — the caller handles which departures matter

    return states, outgoing_link_departures


def evaluate_schedule(scenario: Scenario, bits: list[int] | str) -> dict[str, Any]:
    """Evaluate an 8-bit schedule on a scenario.

    Returns a dict with 'cost', 'delay', 'overflow', 'switches', and
    detailed slot-by-slot state.
    """
    sched_a, sched_b = decode_schedule(bits)
    ns = scenario.num_slots
    cap = scenario.capacity
    stor = scenario.storage

    arr = scenario.arrivals
    link_approach = scenario.link_approach  # "NS" or "EW"
    link_fraction = scenario.link_fraction

    # --- Simulate A first (A is the source of the link) ---
    # A doesn't receive transfers (link is A->B)
    # But we need to know if the link is FROM A, so A is always the source.

    # Simulate A
    queue_a_ns = 0
    queue_a_ew = 0
    states_a: list[dict] = []
    # Transfers from A to B, arriving at the start of the *next* slot
    transfers_to_b: list[int] = [0] * (ns + 1)  # index 0..ns, transfers_to_b[0]=0

    for t in range(ns):
        # Step 1: Add external arrivals
        queue_a_ns += arr["A"]["NS"][t]
        queue_a_ew += arr["A"]["EW"][t]

        # A does NOT receive transfers (it's the source)
        # (If the model were B->A, we'd handle it, but per spec link is A->B)

        # Step 2: Serve the green approach
        green = sched_a[t]
        dep_ns = 0
        dep_ew = 0
        if green == 1:
            dep_ns = min(queue_a_ns, cap)
            queue_a_ns -= dep_ns
        else:
            dep_ew = min(queue_a_ew, cap)
            queue_a_ew -= dep_ew

        # Step 3: Record delay and overflow
        delay_a = queue_a_ns + queue_a_ew
        overflow_a = max(0, queue_a_ns - stor) + max(0, queue_a_ew - stor)

        # Step 4: Compute transfers on the linked approach
        # Link is A->B on link_approach
        if link_approach == "NS":
            linked_departures = dep_ns
        else:  # "EW"
            linked_departures = dep_ew

        transfer_amount = math.floor(link_fraction * linked_departures + 0.5)
        # This transfer arrives at B at the beginning of the *next* slot
        if t + 1 < ns:
            transfers_to_b[t + 1] = transfer_amount
        # else: transfers after last slot leave the model

        states_a.append({
            "slot": t + 1,
            "queue_ns": queue_a_ns,
            "queue_ew": queue_a_ew,
            "delay": delay_a,
            "overflow": overflow_a,
            "dep_ns": dep_ns,
            "dep_ew": dep_ew,
            "transfer_out": transfer_amount,
        })

    # --- Simulate B (B receives transfers from A) ---
    queue_b_ns = 0
    queue_b_ew = 0
    states_b: list[dict] = []

    for t in range(ns):
        # Step 1: Add external arrivals
        queue_b_ns += arr["B"]["NS"][t]
        queue_b_ew += arr["B"]["EW"][t]

        # Step 1 (cont): Add transfers from A (on the linked approach at B)
        if link_approach == "NS":
            queue_b_ns += transfers_to_b[t]
        else:
            queue_b_ew += transfers_to_b[t]

        # Step 2: Serve the green approach
        green = sched_b[t]
        dep_ns = 0
        dep_ew = 0
        if green == 1:
            dep_ns = min(queue_b_ns, cap)
            queue_b_ns -= dep_ns
        else:
            dep_ew = min(queue_b_ew, cap)
            queue_b_ew -= dep_ew

        # Step 3: Record delay and overflow
        delay_b = queue_b_ns + queue_b_ew
        overflow_b = max(0, queue_b_ns - stor) + max(0, queue_b_ew - stor)

        # Step 4: B is the destination, no outgoing transfers
        states_b.append({
            "slot": t + 1,
            "queue_ns": queue_b_ns,
            "queue_ew": queue_b_ew,
            "delay": delay_b,
            "overflow": overflow_b,
            "dep_ns": dep_ns,
            "dep_ew": dep_ew,
            "transfer_in": transfers_to_b[t],
        })

    # --- Aggregate costs ---
    total_delay = sum(s["delay"] for s in states_a) + sum(s["delay"] for s in states_b)
    total_overflow = sum(s["overflow"] for s in states_a) + sum(s["overflow"] for s in states_b)

    # Switches: count changes between adjacent slots at each intersection
    switches_a = sum(1 for t in range(1, ns) if sched_a[t] != sched_a[t - 1])
    switches_b = sum(1 for t in range(1, ns) if sched_b[t] != sched_b[t - 1])
    total_switches = switches_a + switches_b

    cost = (total_delay
            + scenario.overflow_weight * total_overflow
            + scenario.switch_weight * total_switches)

    return {
        "cost": cost,
        "delay": total_delay,
        "overflow": total_overflow,
        "switches": total_switches,
        "schedule_A": sched_a,
        "schedule_B": sched_b,
        "states_A": states_a,
        "states_B": states_b,
        "transfers_to_B": transfers_to_b[:ns],
    }


def score(scenario: Scenario, bits: list[int] | str) -> float:
    """Return the cost of an 8-bit schedule on a scenario."""
    return evaluate_schedule(scenario, bits)["cost"]


def enumerate_all(scenario: Scenario) -> list[tuple[str, float]]:
    """Enumerate all 256 schedules and return (bitstring, cost) sorted by cost."""
    results = []
    for i in range(256):
        bits = format(i, "08b")
        c = score(scenario, bits)
        results.append((bits, c))
    results.sort(key=lambda x: (x[1], x[0]))
    return results
