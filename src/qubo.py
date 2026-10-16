"""
QUBO construction for the Quantum Traffic Gridlock challenge.

Surrogate cost:
    C_tilde(x) = SUM q_t(x) + gamma * SUM (q_t(x) - S/2)^2
                 + w_sw * switches(x)

Each queue expression q_t is LINEAR in the 8 decision bits x, so its square
is quadratic, keeping everything within a standard QUBO.

Components:
  1. Delay (linear):    SUM q_t(x)  — cumulative remaining vehicles.
  2. Overflow surrogate (quadratic): gamma * SUM (q_t(x)-S/2)^2 — a
     centered smooth penalty approximating the true max(0, q-S) overflow.
  3. Switching (quadratic): w_sw * SUM (x_t XOR x_{t+1})  — exact.

The queue dynamics drop the min(queue, capacity) clipping:
    q_t = cumulative_arrivals + cumulative_transfers - capacity * (green slots)
Transfers are approximated as fraction * capacity * green_indicator(linked approach).

gamma formula:  2 * w_ovf / S^2   (documented, scenario-independent).
"""

from __future__ import annotations

import json
import numpy as np

from src.model import Scenario, score


# Canonical variable labels
VAR_LABELS = [
    "A_slot1", "A_slot2", "A_slot3", "A_slot4",
    "B_slot1", "B_slot2", "B_slot3", "B_slot4",
]


# ---------------------------------------------------------------------------
# Queue linear-coefficient builder
# ---------------------------------------------------------------------------

def _queue_coefficients(
    scenario: Scenario,
) -> list[tuple[str, float, np.ndarray]]:
    """Compute the linear expression for every queue-slot pair.

    Each queue at the end of slot *t* is written as:
        q_t = c0  +  c[0]*x0 + c[1]*x1 + ... + c[7]*x7

    Decision bits:  x0–x3 = A slots 1-4,  x4–x7 = B slots 1-4.
    x_k = 1 ⟹ NS green,  x_k = 0 ⟹ EW green.

    Returns a list of (label, c0, coeff_array_of_8).
    """
    C = scenario.capacity
    f = scenario.link_fraction
    link = scenario.link_approach          # "NS" or "EW"
    arr = scenario.arrivals
    T = scenario.num_slots                 # 4
    n = 2 * T                              # 8 decision bits

    result: list[tuple[str, float, np.ndarray]] = []

    for t in range(T):
        # Cumulative external arrivals through slot t (inclusive, 0-indexed)
        cum_a_ns = float(sum(arr["A"]["NS"][: t + 1]))
        cum_a_ew = float(sum(arr["A"]["EW"][: t + 1]))
        cum_b_ns = float(sum(arr["B"]["NS"][: t + 1]))
        cum_b_ew = float(sum(arr["B"]["EW"][: t + 1]))

        # ---- A  NS ----
        # Service at A-NS in slot s: C * x_s  (NS green ⟺ x_s = 1)
        c0 = cum_a_ns
        c = np.zeros(n)
        for s in range(t + 1):
            c[s] = -C
        result.append((f"A_NS_t{t}", c0, c))

        # ---- A  EW ----
        # Service at A-EW in slot s: C * (1 - x_s)
        # q = cum_ew - C*(t+1) + C * sum(x_0..x_t)
        c0 = cum_a_ew - C * (t + 1)
        c = np.zeros(n)
        for s in range(t + 1):
            c[s] = C
        result.append((f"A_EW_t{t}", c0, c))

        # ---- B  NS ----
        c0 = cum_b_ns
        c = np.zeros(n)
        if link == "NS":
            # Transfers from A-NS departures arrive at B-NS.
            # At A slot s, NS departs C * x_s, transfer = f*C*x_s,
            # arrives at B at slot s+1.  Cumul. transfers at slot t:
            #   f*C * sum(x_0 .. x_{t-1})
            for s in range(t):        # x_0 … x_{t-1}
                c[s] += f * C
        # Service at B-NS in slot s: C * x_{s+T}
        for s in range(t + 1):
            c[s + T] = -C
        result.append((f"B_NS_t{t}", c0, c))

        # ---- B  EW ----
        if link == "EW":
            # Transfers from A-EW departures arrive at B-EW.
            # At A slot s, EW departs C*(1-x_s), transfer = f*C*(1-x_s),
            # arrives at B at slot s+1.
            # Cumul. transfers at t: f*C*t  -  f*C * sum(x_0..x_{t-1})
            #
            # Service at B-EW in slot s: C*(1 - x_{s+T})
            # Cumul. service: C*(t+1) - C*sum(x_T..x_{T+t})
            c0 = cum_b_ew + f * C * t - C * (t + 1)
            c = np.zeros(n)
            for s in range(t):        # transfer coeff on A bits
                c[s] = -f * C
            for s in range(t + 1):    # service coeff on B bits
                c[s + T] = C
        else:
            # No transfers to B-EW
            c0 = cum_b_ew - C * (t + 1)
            c = np.zeros(n)
            for s in range(t + 1):
                c[s + T] = C
        result.append((f"B_EW_t{t}", c0, c))

    return result


# ---------------------------------------------------------------------------
# QUBO builder
# ---------------------------------------------------------------------------

def build_qubo(
    scenario: Scenario,
    gamma: float | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Construct the QUBO matrix for traffic signal optimisation.

    Args:
        scenario: A traffic scenario.
        gamma:    Overflow-surrogate coefficient.  Default: ``2*w_ovf / S²``.

    Returns:
        ``(Q, offset, metadata)``

        * **Q** — ``(n, n)`` upper-triangular QUBO matrix  (``n = 8``).
        * **offset** — scalar constant.
        * **metadata** — variable mapping, gamma value, etc.

    Objective:  minimise  ``x^T Q x + offset``  over  ``x ∈ {0,1}^8``.
    """
    T = scenario.num_slots
    n = 2 * T                          # 8 decision bits
    Q = np.zeros((n, n))
    offset = 0.0

    w_ovf = scenario.overflow_weight
    w_sw  = scenario.switch_weight
    S     = scenario.storage

    if gamma is None:
        gamma = 2.0 * w_ovf / (S * S)

    # ---- queue-based terms ----
    queues = _queue_coefficients(scenario)

    for _label, c0, c in queues:
        # (a) Delay  —  q_t = c0 + Σ c_k x_k  (linear)
        offset += c0
        for k in range(n):
            Q[k, k] += c[k]

        # (b) Overflow surrogate — γ(q_t - S/2)² (quadratic soft hinge).
        #     The common center S/2 was selected from a fixed candidate set;
        #     neither it nor gamma depends on an individual scenario.
        #                + 2 Σ_{j<k} c_j c_k  x_j x_k
        #     (using x_k² = x_k for binary variables)
        shifted_c0 = c0 - S / 2.0
        offset += gamma * shifted_c0 * shifted_c0
        for k in range(n):
            Q[k, k] += gamma * (2.0 * shifted_c0 * c[k] + c[k] * c[k])
        for j in range(n):
            for k in range(j + 1, n):
                Q[j, k] += gamma * 2.0 * c[j] * c[k]

    # ---- switching cost (exact) ----
    # switch(t,t+1) = x_t + x_{t+1} − 2 x_t x_{t+1}
    for base in (0, T):                       # A then B
        for s in range(T - 1):
            i, j = base + s, base + s + 1
            Q[i, i] += w_sw
            Q[j, j] += w_sw
            Q[i, j] -= 2.0 * w_sw

    # ---- metadata ----
    metadata = {
        "num_variables": n,
        "num_auxiliary": 0,
        "variable_labels": VAR_LABELS[:n],
        "variable_mapping": {str(k): VAR_LABELS[k] for k in range(n)},
        "variable_to_bit": {VAR_LABELS[k]: k for k in range(n)},
        "auxiliary_variables": [],
        "gamma": gamma,
        "gamma_formula": f"2 * w_ovf / S^2 = 2 * {w_ovf} / {S}^2",
        "components": [
            "delay (linear in bits)",
            "overflow_surrogate (gamma * (queue - storage/2)^2)",
            "switching (exact quadratic)",
        ],
    }
    return Q, offset, metadata


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_qubo(Q: np.ndarray, offset: float, x) -> float:
    """Evaluate  ``x^T Q x + offset``  for binary vector *x*."""
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + offset)


def qubo_best_schedule(
    Q: np.ndarray, offset: float, n: int = 8,
) -> tuple[str, float]:
    """Brute-force the QUBO minimiser over all 2^n assignments.

    Returns ``(bitstring, qubo_value)``.
    """
    best_bs, best_val = None, float("inf")
    for i in range(1 << n):
        bits = format(i, f"0{n}b")
        x = np.array([int(b) for b in bits], dtype=float)
        val = evaluate_qubo(Q, offset, x)
        if val < best_val or (val == best_val and bits < best_bs):
            best_val, best_bs = val, bits
    return best_bs, best_val


def qubo_ranking(
    Q: np.ndarray, offset: float, n: int = 8,
) -> list[tuple[str, float]]:
    """Rank all 2^n assignments by QUBO value (ascending)."""
    results = []
    for i in range(1 << n):
        bits = format(i, f"0{n}b")
        x = np.array([int(b) for b in bits], dtype=float)
        val = evaluate_qubo(Q, offset, x)
        results.append((bits, val))
    results.sort(key=lambda r: (r[1], r[0]))
    return results


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def qubo_to_dict(Q: np.ndarray, offset: float, metadata: dict) -> dict:
    """Serialise the QUBO to a JSON-friendly dictionary."""
    n = Q.shape[0]
    sparse: dict[str, float] = {}
    for i in range(n):
        for j in range(i, n):
            if abs(Q[i, j]) > 1e-12:
                sparse[f"({i},{j})"] = round(float(Q[i, j]), 10)

    return {
        "qubo_matrix_sparse": sparse,
        "qubo_matrix_dense": [[round(float(v), 10) for v in row] for row in Q],
        "offset": round(float(offset), 10),
        **metadata,
    }


def save_qubo(
    Q: np.ndarray, offset: float, metadata: dict, path: str,
) -> None:
    """Write ``qubo.json``."""
    with open(path, "w") as f:
        json.dump(qubo_to_dict(Q, offset, metadata), f, indent=2)
