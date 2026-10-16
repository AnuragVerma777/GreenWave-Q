"""QAOA execution and artifact generation for the 8-bit traffic QUBO."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.model import Scenario, score
from src.qubo import build_qubo, evaluate_qubo, qubo_to_dict

DEFAULT_P = 1
MAX_EVALUATIONS = 300
SHOTS = 4096
BIT_ORDER = "A1,A2,A3,A4,B1,B2,B3,B4 (left-to-right schedule order)"


def qubo_to_ising(Q: np.ndarray, offset: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (constant, h, J) for E(z)=constant+sum h_i z_i+sum J_ij z_i z_j.

    Q follows x.T @ Q @ x + offset with upper-triangular pair coefficients.
    x_i=(1-z_i)/2, where z_i=+1 for x_i=0.
    """
    n = Q.shape[0]
    constant = float(offset + np.trace(Q) / 2.0)
    h = -np.diag(Q).astype(float) / 2.0
    J = np.zeros_like(Q, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            q = float(Q[i, j] + Q[j, i])
            if q:
                constant += q / 4.0
                h[i] -= q / 4.0
                h[j] -= q / 4.0
                J[i, j] = q / 4.0
    return constant, h, J


def verify_ising_all_assignments(Q: np.ndarray, offset: float) -> float:
    """Exhaustively check QUBO/Ising energy parity; return maximum absolute error."""
    constant, h, J = qubo_to_ising(Q, offset)
    maximum = 0.0
    for mask in range(1 << Q.shape[0]):
        x = np.array([(mask >> i) & 1 for i in range(Q.shape[0])], dtype=float)
        z = 1.0 - 2.0 * x
        e_ising = constant + h @ z + sum(J[i, j] * z[i] * z[j]
                                                for i in range(len(z)) for j in range(i + 1, len(z)))
        maximum = max(maximum, abs(e_ising - evaluate_qubo(Q, offset, x)))
    if maximum > 1e-8:
        raise AssertionError(f"Ising conversion mismatch: max error {maximum}")
    return maximum


def _build_circuit(p: int, h: np.ndarray, J: np.ndarray, measured: bool = False):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n = len(h)
    gammas = ParameterVector("gamma", p)
    betas = ParameterVector("beta", p)
    circuit = QuantumCircuit(n, n if measured else 0)
    circuit.h(range(n))
    for layer in range(p):
        g, b = gammas[layer], betas[layer]
        for i, hi in enumerate(h):
            if hi:
                circuit.rz(2 * g * hi, i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j]:
                    circuit.rzz(2 * g * J[i, j], i, j)
        for i in range(n):
            circuit.rx(2 * b, i)
    if measured:
        # q_i is measured to classical bit i; printed strings are c[n-1]...c[0].
        circuit.measure(range(n), range(n))
    return circuit, gammas, betas


def _schedule_from_count_key(key: str) -> str:
    # Qiskit prints classical bit n-1 on the left; schedule variables run x0..x7.
    compact = key.replace(" ", "")
    return compact[::-1]


def run_qaoa(scenario: Scenario, seed: int, out_dir: str | Path) -> dict:
    """Optimize the scenario QUBO and write all requested QAOA artifacts."""
    from qiskit import qasm3
    from qiskit_aer import AerSimulator

    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Q, offset, metadata = build_qubo(scenario)
    ising_error = verify_ising_all_assignments(Q, offset)
    ising_constant, h, J = qubo_to_ising(Q, offset)
    p = DEFAULT_P
    # Scenario-independent normalization: divide the Ising cost Hamiltonian by its standard
    # deviation under uniform sampling, sqrt(sum h_i^2 + sum J_ij^2), so the QAOA angle gamma
    # is O(1) for every scenario. The QUBO itself (qubo.json) is unchanged.
    hamiltonian_scale = float(np.sqrt(np.sum(h ** 2) + np.sum(J ** 2)))
    if hamiltonian_scale < 1e-12:
        hamiltonian_scale = 1.0
    h_circuit, J_circuit = h / hamiltonian_scale, J / hamiltonian_scale
    base, gammas, betas = _build_circuit(p, h_circuit, J_circuit)

    # Exact statevector objective for a deterministic, shot-free COBYLA search.
    energies = np.empty(1 << len(h), dtype=float)
    for idx in range(len(energies)):
        bits = [(idx >> q) & 1 for q in range(len(h))]
        energies[idx] = evaluate_qubo(Q, offset, bits)
    evaluations = 0
    best = {"energy": float("inf"), "theta": None}
    rng = np.random.default_rng(seed)
    theta0 = np.r_[np.linspace(0.45, 0.45, p), np.linspace(0.25, 0.25, p)]
    theta0 += rng.uniform(-0.01, 0.01, len(theta0))
    last_value = float(np.mean(energies))

    def objective(theta):
        nonlocal evaluations, last_value
        if evaluations >= MAX_EVALUATIONS:
            return last_value
        params = {gammas[k]: float(theta[k]) for k in range(p)}
        params.update({betas[k]: float(theta[p + k]) for k in range(p)})
        state = base.assign_parameters(params)
        from qiskit.quantum_info import Statevector
        probs = Statevector.from_instruction(state).probabilities()
        last_value = float(probs @ energies)
        evaluations += 1
        if last_value < best["energy"]:
            best["energy"] = last_value
            best["theta"] = np.array(theta, dtype=float).copy()
        return last_value

    result = minimize(objective, theta0, method="COBYLA",
                      options={"maxiter": MAX_EVALUATIONS, "rhobeg": 0.3, "tol": 1e-4})
    theta = best["theta"] if best["theta"] is not None else result.x
    params = {gammas[k]: float(theta[k]) for k in range(p)}
    params.update({betas[k]: float(theta[p + k]) for k in range(p)})
    measured, _, _ = _build_circuit(p, h_circuit, J_circuit, measured=True)
    # Bind the same parameter values by parameter name for the measured circuit.
    measured_bind = {param: (float(theta[int(param.name.split("[")[1][:-1])])
                              if param.name.startswith("gamma") else
                              float(theta[p + int(param.name.split("[")[1][:-1])]))
                     for param in measured.parameters}
    measured = measured.assign_parameters(measured_bind)

    sim = AerSimulator()
    counts = sim.run(measured, shots=SHOTS, seed_simulator=int(seed)).result().get_counts()
    schedule_counts: dict[str, int] = {}
    for key, count in counts.items():
        schedule = _schedule_from_count_key(key)
        schedule_counts[schedule] = schedule_counts.get(schedule, 0) + int(count)
    if sum(schedule_counts.values()) != SHOTS:
        raise AssertionError("shot count mismatch")
    ranked_samples = sorted(schedule_counts.items(), key=lambda row: (-row[1], row[0]))
    most_frequent = ranked_samples[0][0]
    lowest_qubo = min(schedule_counts, key=lambda b: (evaluate_qubo(Q, offset, [int(x) for x in b]), b))
    hybrid = min(schedule_counts, key=lambda b: (score(scenario, b), b))

    (out / "qubo.json").write_text(json.dumps(qubo_to_dict(Q, offset, metadata), indent=2))
    (out / "counts.json").write_text(json.dumps(dict(sorted(schedule_counts.items())), indent=2))
    (out / "circuit.qasm").write_text(qasm3.dumps(measured))
    parameters = {"p": p, "hamiltonian_scale": hamiltonian_scale,
                  "gamma": [float(theta[k]) for k in range(p)],
                  "beta": [float(theta[p+k]) for k in range(p)]}
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2))
    runtime_seconds = time.perf_counter() - started
    schedule_doc = {
        "bits": most_frequent,
        "bit_order": BIT_ORDER,
        "raw_qaoa": {
            "most_frequent": {"bits": most_frequent, "count": schedule_counts[most_frequent],
                              "qubo_energy": evaluate_qubo(Q, offset, [int(x) for x in most_frequent])},
            "lowest_qubo_energy": {"bits": lowest_qubo, "count": schedule_counts[lowest_qubo],
                                   "qubo_energy": evaluate_qubo(Q, offset, [int(x) for x in lowest_qubo])},
        },
        "hybrid": {"label": "Best official-cost schedule among sampled bitstrings",
                   "bits": hybrid, "count": schedule_counts[hybrid], "official_cost": score(scenario, hybrid)},
        "optimizer": {"name": "COBYLA", "seed": int(seed), "evaluations": evaluations,
                      "initialization": "linear layer ramp with seeded uniform jitter ±0.01",
                      "evaluation_cap": MAX_EVALUATIONS, "success": bool(result.success),
                      "message": str(result.message)},
        "qaoa": {"p": p, "p_default_rationale": "Fixed p=1; see docs/qaoa_ablation.md",
                 "ising_constant": ising_constant, "hamiltonian_scale": hamiltonian_scale, "ising_max_assignment_error": ising_error,
                 "shots": SHOTS, "seed": int(seed)},
    }
    (out / "schedule.json").write_text(json.dumps(schedule_doc, indent=2))
    schedule_doc["qaoa"]["runtime_seconds"] = runtime_seconds
    return schedule_doc
