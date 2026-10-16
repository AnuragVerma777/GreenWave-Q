import math
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector
from calibrate_surrogate import sample_scenario
from src.model import score
from src.qubo import build_qubo, evaluate_qubo
from src.qaoa import qubo_to_ising, _build_circuit

SEED, N, CAP = 20261018, 20, 300
VARIANTS = ("none", "maxcoef", "std")

def bits_of(i):
    return "".join(str((i >> q) & 1) for q in range(8))

def run(scn, variant, seed):
    Q, off, _ = build_qubo(scn)
    _, h, J = qubo_to_ising(Q, off)
    if variant == "maxcoef":
        s = max(float(abs(h).max()), float(abs(J).max()))
    elif variant == "std":
        s = math.sqrt(float((h ** 2).sum() + (J ** 2).sum()))
    else:
        s = 1.0
    s = s if s > 1e-12 else 1.0
    circ, g, b = _build_circuit(1, h / s, J / s)
    E = np.array([evaluate_qubo(Q, off, [(i >> q) & 1 for q in range(8)]) for i in range(256)])
    calls = 0
    best = [float("inf"), None]
    last = [float(E.mean())]
    def obj(theta):
        nonlocal calls
        if calls >= CAP:
            return last[0]
        st = circ.assign_parameters({g[0]: float(theta[0]), b[0]: float(theta[1])})
        last[0] = float(Statevector.from_instruction(st).probabilities() @ E)
        calls += 1
        if last[0] < best[0]:
            best[0], best[1] = last[0], np.array(theta, dtype=float)
        return last[0]
    rng = np.random.default_rng(seed)
    th0 = np.array([0.45, 0.25]) + rng.uniform(-0.01, 0.01, 2)
    res = minimize(obj, th0, method="COBYLA", options={"maxiter": CAP, "rhobeg": 0.3, "tol": 1e-4})
    th = best[1] if best[1] is not None else res.x
    st = circ.assign_parameters({g[0]: float(th[0]), b[0]: float(th[1])})
    return Statevector.from_instruction(st).probabilities(), calls

def main():
    rng = random.Random(SEED)
    scns = [sample_scenario(rng, f"S{i:02d}") for i in range(N)]
    stats = {v: {"lift": [], "enr": [], "calls": []} for v in VARIANTS}
    used = 0
    for k, scn in enumerate(scns):
        costs = np.array([score(scn, bits_of(i)) for i in range(256)])
        c_star, c_rand = float(costs.min()), float(costs.mean())
        if c_rand - c_star < 5:
            continue
        order = sorted(range(256), key=lambda i: (costs[i], bits_of(i)))
        elite = order[:13]
        used += 1
        for v in VARIANTS:
            probs, calls = run(scn, v, 1000 + k)
            stats[v]["lift"].append((c_rand - float(probs @ costs)) / (c_rand - c_star))
            stats[v]["enr"].append(float(probs[elite].sum()) / (13 / 256))
            stats[v]["calls"].append(calls)
    print(f"{used} scenarios used (seed {SEED}); ideal statevector, <= {CAP} evals")
    for v in VARIANTS:
        L = stats[v]["lift"]
        print(f"{v:8s} lift mean {np.mean(L):.4f} median {np.median(L):.4f} min {min(L):.4f} | enrichment mean {np.mean(stats[v]['enr']):.3f} | evals mean {np.mean(stats[v]['calls']):.0f}")

if __name__ == "__main__":
    main()

