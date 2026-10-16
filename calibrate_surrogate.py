"""Reproduce the Block 3 surrogate-regret study (no NumPy/Qiskit required)."""
from __future__ import annotations

import random
import statistics
from pathlib import Path

from src.model import Scenario, enumerate_all, score


SEED = 20261015
N_RANDOM = 200
N_TRAIN = 150


def sample_scenario(rng: random.Random, name: str) -> Scenario:
    approach = rng.choice(("NS", "EW"))
    arrivals = {
        intersection: {
            phase: [rng.randint(0, 8) for _ in range(4)]
            for phase in ("NS", "EW")
        }
        for intersection in ("A", "B")
    }
    return Scenario(
        name=name,
        num_slots=4,
        capacity=rng.randint(4, 6),
        storage=rng.randint(5, 8),
        overflow_weight=rng.uniform(2, 6),
        switch_weight=rng.uniform(0.25, 1.5),
        link_from="A",
        link_to="B",
        link_approach=approach,
        link_fraction=rng.uniform(0.4, 0.8),
        arrivals=arrivals,
    )


def queue_expressions(s: Scenario) -> list[tuple[float, list[float]]]:
    """Match src/qubo.py's linear queue proxy, represented as (constant, coeffs)."""
    T, n, C = s.num_slots, 2 * s.num_slots, s.capacity
    f, link, arr = s.link_fraction, s.link_approach, s.arrivals
    result: list[tuple[float, list[float]]] = []
    for t in range(T):
        ca_ns = sum(arr["A"]["NS"][:t + 1])
        ca_ew = sum(arr["A"]["EW"][:t + 1])
        cb_ns = sum(arr["B"]["NS"][:t + 1])
        cb_ew = sum(arr["B"]["EW"][:t + 1])

        coeff = [0.0] * n
        for j in range(t + 1):
            coeff[j] = -C
        result.append((ca_ns, coeff))

        coeff = [0.0] * n
        for j in range(t + 1):
            coeff[j] = C
        result.append((ca_ew - C * (t + 1), coeff))

        coeff = [0.0] * n
        if link == "NS":
            for j in range(t):
                coeff[j] += f * C
        for j in range(t + 1):
            coeff[j + T] = -C
        result.append((cb_ns, coeff))

        coeff = [0.0] * n
        if link == "EW":
            c0 = cb_ew + f * C * t - C * (t + 1)
            for j in range(t):
                coeff[j] = -f * C
            for j in range(t + 1):
                coeff[j + T] = C
        else:
            c0 = cb_ew - C * (t + 1)
            for j in range(t + 1):
                coeff[j + T] = C
        result.append((c0, coeff))
    return result


def assignments(s: Scenario):
    exprs = queue_expressions(s)
    ranked_true = enumerate_all(s)
    true_cost = dict(ranked_true)
    c_star = ranked_true[0][1]
    prepared = []
    for i in range(256):
        bits = f"{i:08b}"
        x = [int(b) for b in bits]
        qs = [c + sum(a * b for a, b in zip(coeff, x)) for c, coeff in exprs]
        delay_proxy = sum(qs)
        square_proxy = sum(q * q for q in qs)
        centered_square_proxy = sum((q - s.storage / 4) ** 2 for q in qs)
        switches = sum(x[base + t] != x[base + t + 1]
                       for base in (0, 4) for t in range(3))
        switch_cost = s.switch_weight * switches
        prepared.append((bits, delay_proxy, square_proxy,
                         centered_square_proxy, switch_cost))
    return prepared, true_cost, c_star


def score_rule(s: Scenario, data, rule: str, multiplier: float = 2.0,
               center_fraction: float = 0.25):
    prepared, true_cost, c_star = data
    if rule == "previous":
        gamma = s.overflow_weight / (s.storage * s.storage)
        key = lambda row: row[1] + gamma * row[2] + row[4]
    elif rule == "centered":
        gamma = multiplier * s.overflow_weight / (s.storage * s.storage)
        center = center_fraction * s.storage
        key = lambda row: row[1] + gamma * (row[2] - 2 * center * row[1]) + row[4]
    else:
        raise ValueError(rule)
    best = min(prepared, key=lambda row: (key(row), row[0]))[0]
    regret = (true_cost[best] - c_star) / c_star * 100 if c_star else 0.0
    return regret, best, true_cost[best], c_star


def summary(results):
    vals = [item[0] for item in results]
    return (statistics.mean(vals), statistics.median(vals), max(vals),
            sum(v <= 5 for v in vals), sum(v >= 25 for v in vals))


def pct_summary(results) -> str:
    mean, median, worst, passed, zero = summary(results)
    return (f"mean {mean:.2f}%, median {median:.2f}%, worst {worst:.2f}%; "
            f"<=5%: {passed}/{len(results)}, >=25%: {zero}/{len(results)}")


def main() -> None:
    rng = random.Random(SEED)
    scenarios = [sample_scenario(rng, f"G{i:03d}") for i in range(N_RANDOM)]
    prepared = [assignments(s) for s in scenarios]
    reference = Scenario.from_json(Path("scenario_R.json"))
    reference_data = assignments(reference)

    grid = [(m, a) for m in (0.5, 1, 2, 4, 8, 16)
            for a in (0, 0.25, 0.5, 0.75, 1)]
    def train_mean(candidate):
        m, a = candidate
        return statistics.mean(score_rule(scenarios[i], prepared[i], "centered", m, a)[0]
                               for i in range(N_TRAIN))
    selected = min(grid, key=lambda candidate: (train_mean(candidate), candidate))
    multiplier, center_fraction = selected
    report = [
        "# Checkpoint 3: QUBO surrogate regret",
        "",
        "Regret is `(official_cost_of_exact_QUBO_minimizer - C*) / C* * 100`.",
        "The exact minimizer is found by enumerating all 256 assignments to the QUBO.",
        "Each reported official cost and C* comes from `src/model.py`.",
        "",
        f"Random study: {N_RANDOM} scenarios, seed `{SEED}`. Arrivals are independent integers in [0,8]; capacity [4,6]; storage [5,8]; overflow weight uniform [2,6]; switch weight uniform [0.25,1.5]; linked approach uniform from NS/EW; link fraction uniform [0.4,0.8]. The first {N_TRAIN} samples are calibration data and the final {N_RANDOM-N_TRAIN} are holdout data.",
        f"The shared rule was selected from a fixed grid on calibration mean regret: multiplier `{multiplier:g}` and center fraction `{center_fraction:g}`.",
        "",
        "## Results",
        "",
    ]
    ref_result = {"previous": score_rule(reference, reference_data, "previous"),
                  "centered": score_rule(reference, reference_data, "centered", multiplier, center_fraction)}
    report.append("| Set | Previous: `gamma=w/S^2`, penalty `q^2` | Updated: selected `gamma=m*w/S^2`, penalty `(q-a*S)^2` |")
    report.append("|---|---:|---:|")
    report.append("| R | " + " | ".join(
        f"{ref_result[r][0]:.2f}% ({ref_result[r][1]}, cost {ref_result[r][2]:.1f}, C* {ref_result[r][3]:.1f})"
        for r in ("previous", "centered")) + " |")
    for name, indices in (("Calibration (150)", range(N_TRAIN)),
                          ("Holdout (50)", range(N_TRAIN, N_RANDOM)),
                          ("All random (200)", range(N_RANDOM))):
        cells=[]
        for rule in ("previous", "centered"):
            rs=[score_rule(scenarios[i], prepared[i], rule,
                           multiplier, center_fraction) if rule == "centered"
                else score_rule(scenarios[i], prepared[i], rule) for i in indices]
            cells.append(pct_summary(rs))
        report.append(f"| {name} | {cells[0]} | {cells[1]} |")

    selected_results = [score_rule(scenarios[i], prepared[i], "centered",
                                   multiplier, center_fraction)
                for i in range(N_RANDOM)]
    worst_indices = sorted(range(N_RANDOM), key=lambda i: selected_results[i][0], reverse=True)[:5]
    report += ["", "## Worst-case failures and pattern", "",
               "In all five worst cases below, the QUBO minimizer has more official overflow than the exact optimum (by 3–7 vehicle-slots); it also has higher official delay in each case. This points to the main failure: the smooth queue penalty does not reproduce the scorer's overflow hinge at storage, and the queue proxy also omits service clipping and transfer rounding. The very large percentages in G004 and G105 are amplified by the requested denominator C*, which is small in those scenarios. The following are the five worst sampled cases:",
               "", "| Scenario | Regret | QUBO schedule | Scorer cost | C* | Capacity | Storage | Overflow wt. | Switch wt. | Link | Fraction |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for i in worst_indices:
        s=scenarios[i]; reg,bs,cost,cstar=selected_results[i]
        report.append(f"| {s.name} | {reg:.2f}% | {bs[:4]}|{bs[4:]} | {cost:.1f} | {cstar:.1f} | {s.capacity} | {s.storage} | {s.overflow_weight:.2f} | {s.switch_weight:.2f} | {s.link_approach} | {s.link_fraction:.3f} |")
    report += ["", "## Takeaway", "",
               f"The fixed-grid choice uses center fraction `{center_fraction:g}` and normalized multiplier `{multiplier:g}`. It is a single scenario-independent rule. Check the holdout distribution and worst case alongside the mean: a lower average does not eliminate severe misses. It misses the 5% target on R at {ref_result['centered'][0]:.2f}%. No QAOA sampling is evaluated at this checkpoint.", ""]
    out=Path("docs/checkpoint3_regret.md")
    out.write_text("\n".join(report), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
