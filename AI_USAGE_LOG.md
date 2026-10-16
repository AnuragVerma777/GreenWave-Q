# AI Usage Log

**QC@IITI Qiskit Fall Fest 2026 — Quantum Traffic Gridlock**

This document discloses all AI-assisted work per hackathon rules.

---

## Day 1 — 2026-10-15

### Block 1: Project Skeleton and Traffic Model

| What AI Generated | What I Verified |
|---|---|
| Project directory structure (`src/`, `scenarios/`, `tests/`, `docs/`, `outputs/`) | Reviewed structure matches deliverable requirements |
| `scenarios/R.json` — scenario R in JSON format | Verified all parameters match the problem statement PDF |
| `src/model.py` — traffic simulation (queues, arrivals, service, transfers, delay/overflow/switches) | **Verified against 4 official test vectors: 1010\|1010→116.0, 1111\|1111→146.0, 0000\|0000→237.0, 0111\|1001→92.5 — all match exactly** |
| `src/baselines.py` — four required classical baselines | Reviewed logic; verified outputs on scenario R |
| `src/metrics.py` — lift, elite enrichment, surrogate regret | Reviewed formulas match problem statement definitions |
| `src/qubo.py` — placeholder stub | N/A (not yet implemented) |
| `run.py` — CLI entry point | Tested end-to-end on scenario R |
| `requirements.txt`, `README.md` | Reviewed content |

**Model design decisions:**
- The A→B link transfers vehicles to B's queue on the linked approach (e.g., link on EW → transfers go to B's EW queue). This was confirmed by matching all 4 test vectors.
- Transfer amount = `floor(fraction × departures + 0.5)` — standard rounding.
- Delay = sum of all remaining queues (both NS + EW) at both intersections after service, across all slots.
- Overflow = sum of `max(0, queue - storage)` for each queue at each intersection after service, across all slots.
- Switches = count of phase changes between adjacent slots per intersection. No pre-slot-1 switch cost.

### Block 2: Classical Baselines

| What AI Generated | What I Verified |
|---|---|
| `src/baselines.py` — four alternating schedules, per-slot greedy, exact enumeration, uniform random mean | Reviewed all baseline costs on scenario R; cross-checked greedy logic (ties→NS); confirmed C*=72.0 at 1110\|1010 |
| `src/metrics.py` — elite set computation (13 lowest-cost, ties by ascending bitstring) | Verified elite set membership and ordering against full enumeration |
| `compute_baselines.py` — summary script | Ran and reviewed output |

### Block 3: QUBO Derivation

| What AI Generated | What I Verified |
|---|---|
| `docs/derivation.md` — scenario-independent queue proxy, objective, coefficient expansion, approximation limits, and variable order | Checked equations against `src/qubo.py`; documented unclipped linear service and fractional-transfer approximations, plus the quadratic soft congestion surrogate for the scorer's clipped overflow |
| `outputs/R/qubo.json` — dense and sparse upper-triangular QUBO, offset, mapping, gamma, and auxiliary count | Generated coefficients from the same documented expansion for scenario R; 8 variables and 0 auxiliaries |
| `calibrate_surrogate.py`, `docs/checkpoint3_regret.md` — seeded 200-scenario study and regret/failure report | Exact QUBO minimizers exhaustively ranked over 256 assignments; official costs and optima computed with the traffic scorer; first 150 scenarios used to select one shared rule, last 50 held out |
| `improvement_pass.py`, updated `docs/checkpoint3_regret.md` — Spearman/elite diagnostics, failure analysis, and comparison of two global quadratic centers on a fresh 100-scenario holdout (seed `77441931`) | Compared current and two candidates on identical scenarios; exact ranks from all 256 schedules; R excluded from calibration; no H data used; no QAOA run |

The coefficient multiplier `gamma = 2 * overflow_weight / storage^2` was selected on the earlier 150-scenario calibration sample (seed `20261015`). On the new 100-scenario holdout, the predeclared center candidates `storage/4`, `storage/2`, and `storage` were compared; `storage/2` won by mean regret, P90, worst regret, then correlation. Neither R nor H was used to tune these constants. The selected rule improves mean and P90 regret over the prior `storage/4` rule on that holdout, but R regret remains 10.42% and worst holdout regret is 47.11%. The exact clipped scorer remains the schedule evaluator. Local coefficient export used a dependency-free calculation because NumPy is not installed in this workspace; the repository's normal builder still depends on NumPy.

---

*Log maintained as required by hackathon AI-use disclosure rules.*
