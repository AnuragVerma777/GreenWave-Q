# QAOA depth ablation

Training bank: 16 independently generated scenarios (seed 20261016); random generator and ranges match `calibrate_surrogate.py`. Neither R nor H is part of this bank.
Both depths use seeded COBYLA, linear-ramp initialization, and at most 80 evaluations per scenario. Metric is normalized expected-QUBO-energy gap `(E_QAOA - E_min)/(E_max-E_min)` after optimization. Lower is better.

| p | Mean normalized gap | Median normalized gap | Mean optimizer time (s) | Maximum ZZ gates |
|---:|---:|---:|---:|---:|
| 1 | 0.09161 | 0.09060 | 0.1596 | 28 |
| 2 | 0.07907 | 0.06608 | 0.3076 | 56 |

Default selection: p=1. On this training bank, p=2 reduced normalized gap by 0.01254 while taking 1.93 times the mean optimizer time and doubling the maximum two-qubit gate count. We choose the smaller layer count for the production default; it is fixed and is not selected per scenario.
