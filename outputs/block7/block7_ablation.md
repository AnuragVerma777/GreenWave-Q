# Block 7 ablations and generalization

Five seeds: 1, 7, 21, 42, 101. Random bank: 4 scenarios generated with new seed `20261017` (R excluded from generation). QAOA expectation values are computed from the optimized ideal statevector; they are not finite-shot estimates.
Each ablation optimizer has an explicit 80-evaluation cap. Factors: p=1..4; gamma multiplier x0.5/x1/x2; initialization linear-ramp/random/fixed.

## Depth ablation (averaged over penalty, initialization, scenarios, and seeds)

| p | Expected official cost | Lift | Elite enrichment | Mean optimizer seconds |
|---:|---:|---:|---:|---:|
| 1 | 164.161 | 0.32547 | 2.5152 | 0.175 |
| 2 | 162.561 | 0.35088 | 2.6015 | 0.307 |
| 3 | 162.417 | 0.35202 | 2.6020 | 0.426 |
| 4 | 165.885 | 0.28003 | 1.9674 | 0.503 |

## Penalty-scale ablation

| Gamma multiplier | Expected official cost | Lift | Elite enrichment |
|---:|---:|---:|---:|
| x0.5 | 163.238 | 0.32871 | 2.5201 |
| x1 | 164.178 | 0.32223 | 2.3729 |
| x2 | 163.852 | 0.33036 | 2.3716 |

## Initialization ablation

| Initialization | Expected official cost | Lift | Elite enrichment |
|---|---:|---:|---:|
| linear-ramp | 160.697 | 0.39216 | 2.5977 |
| random | 170.562 | 0.18812 | 1.8795 |
| fixed | 160.009 | 0.40102 | 2.7874 |

## Scenario-level default-pipeline generalization

These results come from the normal `run.py` CLI at its unchanged defaults (p=1, gamma x1). Each scenario uses all five seeds and the usual 4,096 shots. Runtime shown is the QAOA stage runtime printed by the CLI.

| Scenario | Type | C* | C_rand | Mean E_Q | Mean lift | Minimum lift clipped [0,1] | Mean elite enrichment | Max evaluations | Max runtime/seed (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | unmodified default pipeline | 72.000 | 151.250 | 110.007 | 0.52042 | 0.51533 | 3.3596 | 74/300 | 0.279 |
| G001 | unmodified default pipeline | 333.541 | 377.283 | 357.872 | 0.44375 | 0.43554 | 3.0856 | 77/300 | 0.281 |
| G002 | unmodified default pipeline | 154.221 | 180.767 | 170.317 | 0.39366 | 0.38808 | 2.7529 | 77/300 | 0.330 |
| G003 | unmodified default pipeline | 58.774 | 97.072 | 76.982 | 0.52456 | 0.52108 | 3.1913 | 73/300 | 0.272 |
| G004 | unmodified default pipeline | 38.820 | 96.850 | 77.884 | 0.32683 | 0.32074 | 2.9279 | 96/300 | 0.322 |

### Default-pipeline results by scenario and seed

| Scenario | Seed | E_Q | Lift | Lift clipped | Elite enrichment | Evaluations | QAOA runtime (s) | CLI wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | 1 | 110.410 | 0.51533 | 0.51533 | 3.3846 | 63/300 | 0.217 | 1.955 |
| R | 7 | 110.025 | 0.52019 | 0.52019 | 3.3606 | 74/300 | 0.279 | 2.022 |
| R | 21 | 110.096 | 0.51929 | 0.51929 | 3.2163 | 62/300 | 0.222 | 1.872 |
| R | 42 | 109.601 | 0.52554 | 0.52554 | 3.4856 | 58/300 | 0.246 | 1.958 |
| R | 101 | 109.902 | 0.52174 | 0.52174 | 3.3510 | 55/300 | 0.191 | 1.891 |
| G001 | 1 | 358.126 | 0.43796 | 0.43796 | 3.0865 | 64/300 | 0.281 | 1.966 |
| G001 | 7 | 357.842 | 0.44444 | 0.44444 | 3.0048 | 55/300 | 0.209 | 1.946 |
| G001 | 21 | 358.231 | 0.43554 | 0.43554 | 3.1346 | 72/300 | 0.262 | 1.952 |
| G001 | 42 | 357.457 | 0.45324 | 0.45324 | 3.1394 | 77/300 | 0.271 | 2.042 |
| G001 | 101 | 357.704 | 0.44759 | 0.44759 | 3.0625 | 74/300 | 0.275 | 2.017 |
| G002 | 1 | 170.308 | 0.39397 | 0.39397 | 2.8221 | 71/300 | 0.232 | 1.977 |
| G002 | 7 | 170.465 | 0.38808 | 0.38808 | 2.6250 | 51/300 | 0.204 | 1.934 |
| G002 | 21 | 170.310 | 0.39392 | 0.39392 | 2.7452 | 64/300 | 0.235 | 1.909 |
| G002 | 42 | 170.192 | 0.39835 | 0.39835 | 2.7981 | 77/300 | 0.330 | 2.082 |
| G002 | 101 | 170.309 | 0.39396 | 0.39396 | 2.7740 | 47/300 | 0.182 | 1.967 |
| G003 | 1 | 77.115 | 0.52108 | 0.52108 | 3.2981 | 73/300 | 0.272 | 2.039 |
| G003 | 7 | 76.849 | 0.52803 | 0.52803 | 3.2163 | 72/300 | 0.235 | 1.896 |
| G003 | 21 | 76.937 | 0.52573 | 0.52573 | 3.1010 | 63/300 | 0.245 | 2.088 |
| G003 | 42 | 76.951 | 0.52537 | 0.52537 | 3.1490 | 71/300 | 0.234 | 1.902 |
| G003 | 101 | 77.057 | 0.52260 | 0.52260 | 3.1923 | 72/300 | 0.272 | 1.989 |
| G004 | 1 | 78.189 | 0.32157 | 0.32157 | 2.9038 | 59/300 | 0.210 | 1.932 |
| G004 | 7 | 77.873 | 0.32702 | 0.32702 | 3.0240 | 59/300 | 0.229 | 1.984 |
| G004 | 21 | 78.237 | 0.32074 | 0.32074 | 2.7452 | 50/300 | 0.209 | 1.978 |
| G004 | 42 | 77.489 | 0.33363 | 0.33363 | 3.1106 | 96/300 | 0.322 | 2.018 |
| G004 | 101 | 77.632 | 0.33118 | 0.33118 | 2.8558 | 64/300 | 0.255 | 2.116 |

Cumulative CLI wall time across all scenarios per seed: seed 1: 9.87s, seed 7: 9.78s, seed 21: 9.80s, seed 42: 10.00s, seed 101: 9.98s (maximum 10.00s).

Scenario H was not present at run time. The random bank is persisted alongside this report so runs can be reproduced. No scenario-specific parameters are selected.
