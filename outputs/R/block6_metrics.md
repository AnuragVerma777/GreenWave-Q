# Block 6 metrics — R

Seeds: 1, 7, 21, 42, 101. Each raw sample set contains 4,096 shots.
C* = 72.000; C_rand = 151.250.

## Method comparison

| Method | Schedule | Official cost |
|---|---:|---:|
| Best alternating | 10101010 | 116.000 |
| Greedy (ties to NS) | 01111001 | 92.500 |
| Exact enumeration | 11101010 | 72.000 |
| Uniform-random mean | — | 151.250 |
| Exact QUBO minimizer (officially scored) | 11101001 | 79.500 |
| QAOA raw samples (mean E_Q, 5 seeds) | 4,096 samples/seed | 110.007 |
| Hybrid (best official cost among all sampled bitstrings) | 11101010 | 72.000 |

## Per-seed raw-QAOA metrics

| Seed | E_Q | Lift | Lift clipped | Elite shots | Elite enrichment | Raw mode | Raw mode cost | Hybrid schedule | Hybrid cost | Evaluations | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 110.410 | 0.51533 | 0.51533 | 704 | 3.3846 | 11100110 | 99.500 | 11101010 | 72.000 | 63/300 | 0.214 |
| 7 | 110.025 | 0.52019 | 0.52019 | 699 | 3.3606 | 10111000 | 101.500 | 11101010 | 72.000 | 74/300 | 0.255 |
| 21 | 110.096 | 0.51929 | 0.51929 | 669 | 3.2163 | 11111000 | 102.500 | 11101010 | 72.000 | 62/300 | 0.205 |
| 42 | 109.601 | 0.52554 | 0.52554 | 725 | 3.4856 | 11100100 | 83.500 | 11101010 | 72.000 | 58/300 | 0.214 |
| 101 | 109.902 | 0.52174 | 0.52174 | 697 | 3.3510 | 11101001 | 79.500 | 11101010 | 72.000 | 55/300 | 0.182 |

**Minimum lift clipped to [0,1]: 0.51533.**
Elite enrichment is each seed's fraction of shots in the 13-schedule elite divided by 13/256. The hybrid row reports the best official-cost schedule found among all sampled bitstrings across these seeds; it is distinct from the raw QAOA mean.
