# Block 6 metrics — H

Seeds: 1, 7, 21, 42, 101. Each raw sample set contains 4,096 shots.
C* = 234.000; C_rand = 300.281.

## Method comparison

| Method | Schedule | Official cost |
|---|---:|---:|
| Best alternating | 10100101 | 261.000 |
| Greedy (ties to NS) | 10100111 | 244.000 |
| Exact enumeration | 10000111 | 234.000 |
| Uniform-random mean | — | 300.281 |
| Exact QUBO minimizer (officially scored) | 10010110 | 236.000 |
| QAOA raw samples (mean E_Q, 5 seeds) | 4,096 samples/seed | 280.929 |
| Hybrid (best official cost among all sampled bitstrings) | 10000111 | 234.000 |

## Per-seed raw-QAOA metrics

| Seed | E_Q | Lift | Lift clipped | Elite shots | Elite enrichment | Raw mode | Raw mode cost | Hybrid schedule | Hybrid cost | Evaluations | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 281.204 | 0.28783 | 0.28783 | 502 | 2.4135 | 10000110 | 235.000 | 10000111 | 234.000 | 61/300 | 0.238 |
| 7 | 281.056 | 0.29006 | 0.29006 | 482 | 2.3173 | 01010110 | 262.000 | 10000111 | 234.000 | 82/300 | 0.258 |
| 21 | 280.418 | 0.29969 | 0.29969 | 500 | 2.4038 | 00110110 | 252.000 | 10000111 | 234.000 | 69/300 | 0.238 |
| 42 | 281.155 | 0.28857 | 0.28857 | 502 | 2.4135 | 01010110 | 262.000 | 10000111 | 234.000 | 64/300 | 0.212 |
| 101 | 280.814 | 0.29370 | 0.29370 | 485 | 2.3317 | 00110110 | 252.000 | 10000111 | 234.000 | 52/300 | 0.198 |

**Minimum lift clipped to [0,1]: 0.28783.**
Elite enrichment is each seed's fraction of shots in the 13-schedule elite divided by 13/256. The hybrid row reports the best official-cost schedule found among all sampled bitstrings across these seeds; it is distinct from the raw QAOA mean.
