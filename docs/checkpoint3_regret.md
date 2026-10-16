# Checkpoint 3 improvement pass

Fresh holdout: 100 scenarios, seed `77441931`. This seed was not used in earlier work. Arrivals are independent integers [0,8]; capacity [4,6]; storage [5,8]; overflow weight U[2,6]; switch weight U[0.25,1.5]; link approach NS/EW; fraction U[0.4,0.8]. All rules are evaluated on the same scenarios.

Regret is `(QUBO minimizer official cost - C*) / C*`; absolute regret is the cost difference. Exact QUBO minimizers and energy rankings enumerate all 256 schedules. Spearman uses average ranks for ties. Elite overlap compares the official 13 lowest-cost schedules with the QUBO's 13 lowest-energy schedules.

## Candidate definitions

| Rule | Penalty | Global constants |
|---|---|---|
| Current | `gamma * sum (q-aS)^2` | `gamma=2w/S²`, `a=0.25` |
| Candidate 1 | same form | `gamma=2w/S²`, `a=0.50` |
| Candidate 2 | same form | `gamma=2w/S²`, `a=1.00` |

The multiplier 2 in `gamma=2w/S²` was retained from the earlier fixed-grid calibration on 150 random scenarios (seed `20261015`). The three center fractions were fixed before this pass and compared on this fresh holdout; `a=0.50` won the requested tuple. Neither R nor H was used to tune constants. There is no per-scenario tuning. All candidates use no auxiliaries and remain at 8 variables.

## Fresh holdout results

| Rule | Mean regret | Median | P90 | Worst | Mean abs. | P90 abs. | Spearman mean / median / min | Elite mean / median / min (of 13) | R regret / abs. | R Spearman | R elite / 13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current S/4 | 6.16% | 1.49% | 19.77% | 47.11% | 5.00 | 15.06 | 0.8701 / 0.9033 / 0.5041 | 7.14 / 7.0 / 0 | 10.42% / 7.50 | 0.9547 | 6/13 |
| candidate S/2 | 5.58% | 1.19% | 17.73% | 47.11% | 4.64 | 14.78 | 0.8699 / 0.9013 / 0.4976 | 7.26 / 7.0 / 0 | 10.42% / 7.50 | 0.9493 | 4/13 |
| candidate S | 5.71% | 1.51% | 15.41% | 51.30% | 5.23 | 14.93 | 0.8568 / 0.8740 / 0.4795 | 6.90 / 7.0 / 0 | 33.33% / 24.00 | 0.9293 | 2/13 |

Selected by (mean regret, P90, worst, then highest mean correlation): **candidate S/2**.

## Five worst random scenarios

All cases below use the selected rule (candidate S/2) to characterize its failure modes. Arrival level is the total of the 16 external-arrival entries. Proxy hinge is `sum max(0,q-S)` on the linear queue expressions, shown beside official overflow. Under-capacity green slots indicate where the linear service proxy assumes full capacity although the selected approach has less queue. Transfer error sums the absolute difference between `fraction*capacity*green` and the scorer's rounded transfer.

| Scenario | QUBO schedule | C* | QUBO cost | Regret | Arrivals total | Official overflow / proxy hinge | Under-capacity greens | Transfer error / mismatched slots |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| N003 | 0111|1011 | 61.04 | 89.80 | 47.11% | 65 | 5 / 0.00 | 1 | 0.46 / 2 |
| N046 | 0101|0111 | 33.76 | 46.34 | 37.28% | 50 | 0 / 0.00 | 2 | 0.19 / 1 |
| N011 | 0111|0110 | 35.96 | 48.44 | 34.68% | 48 | 1 / 1.00 | 3 | 0.07 / 1 |
| N059 | 0010|0011 | 66.80 | 87.85 | 31.51% | 42 | 8 / 0.05 | 2 | 0.98 / 2 |
| N008 | 0110|0110 | 53.24 | 68.01 | 27.76% | 54 | 1 / 0.00 | 3 | 1.23 / 1 |

## Failure analysis

Use the table above to separate the mismatch sources. When proxy hinge and official overflow differ, the queue surrogate's smooth square is not reproducing the storage hinge. Under-capacity green counts expose the unclipped-service bias, which makes the proxy queue too small when the approach empties before using full capacity. Transfer error exposes the fractional, unrounded transfer approximation. These effects can stack in one schedule; the table reports each diagnostic rather than attributing the full cost gap to only one.

No QAOA was run. The holdout was used only for the requested comparison and rule selection; R and H were not used to tune constants.
