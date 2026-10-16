# Scenario-independent QUBO derivation

## Decision variables and bit order

There are four slots at each of two intersections. Let `x[A,t]` and `x[B,t]` be binary, with `1` selecting the NS phase and `0` selecting EW. The eight variables are ordered

```text
x0..x3 = A slots 1..4
x4..x7 = B slots 1..4
```

This order is also the order of bits in a schedule string. No auxiliary variables are used, so the QUBO has eight variables (within the 16-qubit limit).

## Linear queue proxy

For each approach and slot, construct a linear expression for its end-of-slot queue proxy. External arrivals are accumulated through the slot. The green service contribution is `capacity * x` for NS and `capacity * (1-x)` for EW. The directed A-to-B link contributes `fraction * capacity` times A's linked-phase green indicator to B's linked approach from the following slot onward. Thus every proxy has the form

```text
q_i(x) = c_i + sum_k a[i,k] * x_k
```

The coefficient builder in `src/qubo.py` applies this rule to any scenario. It deliberately uses the stated linear service and transfer approximations: service is not clipped by the available queue, and fractional transfers are not rounded. The official scorer remains the authority for evaluating any resulting schedule.

## Surrogate objective and clipping rule

The exact delay term is the sum of queues, so its queue-proxy contribution is linear: `sum_i q_i(x)`. The scorer's overflow term is `max(0, queue-storage)`, a hinge that is not quadratic. To retain a QUBO without adding one auxiliary variable per queue-slot, approximate the hinge by a smooth quadratic congestion penalty on the queue proxy:

```text
C_tilde(x) = sum_i q_i(x)
           + gamma * sum_i (q_i(x) - storage/2)^2
           + switch_weight * sum_intersection,t (x[t] + x[t+1] - 2*x[t]*x[t+1])
```

The scenario-independent clipping surrogate is a centered quadratic soft penalty instead of a hard `max`. It has no threshold branch and increases its marginal penalty as the queue proxy moves above half of storage. The global coefficient multiplier `gamma = 2 * overflow_weight / storage^2` was selected during the earlier fixed-grid calibration on 150 random scenarios (seed `20261015`). For this pass, the center fraction was selected from the predeclared candidates `{0.25, 0.50, 1.00}` on the new 100-scenario holdout (seed `77441931`), using the requested ordering of mean regret, P90, worst regret, then correlation; `0.50` won. Neither R nor H was used to select constants. This normalization scales the penalty by overflow weight and storage. It does not reproduce the scorer's clipped overflow exactly, and the linear queue proxy can become negative when the capacity approximation over-serves an approach. These are explicit surrogate limitations; all reported schedule costs must be computed with the official traffic model.

The switching expression is exact because, for binary variables, `x XOR y = x + y - 2xy`.

## Conversion to QUBO

For a proxy `q = c + sum_k a_k x_k`, binary idempotence gives

```text
q_shift = (c - storage/2) + sum_k a_k*x_k
q_shift^2 = (c-storage/2)^2
             + sum_k (2*(c-storage/2)*a_k + a_k^2) x_k
            + 2*sum_(j<k) a_j*a_k*x_j*x_k.
```

Each `q_i` adds `c_i` to the objective offset and `a[i,k]` to diagonal `Q[k,k]`. Its shifted square adds `gamma*(c_i-storage/2)^2` to the offset, `gamma*(2*(c_i-storage/2)*a[i,k] + a[i,k]^2)` to `Q[k,k]`, and `2*gamma*a[i,j]*a[i,k]` to upper-triangular `Q[j,k]`. Switching adds `switch_weight` to each adjacent diagonal and `-2*switch_weight` to the corresponding pair coefficient.

The serialized convention is `x^T Q x + offset`, with only the upper triangle populated. `outputs/R/qubo.json` contains the scenario-R coefficients, offset, variable mapping, gamma, and auxiliary-variable count. Rebuild it from any other scenario by running the normal pipeline; the formulas above do not depend on scenario R.

## Validation boundary

The QUBO minimizer is a surrogate candidate, not a claim of exact traffic optimality. Compare its decoded schedule under `src/model.py` with the official scorer, exact enumeration, and the required baselines. The four official scenario-R vectors validate the scorer implementation, not the surrogate.
