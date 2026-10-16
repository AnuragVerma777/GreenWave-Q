"""Compare two global quadratic variants on a fresh hidden-range holdout."""
from __future__ import annotations

import math
import random
import statistics
from pathlib import Path

from src.model import Scenario, enumerate_all, evaluate_schedule
from calibrate_surrogate import queue_expressions, sample_scenario


DEV_SEED = 20261015  # prior calibration corpus; never used for final holdout
DEV_N = 200
HOLDOUT_SEED = 77441931  # new seed for this improvement pass
HOLDOUT_N = 100
RULES = {
    "current S/4": (2.0, 0.25),
    "candidate S/2": (2.0, 0.50),
    "candidate S": (2.0, 1.00),
}


def rank_average(values):
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    j = 0
    while j < len(order):
        k = j + 1
        while k < len(order) and values[order[k]] == values[order[j]]:
            k += 1
        rank = (j + 1 + k) / 2.0
        for pos in range(j, k):
            ranks[order[pos]] = rank
        j = k
    return ranks


def spearman(a, b):
    ra, rb = rank_average(a), rank_average(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x-ma)*(y-mb) for x,y in zip(ra,rb))
    va = sum((x-ma)**2 for x in ra); vb = sum((y-mb)**2 for y in rb)
    return cov / math.sqrt(va*vb) if va and vb else 0.0


def evaluate_rule(s: Scenario, rule):
    multiplier, center_fraction = rule
    gamma = multiplier*s.overflow_weight/(s.storage*s.storage)
    expr = queue_expressions(s)
    true = enumerate_all(s)
    cstar = true[0][1]
    true_cost = dict(true)
    true_energy_order = [x[0] for x in true]
    energies = {}
    for i in range(256):
        bits=f"{i:08b}"; x=[int(b) for b in bits]
        qs=[c+sum(a*b for a,b in zip(coef,x)) for c,coef in expr]
        delay=sum(qs)
        penalty=sum((q-center_fraction*s.storage)**2 for q in qs)
        sw=sum(x[base+t] != x[base+t+1] for base in (0,4) for t in range(3))
        energies[bits]=delay+gamma*penalty+s.switch_weight*sw
    qubo_order=sorted(energies, key=lambda b:(energies[b],b))
    best=qubo_order[0]
    elite=set(true_energy_order[:13])
    overlap=len(elite.intersection(qubo_order[:13]))
    rho=spearman([true_cost[b] for b in true_energy_order],
                 [energies[b] for b in true_energy_order])
    abs_regret=true_cost[best]-cstar
    relative=abs_regret/cstar*100 if cstar else 0.0
    return {"regret":relative,"absolute":abs_regret,"best":best,
            "best_cost":true_cost[best],"cstar":cstar,"rho":rho,
            "elite":overlap,"energy_order":qubo_order,"energies":energies,
            "true_order":true_energy_order,"true_cost":true_cost}


def percentile90(values):
    xs=sorted(values)
    return xs[math.ceil(.90*len(xs))-1]


def summary(rows):
    regrets=[r["regret"] for r in rows]
    absolute=[r["absolute"] for r in rows]
    return {"mean":statistics.mean(regrets),"median":statistics.median(regrets),
            "p90":percentile90(regrets),"worst":max(regrets),
            "mean_abs":statistics.mean(absolute),"p90_abs":percentile90(absolute),
            "rho":statistics.mean(r["rho"] for r in rows),
            "rho_median":statistics.median(r["rho"] for r in rows),
            "rho_min":min(r["rho"] for r in rows),
            "elite":statistics.mean(r["elite"] for r in rows),
            "elite_median":statistics.median(r["elite"] for r in rows),
            "elite_min":min(r["elite"] for r in rows)}


def failure_detail(s: Scenario, row):
    result=evaluate_schedule(s,row["best"])
    qs=[c+sum(a*b for a,b in zip(coef,map(int,row["best"])))
        for c,coef in queue_expressions(s)]
    proxy_hinge=sum(max(0,q-s.storage) for q in qs)
    underused=0
    for inter,key in (("A","states_A"),("B","states_B")):
        for st in result[key]:
            if st["dep_ns"]+st["dep_ew"] < s.capacity:
                # Only count slots where the selected green approach is under capacity.
                bit_index=(st["slot"]-1)+(4 if inter=="B" else 0)
                selected=int(row["best"][bit_index])
                if (selected and st["dep_ns"] < s.capacity) or (not selected and st["dep_ew"] < s.capacity):
                    underused+=1
    transfer_error=0.0; transfer_mismatch_count=0
    for t,st in enumerate(result["states_A"]):
        if t + 1 >= s.num_slots:  # Final-slot transfers leave the modeled network.
            continue
        bit=int(row["best"][t])
        linked_green=(bit==1) if s.link_approach=="NS" else (bit==0)
        predicted=s.link_fraction*s.capacity if linked_green else 0.0
        actual=st["transfer_out"]
        transfer_error+=abs(actual-predicted)
        transfer_mismatch_count += abs(actual-predicted)>1e-9
    return {"true_overflow":result["overflow"],"proxy_hinge":proxy_hinge,
            "true_delay":result["delay"],"underused":underused,
            "transfer_error":transfer_error,"transfer_mismatch":transfer_mismatch_count}


def main():
    dev_rng=random.Random(DEV_SEED)
    dev=[sample_scenario(dev_rng,f"D{i:03d}") for i in range(DEV_N)]
    h_rng=random.Random(HOLDOUT_SEED)
    holdout=[sample_scenario(h_rng,f"N{i:03d}") for i in range(HOLDOUT_N)]
    r=Scenario.from_json("scenario_R.json")
    rrows={name:evaluate_rule(r,rule) for name,rule in RULES.items()}
    results={name:[evaluate_rule(s,rule) for s in holdout] for name,rule in RULES.items()}
    # Selection key follows the requested order; greater correlation is preferred last.
    summaries={name:summary(rows) for name,rows in results.items()}
    winner=min(RULES,key=lambda name:(summaries[name]["mean"],summaries[name]["p90"],
                                      summaries[name]["worst"],-summaries[name]["rho"]))

    out=["# Checkpoint 3 improvement pass", "",
         f"Fresh holdout: {HOLDOUT_N} scenarios, seed `{HOLDOUT_SEED}`. This seed was not used in earlier work. Arrivals are independent integers [0,8]; capacity [4,6]; storage [5,8]; overflow weight U[2,6]; switch weight U[0.25,1.5]; link approach NS/EW; fraction U[0.4,0.8]. All rules are evaluated on the same scenarios.",
         "", "Regret is `(QUBO minimizer official cost - C*) / C*`; absolute regret is the cost difference. Exact QUBO minimizers and energy rankings enumerate all 256 schedules. Spearman uses average ranks for ties. Elite overlap compares the official 13 lowest-cost schedules with the QUBO's 13 lowest-energy schedules.",
         "", "## Candidate definitions", "",
         "| Rule | Penalty | Global constants |", "|---|---|---|",
         "| Current | `gamma * sum (q-aS)^2` | `gamma=2w/S²`, `a=0.25` |",
         "| Candidate 1 | same form | `gamma=2w/S²`, `a=0.50` |",
         "| Candidate 2 | same form | `gamma=2w/S²`, `a=1.00` |",
         "", "The multiplier 2 in `gamma=2w/S²` was retained from the earlier fixed-grid calibration on 150 random scenarios (seed `20261015`). The three center fractions were fixed before this pass and compared on this fresh holdout; `a=0.50` won the requested tuple. Neither R nor H was used to tune constants. There is no per-scenario tuning. All candidates use no auxiliaries and remain at 8 variables.",
         "", "## Fresh holdout results", "",
         "| Rule | Mean regret | Median | P90 | Worst | Mean abs. | P90 abs. | Spearman mean / median / min | Elite mean / median / min (of 13) | R regret / abs. | R Spearman | R elite / 13 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in RULES:
        sm=summaries[name]; r0=rrows[name]
        out.append(f"| {name} | {sm['mean']:.2f}% | {sm['median']:.2f}% | {sm['p90']:.2f}% | {sm['worst']:.2f}% | {sm['mean_abs']:.2f} | {sm['p90_abs']:.2f} | {sm['rho']:.4f} / {sm['rho_median']:.4f} / {sm['rho_min']:.4f} | {sm['elite']:.2f} / {sm['elite_median']:.1f} / {sm['elite_min']} | {r0['regret']:.2f}% / {r0['absolute']:.2f} | {r0['rho']:.4f} | {r0['elite']}/13 |")
    out += ["", f"Selected by (mean regret, P90, worst, then highest mean correlation): **{winner}**.",
            "", "## Five worst random scenarios", "",
            f"All cases below use the selected rule ({winner}) to characterize its failure modes. Arrival level is the total of the 16 external-arrival entries. Proxy hinge is `sum max(0,q-S)` on the linear queue expressions, shown beside official overflow. Under-capacity green slots indicate where the linear service proxy assumes full capacity although the selected approach has less queue. Transfer error sums the absolute difference between `fraction*capacity*green` and the scorer's rounded transfer.",
            "", "| Scenario | QUBO schedule | C* | QUBO cost | Regret | Arrivals total | Official overflow / proxy hinge | Under-capacity greens | Transfer error / mismatched slots |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    current=results[winner]
    worst_ids=sorted(range(HOLDOUT_N),key=lambda i:current[i]["regret"],reverse=True)[:5]
    for i in worst_ids:
        s=holdout[i]; row=current[i]; d=failure_detail(s,row)
        arrival_total=sum(sum(s.arrivals[j][p]) for j in ("A","B") for p in ("NS","EW"))
        out.append(f"| {s.name} | {row['best'][:4]}|{row['best'][4:]} | {row['cstar']:.2f} | {row['best_cost']:.2f} | {row['regret']:.2f}% | {arrival_total} | {d['true_overflow']} / {d['proxy_hinge']:.2f} | {d['underused']} | {d['transfer_error']:.2f} / {d['transfer_mismatch']} |")
    out += ["", "## Failure analysis", "",
            "Use the table above to separate the mismatch sources. When proxy hinge and official overflow differ, the queue surrogate's smooth square is not reproducing the storage hinge. Under-capacity green counts expose the unclipped-service bias, which makes the proxy queue too small when the approach empties before using full capacity. Transfer error exposes the fractional, unrounded transfer approximation. These effects can stack in one schedule; the table reports each diagnostic rather than attributing the full cost gap to only one.",
            "", "No QAOA was run. The holdout was used only for the requested comparison and rule selection; R and H were not used to tune constants.", ""]
    Path("docs/checkpoint3_regret.md").write_text("\n".join(out),encoding="utf-8")
    print("\n".join(out))


if __name__=="__main__": main()
