# Quantum Traffic Gridlock

QC@IITI Qiskit Fall Fest 2026 — IIT Indore

This repository models two linked signal intersections (A and B) over four slots and compares a scenario-independent QUBO with an official traffic scorer and QAOA sampling. The scorer is authoritative for official costs.

## Environment

The clean-release verification used Python 3.14.2, Qiskit 2.5.2, Qiskit Aer 0.17.2, qiskit-qasm3-import 0.6.0, NumPy 2.5.4, and SciPy 1.18.1. Pytest 9.1.1 runs the test suite. Python 3.10+ is required by the project source.

Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the pipeline and find its outputs

The CLI syntax is `python run.py --scenario <json> --seed <int> --out <dir>`; write `--scenario` without a space after the two hyphens. The saved main R and H runs use seed 1. 

```bash
python run.py --scenario scenario_R.json --seed 1 --out outputs/R
python run.py --scenario "scenario_H-1.json" --seed 1 --out outputs/H
```

The required files for each scenario are:

| Scenario | QUBO | Bound circuit | Raw counts | Decoded schedule |
|---|---|---|---|---|
| R | `outputs/R/qubo.json` | `outputs/R/circuit.qasm` | `outputs/R/counts.json` | `outputs/R/schedule.json` |
| H | `outputs/H/qubo.json` | `outputs/H/circuit.qasm` | `outputs/H/counts.json` | `outputs/H/schedule.json` |

Each run directory also contains `parameters.json` (selected QAOA angles and Ising scale) and `baselines.json` (scenario baseline costs). `outputs/R/seed_<n>/` holds the independent Block 6 runs for seeds 1, 7, 21, 42, and 101.
`outputs/H/seed_<n>/` holds the corresponding H runs; its five-seed comparison is saved as `outputs/H/block6_metrics.json` and `outputs/H/block6_metrics.md`.

Bit order is A slots 1–4 followed by B slots 1–4. A bit of `1` means NS; `0` means EW. Qiskit count strings are little-endian and are reversed during decode.

Score the saved main schedule:

```bash
python scorer.py --scenario scenario_R.json --schedule outputs/R/schedule.json
python scorer.py --scenario "scenario_H-1.json" --schedule outputs/H/schedule.json
```

## Tests and experiment tables

Run the full automated suite:

```bash
python -m pytest -q
```

The deterministic main run and five-seed Block 6 metrics can be recreated with:

```bash
python run.py --scenario scenario_R.json --seed 1 --out outputs/R
python block6_metrics.py --scenario scenario_R.json --seed 1 7 21 42 101 --out outputs/R
python run.py --scenario "scenario_H-1.json" --seed 1 --out outputs/H
python block6_metrics.py --scenario "scenario_H-1.json" --seed 1 7 21 42 101 --out outputs/H
```

Block 7 ablation and generalization tables are reproducible with:

```bash
python block7_ablation.py --reuse-results
# To rerun the 900-case ablation plus the generalization runs:
python block7_ablation.py --bank-size 4 --eval-cap 80
```

The saved Block 6 comparison and per-seed rows are in `outputs/R/block6_metrics.json` and `.md`, and the matching H table is in `outputs/H/`. Block 7 factor tables, generalization summaries, random-bank scenarios, and per-seed artifacts are under `outputs/block7/`. The ablation varies p=1..4, penalty multiplier x0.5/x1/x2, and linear-ramp/random/fixed initialization. Its detailed command options are available with `python block7_ablation.py --help`.

The QAOA depth calibration and Ising-scale comparison are reproducible with `python calibrate_qaoa_depth.py` and `python scale_ablation.py`; both use fixed random-bank seeds and exclude R/H from tuning.

## Scenarios and reports

`scenario_R.json` is the R scenario. The supplied H scenario is `scenario_H-1.json`; its main seed-1 pipeline output is in `outputs/H/`. The unchanged CLI also accepts other scenario files, including generated hidden-range examples in `outputs/block7/random_bank/`.

- `report/report.pdf`: three-page technical report.
- `report/ai_disclosure.pdf`: separate one-page disclosure; the author confirmation boxes are intentionally left blank.
- Rebuild both PDFs from the saved results with `python report/build_release_pdfs.py`.
- `docs/derivation.md`: QUBO derivation and calibration record.
- `docs/checkpoint3_regret.md`: surrogate regret analysis and tuning evidence.
- `docs/qaoa_ablation.md`: depth selection evidence.
- `AI_USAGE_LOG.md`: AI-use history supplied for this project.

## Repository map

```text
run.py, scorer.py              CLI and official scorer
src/model.py                   Scenario and official traffic scoring
src/qubo.py                    Scenario-independent QUBO builder
src/qaoa.py                    Ising mapping, QAOA, optimization, QASM/count artifacts
src/baselines.py, metrics.py   Baselines and benchmark metrics
block6_metrics.py              Five-seed metrics and comparison table
block7_ablation.py             Ablation and random-scenario generalization runs
calibrate_*.py                 Fixed-seed calibration experiments
scale_ablation.py              Ising cost-scale comparison experiment
scenario_R.json                Official R input
scenario_H-1.json              Supplied H input
outputs/R/                     Designated main run and five seed_<n> runs
outputs/H/                     H main run (seed 1)
outputs/block7/                Ablation, random scenarios, and generalization artifacts
docs/                          Derivation and checkpoint evidence
report/                         Technical report, disclosure, and PDF builder
tests/                          Unit, parity, reproducibility, and integration tests
requirements.txt                Python dependencies
AI_USAGE_LOG.md                 Existing AI-use record
```





https://github.com/AnuragVerma777/GreenWave-Q

