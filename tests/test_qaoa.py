import json
import sys
import tempfile
import unittest
from pathlib import Path

import subprocess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import Scenario
from src.qubo import build_qubo
from src.qaoa import (run_qaoa, verify_ising_all_assignments, MAX_EVALUATIONS,
                      SHOTS, qubo_to_ising, _build_circuit)


class TestQAOA(unittest.TestCase):
    def test_ising_matches_every_assignment(self):
        scenario = Scenario.from_json(ROOT / "scenario_R.json")
        Q, offset, _ = build_qubo(scenario)
        self.assertLess(verify_ising_all_assignments(Q, offset), 1e-8)

    def test_cli_artifacts_and_determinism(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            command = [sys.executable, str(ROOT / "run.py"), "--scenario",
                       str(ROOT / "scenario_R.json"), "--seed", "1", "--out"]
            first_run = subprocess.run(command + [a], check=True, capture_output=True, text=True)
            second_run = subprocess.run(command + [b], check=True, capture_output=True, text=True)
            self.assertIn("Optimizer evaluations:", first_run.stdout)
            self.assertIn("Runtime:", first_run.stdout)
            first = json.loads((Path(a) / "schedule.json").read_text())
            second = json.loads((Path(b) / "schedule.json").read_text())
            counts = json.loads((Path(a) / "counts.json").read_text())
            self.assertEqual(sum(counts.values()), SHOTS)
            self.assertEqual(first, second)
            self.assertEqual(counts, json.loads((Path(b) / "counts.json").read_text()))
            self.assertLessEqual(first["optimizer"]["evaluations"], MAX_EVALUATIONS)
            for name in ("qubo.json", "circuit.qasm", "parameters.json"):
                self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes())
            self.assertEqual(json.loads((Path(a) / "parameters.json").read_text())["p"], 1)

    def test_bound_qasm_reproduces_distribution(self):
        from qiskit import qasm3
        from qiskit_aer import AerSimulator
        scenario = Scenario.from_json(ROOT / "scenario_R.json")
        with tempfile.TemporaryDirectory() as td:
            result = run_qaoa(scenario, 1, td)
            counts = json.loads((Path(td) / "counts.json").read_text())
            circuit = qasm3.loads((Path(td) / "circuit.qasm").read_text())
            self.assertFalse(circuit.parameters)
            parameters = json.loads((Path(td) / "parameters.json").read_text())
            Q, offset, _ = build_qubo(scenario)
            _, h, J = qubo_to_ising(Q, offset)
            hamiltonian_scale = parameters["hamiltonian_scale"]
            expected, _, _ = _build_circuit(
                parameters["p"], h / hamiltonian_scale, J / hamiltonian_scale,
                measured=True)
            bind = {param: (parameters["gamma"][int(param.name.split("[")[1][:-1])]
                            if param.name.startswith("gamma") else
                            parameters["beta"][int(param.name.split("[")[1][:-1])])
                    for param in expected.parameters}
            self.assertEqual(qasm3.dumps(expected.assign_parameters(bind)),
                             (Path(td) / "circuit.qasm").read_text())
            replay = AerSimulator().run(circuit, shots=SHOTS, seed_simulator=21).result().get_counts()
            from src.qaoa import _schedule_from_count_key
            replay = {_schedule_from_count_key(k): v for k, v in replay.items()}
            # Compare all eight marginal probabilities. Their binomial standard
            # error is at most 0.008 at 4096 shots; 0.07 is a conservative
            # finite-sample tolerance while remaining sensitive to bit reversal.
            for bit in range(8):
                p_saved = sum(v for k, v in counts.items() if k[bit] == "1") / SHOTS
                p_replay = sum(v for k, v in replay.items() if k[bit] == "1") / SHOTS
                self.assertLess(abs(p_saved - p_replay), 0.07, f"bit {bit} marginal differs")


if __name__ == "__main__":
    unittest.main()
