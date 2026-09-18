import json
from pathlib import Path
import subprocess
import sys

from experiments.v2_quantitative.baselines import BASELINES
from tools.analyze_paper_fixation_b3 import drift_checks


def test_cli_defaults_to_benefit_three_and_fast_schedule(tmp_path):
    candidate = tmp_path / "cooperator.py"
    candidate.write_text(BASELINES["ALLC"], encoding="utf-8")
    output = tmp_path / "measurement"
    completed = subprocess.run([
        sys.executable, "-m", "experiments.analysis.invasion.run_fixation_benchmark",
        "--candidate", f"test=agent-type1={candidate}", "--probes", "ALLC",
        "--population-size", "4", "--burn-in", "4", "--measure", "8",
        "--replicates", "1", "--workers", "1", "--output", str(output),
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((output / "fixation_benchmark.json").read_text())
    assert result["config"]["benefit"] == 3
    assert result["config"]["cost"] == 1
    assert result["config"]["observation_schedule"] == "asynchronous"
    for entry in result["results"]["ALLC"]["candidate_invades_probe"]["curve"]:
        assert entry["payoff_mutant"] == entry["payoff_resident"] == 2


def test_within_repetition_drift_does_not_confuse_different_levels():
    blocks = [{"mutant": value, "resident": 0} for value in [0, 0, 1, 1]]
    stable = drift_checks([{"mutant_count": 1, "replicates": 2, "block_payoffs": blocks}])
    assert stable == {"max_gap": 0, "flagged": []}
    drifting = drift_checks([{"mutant_count": 1, "replicates": 1, "block_payoffs": blocks}])
    assert drifting == {"max_gap": 1, "flagged": [{"k": 1, "replicate": 0, "gap": 1}]}
