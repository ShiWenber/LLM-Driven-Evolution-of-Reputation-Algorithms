"""Validate and summarize the five b=3 fixed-strategy benchmarks."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/manuscript_v2/fixation_b3_20260918"
ARCHIVE = ROOT / "results/manuscript_v2/v4_flash/benchmarks/fixation"
PROBES = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "ALLC", "ALLD"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixation(differences, beta=1):
    total, cumulative = 1.0, 0.0
    for value in differences:
        cumulative += value
        total += math.exp(-beta*cumulative)
    return 1.0 / total


def drift_checks(curve):
    checks = []
    for entry in curve:
        repetitions = entry["replicates"]
        blocks = entry["block_payoffs"]
        assert len(blocks) % repetitions == 0
        n = len(blocks) // repetitions
        for r in range(repetitions):
            values = [block["mutant"]-block["resident"] for block in blocks[r*n:(r+1)*n]]
            h = len(values) // 2
            gap = abs(statistics.mean(values[:h]) - statistics.mean(values[h:]))
            checks.append({"k": entry["mutant_count"], "replicate": r, "gap": gap})
    return {"max_gap": max(c["gap"] for c in checks),
            "flagged": [c for c in checks if c["gap"] > .1]}


def tex_probability(value):
    if value >= .0001 or value == 0:
        return f"${value:.4f}$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def analyze():
    rows, sources, diagnostics = [], [], {}
    expected = {"population_size": 20, "benefit": 3, "cost": 1, "beta": 1,
                "burn_in_interactions": 100000, "measure_interactions": 300000,
                "replicates": 5, "seed": 0, "action_error_probability": .01,
                "observation_error_probability": .01, "observation_schedule": "asynchronous",
                "pairing_scheme": "random_matching", "reputation_reset_between_rounds": False}
    for seed in range(5):
        path = DATA / f"seed{seed}/fixation_benchmark.json"
        data = read(path)
        old = read(ARCHIVE / f"seed{seed}/fixation_benchmark.json")
        assert data["candidate"]["code_sha256"] == old["candidate"]["code_sha256"]
        assert set(data["probes"]) == set(PROBES)
        assert all(data["config"].get(k) == v for k, v in expected.items())
        sources.append({"seed": seed, "source": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "candidate_sha256": data["candidate"]["code_sha256"]})
        diagnostics[str(seed)] = {}
        for probe in PROBES:
            pair = data["results"][probe]
            forward, reverse = pair["candidate_invades_probe"], pair["probe_invades_candidate"]
            curve = forward["curve"]
            assert [c["mutant_count"] for c in curve] == list(range(1, 20))
            for entry in curve:
                assert entry["replicates"] == len(entry["replicate_differences"]) == 5
                assert math.isclose(statistics.mean(entry["replicate_differences"]), entry["payoff_difference"], abs_tol=1e-14)
            d = [c["payoff_difference"] for c in curve]
            assert math.isclose(fixation(d), forward["rho"], abs_tol=1e-14)
            assert math.isclose(fixation([-value for value in d[::-1]]), reverse["rho"], abs_tol=1e-14)
            diagnostic = drift_checks(curve)
            diagnostics[str(seed)][probe] = diagnostic
            rows.append({"model": "v4-flash", "seed": seed, "probe": probe,
                         "rho_candidate_to_probe": forward["rho"],
                         "rho_probe_to_candidate": reverse["rho"],
                         "mean_payoff_difference": statistics.mean(d),
                         "max_within_replicate_gap": diagnostic["max_gap"],
                         "flagged_replicate_compositions": len(diagnostic["flagged"]),
                         "max_legacy_half_gap": data["stationarity"][probe]["max_half_to_half_gap"]})
    summary = {"config": expected, "sources": sources, "rows": rows, "diagnostics": diagnostics,
               "candidate_probe_pairs": len(rows), "composition_runs": len(rows)*19*5,
               "pair_interactions": len(rows)*19*5*400000,
               "flagged_pairs": [{"seed": row["seed"], "probe": row["probe"],
                                   "max_gap": row["max_within_replicate_gap"],
                                   "count": row["flagged_replicate_compositions"]}
                                  for row in rows if row["flagged_replicate_compositions"]]}
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (DATA / "fixation_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = []
    for seed in range(5):
        values = [value for p in ("L1", "ALLC", "ALLD")
                  for row in rows if row["seed"] == seed and row["probe"] == p
                  for value in (row["rho_candidate_to_probe"], row["rho_probe_to_candidate"])]
        table.append(f"v4-flash & {seed} & " + " & ".join(map(tex_probability, values)) + r" \\")
    (DATA / "table_rows.tex").write_text("\n".join(table)+"\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = analyze()
    print(json.dumps({"rows": [r for r in result["rows"] if r["probe"] in ("L1", "ALLC", "ALLD")],
                      "flagged_pairs": result["flagged_pairs"]}, indent=2))
