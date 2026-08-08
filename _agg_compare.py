"""Compare all 100gen x 3seed x 1000inter production runs."""
import json
import statistics
from pathlib import Path

BASE = Path("results/quantitative_baseline")
SEEDS = [0, 1, 2]

LABELS = {
    "LLM_v3_g100_1000inter": "v3 1000inter (legacy tournament)",
    "LLM_v3_g100_1000inter_ADVERSARIAL": "Ablation A v1 (adversarial+canonical hints)",
    "LLM_v3_fermi_z_g100_1000inter": "Fermi Z-like (LLM μ+small-mutate)",
}

for prefix, label in LABELS.items():
    print(f"\n=== {label} ===")
    rows = []
    for s in SEEDS:
        p = BASE / f"{prefix}_seed{s}/evolutionary.json"
        try:
            d = json.load(open(p))
        except FileNotFoundError:
            rows.append({"seed": s, "status": "missing"})
            continue
        t = d["trajectory"]
        rows.append({
            "seed": s,
            "status": "ok",
            "gen0_coop": t[0]["cooperation_rate_mean"],
            "final_coop": t[-1]["cooperation_rate_mean"],
            "final_fitness": t[-1]["fitness_mean"],
            "n_gens": len(t),
        })
    coops = [r["final_coop"] for r in rows if r["status"] == "ok"]
    fits = [r["final_fitness"] for r in rows if r["status"] == "ok"]
    for r in rows:
        if r["status"] == "ok":
            print(f"  seed{r['seed']}: gen0={r['gen0_coop']:.3f}  final={r['final_coop']:.3f}  fitness={r['final_fitness']:.1f}  ({r['n_gens']} gens)")
        else:
            print(f"  seed{r['seed']}: missing")
    if coops:
        print(f"  -> 3-seed: final coop = {statistics.mean(coops):.3f} ± {statistics.pstdev(coops):.3f}, fitness = {statistics.mean(fits):.1f} ± {statistics.pstdev(fits):.1f}")
