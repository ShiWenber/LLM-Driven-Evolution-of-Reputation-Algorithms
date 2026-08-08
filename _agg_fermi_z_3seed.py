"""Aggregate the 3 Fermi Z-like seeds into a single summary + 3-seed curve plot."""
import json
import statistics
from pathlib import Path

BASE = Path("results/quantitative_baseline")
SEEDS = [0, 1, 2]

per_seed = []
trajectories = {}

for s in SEEDS:
    p = BASE / f"LLM_v3_fermi_z_g100_1000inter_seed{s}/evolutionary.json"
    with open(p) as f:
        d = json.load(f)
    traj = d["trajectory"]  # list of 100 gen-dicts
    coop = [g["cooperation_rate_mean"] for g in traj]
    fit = [g["fitness_mean"] for g in traj]
    # global coop fraction = mean(per-agent cooperation_rate) at each gen
    # we already store mean per gen in cooperation_rate field
    per_seed.append({
        "seed": s,
        "gen0_coop": coop[0],
        "final_coop": coop[-1],
        "final_fitness": fit[-1],
    })
    trajectories[s] = coop

coops = [s["final_coop"] for s in per_seed]
fitnesses = [s["final_fitness"] for s in per_seed]
gen0s = [s["gen0_coop"] for s in per_seed]

summary = {
    "label": "LLM_v3_fermi_z_g100_1000inter",
    "scheme": "fermi_z_like (mu=0.1, beta=5.0, updates_per_gen=15)",
    "num_gens": 100,
    "target_interactions_per_gen": 1000,
    "n_seeds": len(SEEDS),
    "seeds": per_seed,
    "aggregate": {
        "gen0_coop_mean": statistics.mean(gen0s),
        "final_coop_mean": statistics.mean(coops),
        "final_coop_std": statistics.pstdev(coops),
        "final_coop_min": min(coops),
        "final_coop_max": max(coops),
        "final_fitness_mean": statistics.mean(fitnesses),
        "final_fitness_std": statistics.pstdev(fitnesses),
    },
}
out = BASE / "LLM_v3_fermi_z_g100_1000inter_3seed_agg.json"
with open(out, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"Wrote {out}")
print()
print("Per-seed final coop:")
for s in per_seed:
    print(f"  seed{s['seed']}: gen0={s['gen0_coop']:.3f}  final={s['final_coop']:.3f}  fitness={s['final_fitness']:.1f}")
print()
agg = summary["aggregate"]
print(f"3-seed: gen0 mean = {agg['gen0_coop_mean']:.3f}")
print(f"3-seed: final coop = {agg['final_coop_mean']:.3f} +/- {agg['final_coop_std']:.3f}")
print(f"3-seed: final coop range = [{agg['final_coop_min']:.3f}, {agg['final_coop_max']:.3f}]")
print(f"3-seed: final fitness = {agg['final_fitness_mean']:.1f} +/- {agg['final_fitness_std']:.1f}")

# Plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, s in enumerate(SEEDS):
        ax.plot(trajectories[s], label=f"seed {s}", color=colors[i], alpha=0.85, linewidth=1.4)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation fraction")
    ax.set_title(f"Fermi Z-like (LLM μ+small-mutate), 3 seeds × 100 gen × 1000 inter\n"
                 f"final coop = {agg['final_coop_mean']:.3f} ± {agg['final_coop_std']:.3f}")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    png = BASE / "fermi_z_g100_3seed_1000inter.png"
    pdf = BASE / "fermi_z_g100_3seed_1000inter.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.savefig(pdf)
    plt.close()
    print(f"\nPlot: {png} ({png.stat().st_size/1024:.1f} KB)")
    print(f"Plot: {pdf}")
except ImportError:
    print("\n[skip plot: matplotlib not available]")
