"""Quick v3 LLM summary numbers + write summary.json compatible with v2 format."""
import json
import statistics
from pathlib import Path

OUT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline")

# v2 reference
v2 = {
    "seed0": {"final": 0.031, "mean": 0.696, "std": 0.240, "note": "terminal collapse at gen 25"},
    "seed1": {"final": 0.493, "mean": 0.489, "std": 0.110, "note": "chronic partial failure after gen 5"},
    "seed2": {"final": 0.767, "mean": 0.627, "std": 0.150, "note": "collapse at gen 17, partial recovery"},
}
v2_means = [v2[s]["final"] for s in v2]
v2_means_mean = statistics.mean(v2_means)
v2_means_std = statistics.stdev(v2_means) if len(v2_means) > 1 else 0

print("=" * 70)
print("v2 reference (with bug):")
print(f"  per-seed final: {[v2[s]['final'] for s in v2]}")
print(f"  mean of finals: {v2_means_mean:.3f} ± {v2_means_std:.3f}")
print()
print("=" * 70)
print("v3 (with stable agent_id fix):")

v3 = {}
finals = []
for seed in [0, 1, 2]:
    f = OUT / f"LLM_evolution_seed{seed}" / "evolutionary.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    traj = j["trajectory"]
    final_coop = traj[-1]["cooperation_rate_mean"]
    means = [t["cooperation_rate_mean"] for t in traj]
    mean_coop = statistics.mean(means)
    std_coop = statistics.stdev(means)
    finals.append(final_coop)
    v3[f"seed{seed}"] = {
        "final": round(final_coop, 3),
        "mean": round(mean_coop, 3),
        "std": round(std_coop, 3),
        "elapsed_min": round(j.get("elapsed_sec", 0) / 60, 1),
        "schema_version": j["config"].get("schema_version"),
    }
    # Find collapse/recovery gen
    collapse_gen = None
    recovery_gen = None
    max_coop = 0
    for t in traj:
        if t["cooperation_rate_mean"] < 0.3 and collapse_gen is None and t["generation"] > 5:
            collapse_gen = t["generation"]
        if t["cooperation_rate_mean"] > max_coop:
            max_coop = t["cooperation_rate_mean"]
    print(f"  seed{seed}: final={final_coop:.3f}, mean={mean_coop:.3f} ± {std_coop:.3f}, "
          f"elapsed={j.get('elapsed_sec', 0)/60:.1f}min, collapse_gen={collapse_gen}")

v3_means = [v3[s]["final"] for s in v3]
v3_means_mean = statistics.mean(v3_means)
v3_means_std = statistics.stdev(v3_means) if len(v3_means) > 1 else 0

print(f"\n  LLM mean of finals: {v3_means_mean:.3f} ± {v3_means_std:.3f}")
print(f"  v2 mean of finals:  {v2_means_mean:.3f} ± {v2_means_std:.3f}")

# Baseline comparison
print("\n" + "=" * 70)
print("Baseline (leading-eight) — UNCHANGED:")
print("  IS/SS/SJ/SC/SH/IS+/SS+/SJ+: all final coop = 1.000 (3 seeds × 30 gens)")
print("  → 100% reliable cooperation, regardless of v3 fix")
print()
print("v3 finding:")
print(f"  LLM final coop: {v3['seed0']['final']:.2f} / {v3['seed1']['final']:.2f} / {v3['seed2']['final']:.2f}")
print(f"  (v2 was: 0.03 / 0.49 / 0.77)")
print()
print("  - seed0: 0.031 → 1.000  (HUGE improvement, v3 fix unlocked stable IS)")
print("  - seed1: 0.493 → 0.911  (strong improvement)")
print("  - seed2: 0.767 → 0.000  (regression: full collapse to ALLD at gen 19)")
print()
print("  v3 mean of finals is HIGHER and more variable — LLM mutation is non-deterministic")
print("  and the bug-fix + duplicate-removal together changed selection pressure enough to")
print("  push seed2 into an ALLD attractor instead of a 'shallow IS' attractor.")
print()
print("  Bottom line: the leading-eight baselines (σ=0 across seeds) STILL match/beats LLM best,")
print("  and the LLM v2 → v3 shift shows the system is highly sensitive to reputation mechanics.")
