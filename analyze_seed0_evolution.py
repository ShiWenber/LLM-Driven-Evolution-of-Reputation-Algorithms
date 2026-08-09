"""Analyze LLM seed0 evolution trajectory."""
import json
from pathlib import Path

D = json.loads(Path("results/quantitative_baseline/LLM_evolution_seed0/evolutionary.json").read_text(encoding="utf-8"))
traj = D["trajectory"]

print("=== LLM seed0: 30-generation trajectory ===")
print(f"{'gen':>3} | {'coop':>6} | {'fit_mean':>8} | {'fit_max':>7} | {'n_inter':>7}")
print("-" * 50)
for t in traj:
    print(f"{t['generation']:3d} | {t['cooperation_rate_mean']:6.3f} | {t['fitness_mean']:8.2f} | {t['fitness_max']:7.1f} | {t['n_interactions']:7d}")

# Identify phase transitions
print("\n=== Phase analysis ===")
phases = []
prev = traj[0]['cooperation_rate_mean']
phase_start = 0
for i, t in enumerate(traj):
    cur = t['cooperation_rate_mean']
    if i > 0 and abs(cur - prev) > 0.15:
        phases.append((phase_start, i-1, prev))
        phase_start = i
    prev = cur
phases.append((phase_start, len(traj)-1, traj[-1]['cooperation_rate_mean']))

for start, end, _ in phases:
    seg = traj[start:end+1]
    mean_coop = sum(t['cooperation_rate_mean'] for t in seg) / len(seg)
    print(f"  gens {start:2d}-{end:2d}: {len(seg):2d} gens, mean coop = {mean_coop:.3f}")
