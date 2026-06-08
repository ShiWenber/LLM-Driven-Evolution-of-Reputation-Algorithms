"""Analyse robustness results: cross-LLM comparison."""
import json
import re
from pathlib import Path

ROBUST = Path('results/exp5_robustness')

# Load 6 trials
rows = []
for trial_dir in sorted(ROBUST.iterdir()):
    if not trial_dir.is_dir():
        continue
    m = re.match(r'([a-z_0-9.]+)_seed(\d+)', trial_dir.name)
    if not m:
        continue
    obs, seed = m.group(1), int(m.group(2))
    agg = list(trial_dir.glob('evolutionary_*.json'))
    if not agg:
        continue
    data = json.loads(agg[0].read_text())
    ts = data.get('trials_summary', [{}])[0]
    final = ts.get('final_mean_cooperation')
    traj = ts.get('trajectory', [])
    gen0 = traj[0]['cooperation_rate_mean'] if traj else None
    rows.append({
        'obs': obs, 'seed': seed,
        'final_coop': final,
        'gen0_coop': gen0,
        'n_gens': len(traj),
    })

# v7-flash numbers (from results/_manifest.json of the main run, partial_0.3 / 0.7 only)
# Recall: v7 (deepseek-v4-flash) had
#   partial_0.3 evolutionary: mean across 3 seeds = 0.000 (all collapsed)
#   partial_0.7 evolutionary: mean across 3 seeds = 0.228
V7_FLASH = {
    'partial_0.3': 0.000,
    'partial_0.7': 0.228,
}

# Aggregate coder by obs
from collections import defaultdict
by_obs = defaultdict(list)
for r in rows:
    by_obs[r['obs']].append(r['final_coop'])

print("=== Cross-LLM robustness ===")
print(f"6 trials of deepseek-coder on partial_0.3 and partial_0.7 (3 seeds each)")
print()
for obs in sorted(by_obs):
    vals = by_obs[obs]
    n = len(vals)
    mean = sum(vals) / n
    sd = (sum((v - mean)**2 for v in vals) / n) ** 0.5
    v7 = V7_FLASH.get(obs, '?')
    print(f"  {obs:<14}  coder: mean={mean:.3f}  sd={sd:.3f}  range=[{min(vals):.3f}, {max(vals):.3f}]  ({n} seeds)")
    print(f"  {'':<14}  flash(v7): mean={v7:.3f}")
    print(f"  {'':<14}  delta: {mean - v7:+.3f}")
    print()

# Per-trial
print("=== Per-trial ===")
for r in rows:
    print(f"  {r['obs']:<14}  seed={r['seed']}  gen0={r['gen0_coop']:.3f}  final={r['final_coop']:.3f}")
