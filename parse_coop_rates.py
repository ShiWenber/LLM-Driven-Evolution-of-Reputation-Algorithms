"""Parse cooperation rates from the new (real) Standard run."""
import json
import re
from pathlib import Path

RESULTS = Path('results')
SUBDIRS = {
    'evolutionary': 'exp1_method',
    'threshold':    'exp2_threshold',
    'static':       'exp3_static',
    'random-mutation': 'exp4_random_mut',
}

# Collect records
records = []
for run, sub in SUBDIRS.items():
    d = RESULTS / sub
    if not d.exists():
        continue
    for trial_dir in sorted(d.iterdir()):
        if not trial_dir.is_dir():
            continue
        m = re.match(r'([a-z_0-9.]+)_seed(\d+)', trial_dir.name)
        if not m:
            continue
        obs = m.group(1)
        seed = int(m.group(2))
        # Look for the aggregate file: evolutionary_*.json or static_control_*.json
        agg_files = (
            list(trial_dir.glob('evolutionary_*.json')) +
            list(trial_dir.glob('static_control_*.json')) +
            list(trial_dir.glob('aggregate_*.json'))
        )
        if not agg_files:
            continue
        agg = json.loads(agg_files[0].read_text())
        # New structure: trials_summary list with one entry
        ts = agg.get('trials_summary', [])
        if ts:
            tr = ts[0]
            final_coop = tr.get('final_mean_cooperation')
            traj = tr.get('trajectory', [])
        else:
            final_coop = agg.get('final_mean_cooperation')
            traj = agg.get('trajectory', [])
        gen0 = traj[0]['cooperation_rate_mean'] if traj else None
        records.append({
            'run': run, 'obs': obs, 'seed': seed,
            'gen0_coop': gen0, 'final_coop': final_coop,
            'n_gens': len(traj),
            'trial_dir': str(trial_dir),
        })

# Print per-trial
print(f"{'#':>3} {'run':<18} {'obs':<12} {'seed':>4}  {'gen0':>6}  {'final':>6}  {'ngens':>5}")
for i, r in enumerate(records, 1):
    g0 = f"{r['gen0_coop']:.3f}" if isinstance(r['gen0_coop'], (int, float)) else "?"
    fc = f"{r['final_coop']:.3f}" if isinstance(r['final_coop'], (int, float)) else "?"
    print(f"{i:3d} {r['run']:<18} {r['obs']:<12} {r['seed']:>4}  {g0:>6}  {fc:>6}  {r['n_gens']:>5}")

# Aggregate by run,obs
print()
print("=== mean final cooperation by (run, obs) ===")
from collections import defaultdict
groups = defaultdict(list)
for r in records:
    if isinstance(r['final_coop'], (int, float)):
        groups[(r['run'], r['obs'])].append(r['final_coop'])
keys = sorted(groups.keys())
for k in keys:
    vals = groups[k]
    mean = sum(vals) / len(vals)
    print(f"  {k[0]:<18} {k[1]:<12}  n={len(vals)}  mean={mean:.3f}  values={[f'{v:.3f}' for v in vals]}")
