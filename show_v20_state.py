"""v20 complete: show static + random final state with means/stds.
Handles both evo_*.json and static_control_*.json formats."""
import json, os
from pathlib import Path
import statistics

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

def collect(base):
    out = {}
    for d in base.iterdir():
        if not d.is_dir(): continue
        jsons = [f for f in d.iterdir()
                 if (f.name.startswith('evo_') or f.name.startswith('static_control_') or f.name.startswith('evolutionary_'))
                 and f.name.endswith('.json')]
        if not jsons: continue
        jsons.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        try:
            with open(jsons[0]) as f:
                t = json.load(f)
            # Different formats:
            if 'trials_summary' in t and t['trials_summary']:
                # static_control_*.json format
                traj = t['trials_summary'][0].get('trajectory', [])
                coop = traj[-1]['cooperation_rate_mean'] if traj else None
            elif t.get('trajectory'):
                coop = t['trajectory'][-1]['cooperation_rate_mean']
            else:
                continue
            if coop is not None:
                out[d.name] = coop
        except Exception: pass
    return out

print('=== STATIC (exp3_static_g10_n10) ===')
STATIC = REPO / 'results' / 'exp3_static_g10_n10'
data = collect(STATIC)
for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
    vals = [v for k, v in data.items() if k.startswith(obs + '_seed')]
    if vals:
        print(f'  {obs:12s} n={len(vals):2d}  mean={statistics.mean(vals):.3f}  std={statistics.stdev(vals) if len(vals)>1 else 0:.3f}  '
              f'range=[{min(vals):.3f}, {max(vals):.3f}]')

print('\n=== RANDOM MUTATION (exp4_random_mut) ===')
RAND = REPO / 'results' / 'exp4_random_mut'
data = collect(RAND)
for obs in ['private', 'partial_0.3', 'full']:
    vals = [v for k, v in data.items() if k.startswith(obs + '_seed')]
    if vals:
        print(f'  {obs:12s} n={len(vals):2d}  mean={statistics.mean(vals):.3f}  std={statistics.stdev(vals) if len(vals)>1 else 0:.3f}  '
              f'range=[{min(vals):.3f}, {max(vals):.3f}]')