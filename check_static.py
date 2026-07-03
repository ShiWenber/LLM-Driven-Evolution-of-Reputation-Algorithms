"""Static OK check (handles both evo_*.json and static_control_*.json)"""
import json
from pathlib import Path
STATIC = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp3_static_g10_n10')
ok = []
for d in sorted(STATIC.iterdir()):
    if d.is_dir():
        # Try both evo_*.json and static_control_*.json
        jsons = [f for f in d.iterdir() if (f.name.startswith('evo_') or f.name.startswith('static_control_')) and f.name.endswith('.json')]
        if not jsons: continue
        # Pick newest
        jsons.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        try:
            with open(jsons[0]) as f:
                t = json.load(f)
            # static_control has trials_summary[0].trajectory
            if 'trials_summary' in t and t['trials_summary']:
                traj = t['trials_summary'][0].get('trajectory', [])
                coop = traj[-1]['cooperation_rate_mean'] if traj else None
            else:
                traj = t.get('trajectory', [])
                coop = traj[-1]['cooperation_rate_mean'] if traj else None
            if coop is not None:
                ok.append((d.name, coop))
        except Exception as e:
            print(f'err {d.name}: {e}')
print(f'OK: {len(ok)}')
for n, c in ok:
    print(f'  {n}: {c:.3f}')