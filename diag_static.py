import re, json
from pathlib import Path
ROOT = Path('results/exp3_static')
for td in sorted(ROOT.iterdir()):
    if not td.is_dir(): continue
    agg = list(td.glob('*.json'))
    if not agg: continue
    for f in agg:
        d = json.loads(f.read_text())
        if 'trials_summary' in d:
            ts = d['trials_summary'][0]
            print(f'{td.name}/{f.name}: trial keys = {list(ts.keys())[:6]}')
            print(f'  final_coop = {ts.get("final_mean_cooperation")}')
            traj = ts.get('trajectory', [])
            if traj:
                print(f'  trajectory_len = {len(traj)}, first gen coop = {traj[0].get("cooperation_rate_mean")}')
            break
