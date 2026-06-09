import json, glob
files = sorted(glob.glob('results/exp3_static_g10/*/static_control_*.json'))
print(f'Found {len(files)} static G=10 files:')
for f in files:
    d = json.load(open(f))
    keys = list(d.keys())[:8]
    fc = d.get('final_mean_cooperation')
    traj = d.get('trajectory', [])
    first = traj[0].get('cooperation_rate_mean') if traj else None
    last = traj[-1].get('cooperation_rate_mean') if traj else None
    print(f'  {f.split("/")[-2]}: keys={keys}')
    print(f'    final_coop={fc}  traj_len={len(traj)}  gen0={first}  genN={last}')
