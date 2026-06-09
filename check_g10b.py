import json, glob
files = sorted(glob.glob('results/exp3_static_g10/*/static_control_*.json'))
for f in files:
    d = json.load(open(f))
    print(f'=== {f.split("/")[-2]} ===')
    print('  trials_summary[0] keys:', list(d['trials_summary'][0].keys()))
    ts = d['trials_summary'][0]
    print(f'  final_coop = {ts.get("final_mean_cooperation")}')
    traj = ts.get('trajectory', [])
    print(f'  traj_len = {len(traj)}')
    if traj:
        print(f'  gen0 = {traj[0].get("cooperation_rate_mean"):.3f}  genN = {traj[-1].get("cooperation_rate_mean"):.3f}')
    print()
