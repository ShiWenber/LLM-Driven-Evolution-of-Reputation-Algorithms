import json
d = json.load(open('results/exp3_static_g10_n10/private_seed0/static_control_20260609_212519.json'))
print('top-level keys:', list(d.keys()))
ts = d.get('trials_summary', [{}])[0]
print('trials_summary[0] keys:', list(ts.keys()))
print('  trajectory len:', len(ts.get('trajectory', [])))
print('  final_mean_coop:', ts.get('final_mean_cooperation'))
print('  num_valid:', ts.get('num_valid'))
# Check for final_population
for k in ts.keys():
    if 'pop' in k.lower() or 'agent' in k.lower():
        print(f'  {k}: {len(ts[k]) if isinstance(ts[k], list) else ts[k]}')
