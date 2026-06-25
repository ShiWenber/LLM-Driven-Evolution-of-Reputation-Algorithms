"""Analyze the cooperate/defect sweep results."""
import json, os, statistics
from collections import Counter

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5'
obs_levels = ['private', 'partial_0.3', 'partial_0.7', 'full']

print('=== Exp 6 sweep v3 (5 seeds x 4 obs, cooperate/defect label) ===\n')
print(f'{"obs":12s} {"mean":>8s} {"std":>8s} {"min":>8s} {"max":>8s} {"OK/total":>10s} {"per-seed finals"}')
print('-' * 100)

for obs in obs_levels:
    finals = []
    for seed in range(5):
        d = os.path.join(base, f'{obs}_seed{seed}')
        if not os.path.exists(d): continue
        files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
        if not files: continue
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        with open(os.path.join(d, files[0])) as f:
            t = json.load(f)
        coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
        finals.append(coops[-1])
    if not finals: continue
    mean = statistics.mean(finals)
    std = statistics.stdev(finals) if len(finals) > 1 else 0
    n_ok = sum(1 for f in finals if f > 0.5)
    print(f'{obs:12s} {mean:8.3f} {std:8.3f} {min(finals):8.3f} {max(finals):8.3f} {n_ok}/{len(finals):>3d}      {[round(f, 3) for f in finals]}')