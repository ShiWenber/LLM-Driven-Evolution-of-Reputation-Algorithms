import json, os, statistics
d = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'
with open(os.path.join(d, 'evolutionary_20260616_105442.json')) as f:
    t = json.load(f)
print('keys:', list(t.keys()))
trial = t['trials_summary'][0] if 'trials_summary' in t else t.get('trials', [{}])[0]
print('trial keys:', list(trial.keys()))
coops = [g['cooperation_rate_mean'] for g in trial['trajectory']]
print()
print('Trajectory (coop rate per gen):')
for i, c in enumerate(coops):
    print(f'  gen {i}: {c:.3f}')
print()
print(f'gen-0: {coops[0]:.3f}, gen-10: {coops[-1]:.3f}')
fp = t['final_population']
print(f'final_pop: {len(fp)} agents')
out = {
    'final_cooperation': coops[-1],
    'trajectory': coops,
    'final_population': fp
}
with open(os.path.join(d, 'summary.json'), 'w') as f:
    json.dump(out, f, indent=2, default=str)
print('Saved summary')
