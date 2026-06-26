"""Inspect the high-cooperation strategies from A and C."""
import json, os

for subdir, obs, seed in [
    ('A_larger_budget', 'full', 0),
    ('A_larger_budget', 'partial_0.7', 1),
    ('C_reputation_noise', 'full', 0),
    ('C_reputation_noise', 'full', 2),
]:
    d = rf'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\{subdir}\{obs}_seed{seed}'
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    coop = t['trajectory'][-1]['cooperation_rate_mean']
    print(f'\n=== {subdir} {obs} seed{seed} (final coop {coop:.3f}) ===')
    # Show agent 0 strategy
    print(t['final_population'][0]['code'][:1800])