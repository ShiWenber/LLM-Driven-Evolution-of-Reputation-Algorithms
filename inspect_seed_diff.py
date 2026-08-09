"""Compare full_seed0 (0.97) vs full_seed4 (0.018) — same obs, very different outcomes.
What's structurally different in the strategies?"""
import json, os, re

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5'

for seed in [0, 1, 2, 3, 4]:
    d = os.path.join(base, f'full_seed{seed}')
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    final_coop = t['trajectory'][-1]['cooperation_rate_mean']
    pop = t['final_population']
    n_unique = len(set(a['code'] for a in pop))

    # Algorithm complexity markers
    n_uses_observation = sum(1 for a in pop if 'observation' in a['code'])
    n_uses_history = sum(1 for a in pop if 'my_history' in a['code'])
    n_uses_round = sum(1 for a in pop if 'round_num' in a['code'])
    n_global = sum(1 for a in pop if re.search(r'^\s*\w+\s*=\s*\[\d', a['code'], re.MULTILINE))
    n_constant_rewr = sum(1 for a in pop if re.search(r'current_reputation\s*[+\-]\s*[-\d.]+', a['code']))

    mean_len = sum(len(a['code']) for a in pop) / len(pop)
    print(f'seed{seed}: coop={final_coop:.3f} | unique={n_unique}/15 | mean_len={mean_len:.0f} | '
          f'uses_obs={n_uses_observation} | uses_hist={n_uses_history} | uses_round={n_uses_round} | '
          f'global={n_global} | const_rewr={n_constant_rewr}')

# Now compare 1 ALLD-like (seed 4) vs 1 Hybrid (seed 0) on side-by-side
print('\n=== seed 0 (high coop 0.97) sample strategy ===')
d = os.path.join(base, 'full_seed0')
files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
with open(os.path.join(d, files[0])) as f:
    t0 = json.load(f)
print(t0['final_population'][0]['code'][:1500])

print('\n\n=== seed 4 (low coop 0.018) sample strategy ===')
d = os.path.join(base, 'full_seed4')
files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
with open(os.path.join(d, files[0])) as f:
    t4 = json.load(f)
print(t4['final_population'][0]['code'][:1500])