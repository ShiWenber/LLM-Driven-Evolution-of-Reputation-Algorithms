"""v19 Intern A 1:1 comparison: N=15,G=10 Intern vs N=15,G=10 DeepSeek main plan.
Also: v18 Intern A (N=10,G=10) vs v15 deepseek A (N=30,G=20) for completeness.
"""
import json, os
import numpy as np

v19_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp8_intern_ceiling_v19_A'
v18_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp8_intern_ceiling_v18'
ds_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp1_method_n10'
v15_ceiling_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling'

# Load v19
with open(os.path.join(v19_base, '_manifest.json')) as f:
    v19 = json.load(f)

v19_data = {}
for m in v19:
    if not m.get('ok'): continue
    v19_data[(m['probe'], m['obs'])] = m['coop_final']

# Load v18
with open(os.path.join(v18_base, '_manifest.json')) as f:
    v18 = json.load(f)

v18_data = {}
for m in v18:
    if not m.get('ok'): continue
    v18_data[(m['probe'], m['obs'])] = m['coop_final']

# v15 deepseek A (N=30, G=20) from exp7
def load_ds_exp7(probe, obs):
    finals = []
    for seed in range(3):
        d = os.path.join(v15_ceiling_base, probe, f'{obs}_seed{seed}')
        if not os.path.exists(d): continue
        files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
        if not files: continue
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        with open(os.path.join(d, files[0])) as f:
            t = json.load(f)
        if t.get('trajectory'):
            finals.append(t['trajectory'][-1]['cooperation_rate_mean'])
    return finals

# v15 deepseek main plan (N=15, G=10)
def load_ds_main(obs):
    finals = []
    for seed in range(10):
        d = os.path.join(ds_base, f'{obs}_seed{seed}')
        if not os.path.exists(d): continue
        files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
        if not files: continue
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        with open(os.path.join(d, files[0])) as f:
            t = json.load(f)
        if t.get('trajectory'):
            finals.append(t['trajectory'][-1]['cooperation_rate_mean'])
    return finals

# === 1:1 N=15, G=10 Intern A vs DeepSeek main plan ===
print('=== 1:1 Intern A (N=15, G=10, v19) vs DeepSeek-V4-Flash main plan (N=15, G=10) ===\n')
print(f'{"obs":12s} {"Intern A (N=15)":>20s} {"DS main plan (N=15)":>25s}')
print('-' * 60)
for obs in ['full', 'partial_0.7']:
    i_vals = [v for (p, o), v in v19_data.items() if p == 'A_larger_budget_N15G10' and o == obs]
    d_vals = load_ds_main(obs)
    i_mean = np.mean(i_vals) if i_vals else None
    d_mean = np.mean(d_vals) if d_vals else None
    i_ok = sum(1 for v in i_vals if v > 0.5) if i_vals else 0
    d_ok = sum(1 for v in d_vals if v > 0.5) if d_vals else 0
    print(f'{obs:12s} {i_mean:.2f} ({i_ok}/{len(i_vals)}) {" " * 8} {d_mean:.2f} ({d_ok}/{len(d_vals)})')

# === Compare v18 N=10 vs v19 N=15 Intern A ===
print('\n=== Intern A config comparison (N=10 G=10 vs N=15 G=10) ===\n')
print(f'{"obs":12s} {"N=10,G=10 (v18)":>20s} {"N=15,G=10 (v19)":>20s}')
print('-' * 55)
for obs in ['full', 'partial_0.7']:
    v18_vals = [v for (p, o), v in v18_data.items() if p == 'A_larger_budget' and o == obs]
    v19_vals = [v for (p, o), v in v19_data.items() if p == 'A_larger_budget_N15G10' and o == obs]
    v18_mean = np.mean(v18_vals) if v18_vals else None
    v19_mean = np.mean(v19_vals) if v19_vals else None
    print(f'{obs:12s} {v18_mean:.2f} (1/{len(v18_vals)}) {v19_mean:.2f} (0/{len(v19_vals)})')

# === Intern A vs v15 deepseek A (different config) ===
print('\n=== Intern A (N=15, G=10) vs DeepSeek A (N=30, G=20) ===\n')
print(f'{"obs":12s} {"Intern A (N=15)":>20s} {"DS A (N=30)":>15s}')
print('-' * 50)
for obs in ['full', 'partial_0.7']:
    i_vals = [v for (p, o), v in v19_data.items() if p == 'A_larger_budget_N15G10' and o == obs]
    d_vals = load_ds_exp7('A_larger_budget', obs)
    i_mean = np.mean(i_vals) if i_vals else None
    d_mean = np.mean(d_vals) if d_vals else None
    i_ok = sum(1 for v in i_vals if v > 0.5) if i_vals else 0
    d_ok = sum(1 for v in d_vals if v > 0.5) if d_vals else 0
    print(f'{obs:12s} {i_mean:.2f} ({i_ok}/3) {d_mean:.2f} ({d_ok}/3)')