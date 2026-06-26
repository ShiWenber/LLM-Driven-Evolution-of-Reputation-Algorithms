"""Plot exp7 results: 4-exp bar chart + complexity markers comparison."""
import json, os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling'

EXPS = [
    ('v15_baseline', 'v15 baseline (N=15,G=10)', 'cd_n5', None, None),
    ('A_larger_budget', 'A: N=30,G=20', 'exp7', None, None),
    ('B_recent_window', 'B: recent-window=5', 'exp7', None, None),
    ('C_reputation_noise', 'C: reputation noise=0.1', 'exp7', None, None),
    ('D_exploration_mutation', 'D: exploration mutation', 'exp7', None, None),
    ('E_all_combined', 'E: all combined', 'exp7', None, None),
]

# v15 baseline (use CD n=5 data)
def load_cd_n5(obs, seed):
    d = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5'
    full_d = os.path.join(d, f'{obs}_seed{seed}')
    files = [f for f in os.listdir(full_d) if f.startswith('evo_') and f.endswith('.json')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(full_d, f)), reverse=True)
    with open(os.path.join(full_d, files[0])) as f:
        t = json.load(f)
    return t['trajectory'][-1]['cooperation_rate_mean'], t['final_population']

def load_exp7(subdir, obs, seed):
    d = os.path.join(base, subdir, f'{obs}_seed{seed}')
    if not os.path.exists(d): return None, None
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    if not files: return None, None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    return t['trajectory'][-1]['cooperation_rate_mean'], t['final_population']

# Build table
data = {}
for subdir, label, kind, _, _ in EXPS:
    data[subdir] = {'label': label, 'full': [], 'p0.7': [],
                    'full_complex': {'iter': 0, 'decay': 0, 'n': 0},
                    'p07_complex': {'iter': 0, 'decay': 0, 'n': 0}}
    for seed in range(3):
        if kind == 'cd_n5':
            coop_f, pop_f = load_cd_n5('full', seed)
            coop_p, pop_p = load_cd_n5('partial_0.7', seed)
        else:
            coop_f, pop_f = load_exp7(subdir, 'full', seed)
            coop_p, pop_p = load_exp7(subdir, 'partial_0.7', seed)
        if coop_f is not None:
            data[subdir]['full'].append(coop_f)
            for a in pop_f:
                code = a['code']
                if 'for ' in code and ' in ' in code:
                    data[subdir]['full_complex']['iter'] += 1
                if any(x in code for x in ['decay', 'alpha', 'beta', 'gamma']):
                    data[subdir]['full_complex']['decay'] += 1
                data[subdir]['full_complex']['n'] += 1
        if coop_p is not None:
            data[subdir]['p0.7'].append(coop_p)
            for a in pop_p:
                code = a['code']
                if 'for ' in code and ' in ' in code:
                    data[subdir]['p07_complex']['iter'] += 1
                if any(x in code for x in ['decay', 'alpha', 'beta', 'gamma']):
                    data[subdir]['p07_complex']['decay'] += 1
                data[subdir]['p07_complex']['n'] += 1

# Plot 1: bar chart — 6 exps x 2 obs
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(EXPS))
w = 0.35
labels = [data[s]['label'] for s, _, _, _, _ in EXPS]
full_means = [np.mean(data[s]['full']) if data[s]['full'] else 0 for s, _, _, _, _ in EXPS]
full_stds = [np.std(data[s]['full']) if len(data[s]['full']) > 1 else 0 for s, _, _, _, _ in EXPS]
p07_means = [np.mean(data[s]['p0.7']) if data[s]['p0.7'] else 0 for s, _, _, _, _ in EXPS]
p07_stds = [np.std(data[s]['p0.7']) if len(data[s]['p0.7']) > 1 else 0 for s, _, _, _, _ in EXPS]
ax.bar(x - w/2, full_means, w, yerr=full_stds, capsize=4, color='#264653', label='full obs', alpha=0.85)
ax.bar(x + w/2, p07_means, w, yerr=p07_stds, capsize=4, color='#E76F51', label='partial 0.7', alpha=0.85)
for i, (s, _, _, _, _) in enumerate(EXPS):
    n_ok_f = sum(1 for c in data[s]['full'] if c > 0.5)
    n_ok_p = sum(1 for c in data[s]['p0.7'] if c > 0.5)
    ax.text(i - w/2, full_means[i] + full_stds[i] + 0.02, f'{full_means[i]:.2f}\n({n_ok_f}/3)',
            ha='center', fontsize=8.5, fontweight='bold')
    ax.text(i + w/2, p07_means[i] + p07_stds[i] + 0.02, f'{p07_means[i]:.2f}\n({n_ok_p}/3)',
            ha='center', fontsize=8.5, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Mean final cooperation rate', fontsize=11)
ax.set_title('Algorithmic-complexity-ceiling probes (Exp 7): 6 conditions, 3 seeds, 2 obs', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\exp7_bar.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\exp7_bar.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')

# Plot 2: complexity markers (iter% + decay%) per exp
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(EXPS))
labels = [data[s]['label'] for s, _, _, _, _ in EXPS]
iter_f = [100 * data[s]['full_complex']['iter'] / max(1, data[s]['full_complex']['n']) for s, _, _, _, _ in EXPS]
decay_f = [100 * data[s]['full_complex']['decay'] / max(1, data[s]['full_complex']['n']) for s, _, _, _, _ in EXPS]
iter_p = [100 * data[s]['p07_complex']['iter'] / max(1, data[s]['p07_complex']['n']) for s, _, _, _, _ in EXPS]
decay_p = [100 * data[s]['p07_complex']['decay'] / max(1, data[s]['p07_complex']['n']) for s, _, _, _, _ in EXPS]
ax.bar(x - 0.3, iter_f, 0.25, label='full: uses for-loop iter', color='#2A9D8F', alpha=0.85)
ax.bar(x - 0.05, decay_f, 0.25, label='full: uses decay/alpha/beta', color='#264653', alpha=0.85)
ax.bar(x + 0.2, iter_p, 0.25, label='partial 0.7: uses iter', color='#E9C46A', alpha=0.85)
ax.bar(x + 0.45, decay_p, 0.25, label='partial 0.7: uses decay/alpha', color='#E76F51', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('% of final-pop strategies', fontsize=11)
ax.set_title('Strategy complexity markers (iter, decay/alpha) by condition', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\exp7_complexity.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\exp7_complexity.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')

# Save the data table to JSON
import json
table = {s: {k: v for k, v in data[s].items() if k != 'full_complex' and k != 'p07_complex'} for s in data}
for s in data:
    table[s]['full_iter_pct'] = iter_f[list(data).keys().index(s)] if False else 100 * data[s]['full_complex']['iter'] / max(1, data[s]['full_complex']['n'])
    table[s]['full_decay_pct'] = 100 * data[s]['full_complex']['decay'] / max(1, data[s]['full_complex']['n'])
    table[s]['p07_iter_pct'] = 100 * data[s]['p07_complex']['iter'] / max(1, data[s]['p07_complex']['n'])
    table[s]['p07_decay_pct'] = 100 * data[s]['p07_complex']['decay'] / max(1, data[s]['p07_complex']['n'])
with open(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\exp7_table.json', 'w') as f:
    json.dump(table, f, indent=2)
print('Saved exp7_table.json')