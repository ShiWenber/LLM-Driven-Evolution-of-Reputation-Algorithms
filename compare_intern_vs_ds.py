"""Compare Intern vs DeepSeek-V4-Flash across ceiling probes.
Data sources:
  - Intern: results/exp8_intern_ceiling/ (skip A — all timeout, use v15 v4-flash data instead)
  - DeepSeek-V4-Flash: results/exp7_algorithmic_ceiling/

Output: comparison table + side-by-side bar chart.
"""
import json, os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

intern_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp8_intern_ceiling'
ds_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling'

# Intern data: B, C, D, E (A skipped — too slow)
INTERN_PROBES = {
    'B_recent_window':    'B recent-window=5',
    'C_reputation_noise': 'C reputation noise=0.1',
    'D_exploration':      'D exploration mutation',
    'E_baseline':         'E baseline (N=15,G=10)',
}

# DeepSeek-V4-Flash data (from v15 exp7)
DS_PROBES = {
    'A_larger_budget':    'A N=30,G=20 (skip Intern)',
    'B_recent_window':    'B recent-window=5',
    'C_reputation_noise': 'C reputation noise=0.1',
    'D_exploration_mutation': 'D exploration mutation',
    'E_all_combined':     'E all combined (skip Intern)',
}

def load_trial(base, subdir, obs, seed):
    d = os.path.join(base, subdir, f'{obs}_seed{seed}')
    if not os.path.exists(d): return None
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    if not files: return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        return json.load(f)

def get_stats(base, subdir, obs_list, n_seeds=3):
    finals, classes = [], []
    for obs in obs_list:
        for seed in range(n_seeds):
            t = load_trial(base, subdir, obs, seed)
            if t is None or not t.get('trajectory'): continue
            finals.append(t['trajectory'][-1]['cooperation_rate_mean'])
            for a in t['final_population']:
                code = a.get('code', '')
                # Complexity markers
                has_iter = 'for ' in code and ' in ' in code
                has_decay = any(x in code for x in ['decay', 'alpha', 'beta', 'gamma'])
                has_window = 'recent_window' in code
                classes.append({'iter': has_iter, 'decay': has_decay, 'window': has_window})
    return finals, classes

# === Build comparison table ===
obs_list = ['full', 'partial_0.7']
print('=== Intern (Paratera) vs DeepSeek-V4-Flash ceiling probes ===\n')
print(f'{"probe":35s} {"Intern full mean":>20s} {"Intern p0.7 mean":>20s} {"DS full mean":>15s} {"DS p0.7 mean":>15s}')
print('-' * 110)

rows = []
for ds_sub, label in DS_PROBES.items():
    intern_sub = ds_sub.replace('D_exploration_mutation', 'D_exploration').replace('E_all_combined', 'E_baseline')
    if intern_sub == 'A_larger_budget':
        # Use v15 deepseek A
        ds_f, _ = get_stats(ds_base, ds_sub, ['full'], 3)
        ds_p, _ = get_stats(ds_base, ds_sub, ['partial_0.7'], 3)
        i_f = [None]; i_p = [None]
        label = 'A N=30,G=20 (Intern skip, too slow)'
    else:
        i_f, _ = get_stats(intern_base, intern_sub, ['full'], 3)
        i_p, _ = get_stats(intern_base, intern_sub, ['partial_0.7'], 3)
        ds_f, _ = get_stats(ds_base, ds_sub, ['full'], 3)
        ds_p, _ = get_stats(ds_base, ds_sub, ['partial_0.7'], 3)
    i_f_mean = np.mean(i_f) if i_f and any(x is not None for x in i_f) else None
    i_p_mean = np.mean(i_p) if i_p and any(x is not None for x in i_p) else None
    ds_f_mean = np.mean(ds_f) if ds_f else None
    ds_p_mean = np.mean(ds_p) if ds_p else None
    i_f_ok = sum(1 for x in i_f if x is not None and x > 0.5) if i_f else 0
    i_p_ok = sum(1 for x in i_p if x is not None and x > 0.5) if i_p else 0
    ds_f_ok = sum(1 for x in ds_f if x > 0.5) if ds_f else 0
    ds_p_ok = sum(1 for x in ds_p if x > 0.5) if ds_p else 0
    i_f_s = f'{i_f_mean:.2f} ({i_f_ok}/3)' if i_f_mean is not None else 'skip'
    i_p_s = f'{i_p_mean:.2f} ({i_p_ok}/3)' if i_p_mean is not None else 'skip'
    print(f'{label:35s} {i_f_s:>20s} {i_p_s:>20s} {ds_f_mean:.2f} ({ds_f_ok}/3) {ds_p_mean:.2f} ({ds_p_ok}/3)')
    rows.append((label, i_f_mean, i_p_mean, ds_f_mean, ds_p_mean, i_f_ok, i_p_ok, ds_f_ok, ds_p_ok))

# Save as JSON
out_json = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp8_intern_ceiling\cross_llm_compare.json'
import json
with open(out_json, 'w') as f:
    json.dump([{'label': r[0], 'intern_full': r[1], 'intern_p07': r[2],
                 'ds_full': r[3], 'ds_p07': r[4],
                 'intern_full_ok': r[5], 'intern_p07_ok': r[6],
                 'ds_full_ok': r[7], 'ds_p07_ok': r[8]} for r in rows], f, indent=2)
print(f'\nSaved {out_json}')

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
labels = [r[0].replace(' (skip Intern, too slow)', '') for r in rows]
x = np.arange(len(labels))
w = 0.35
# Full
ax = axes[0]
i_f_vals = [r[1] if r[1] is not None else 0 for r in rows]
ds_f_vals = [r[3] for r in rows]
ax.bar(x - w/2, i_f_vals, w, label='Intern-S2-Preview (Paratera)', color='#2A9D8F', alpha=0.85)
ax.bar(x + w/2, ds_f_vals, w, label='DeepSeek-V4-Flash (v15)', color='#E76F51', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_title('full obs', fontsize=11)
ax.set_ylabel('mean final cooperation rate', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)
# p0.7
ax = axes[1]
i_p_vals = [r[2] if r[2] is not None else 0 for r in rows]
ds_p_vals = [r[4] for r in rows]
ax.bar(x - w/2, i_p_vals, w, label='Intern-S2-Preview', color='#2A9D8F', alpha=0.85)
ax.bar(x + w/2, ds_p_vals, w, label='DeepSeek-V4-Flash', color='#E76F51', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
ax.set_title('partial 0.7 obs', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)
fig.suptitle('Cross-LLM ceiling probes: Intern (Paratera) vs DeepSeek-V4-Flash (v15)', fontsize=12)
plt.tight_layout()
out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp8_intern_ceiling\cross_llm_compare.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp8_intern_ceiling\cross_llm_compare.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')