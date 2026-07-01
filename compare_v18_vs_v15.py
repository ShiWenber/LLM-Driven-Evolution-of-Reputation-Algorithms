"""v18 Intern ceiling: 30/30 trials OK. Generate updated comparison table."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

v18_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp8_intern_ceiling_v18'
ds_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling'

# Load v18 manifest
with open(os.path.join(v18_base, '_manifest.json')) as f:
    v18 = json.load(f)

# v15 deepseek manifest is split across subdirs; load by probe
def load_ds_stats(probe_subdir, obs):
    finals = []
    for seed in range(3):
        d = os.path.join(ds_base, probe_subdir, f'{obs}_seed{seed}')
        if not os.path.exists(d): continue
        files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
        if not files: continue
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        with open(os.path.join(d, files[0])) as f:
            t = json.load(f)
        if t.get('trajectory'):
            finals.append(t['trajectory'][-1]['cooperation_rate_mean'])
    return finals

# Build table
PROBES = ['A_larger_budget', 'B_recent_window', 'C_reputation_noise', 'D_exploration', 'E_baseline']
PROBE_LABELS = {'A_larger_budget': 'A: N=30,G=20 (DS) / uniform small (Intern)',
                'B_recent_window': 'B: recent-window=5',
                'C_reputation_noise': 'C: reputation noise=0.1',
                'D_exploration': 'D: exploration mutation',
                'E_baseline': 'E: baseline / kitchen-sink'}
OBS = ['full', 'partial_0.7']

# Load v18
v18_data = {}
for m in v18:
    if not m.get('ok'): continue
    v18_data[(m['probe'], m['obs'])] = m['coop_final']

# Load v15 deepseek (A,B,C,D) and v17 kitchen-sink (E)
v15_data = {}
for probe, sub in [('A_larger_budget', 'A_larger_budget'),
                     ('B_recent_window', 'B_recent_window'),
                     ('C_reputation_noise', 'C_reputation_noise'),
                     ('D_exploration', 'D_exploration_mutation'),  # v15 used different name
                     ('E_baseline', 'E_all_combined')]:  # v15 E was kitchen-sink
    for obs in OBS:
        finals = load_ds_stats(sub, obs)
        if finals:
            v15_data[(probe, obs)] = finals

# Print table
print('=== v18 Intern vs v15 DeepSeek-V4-Flash (5 probe x 2 obs, n=3) ===\n')
print(f'{"probe":50s} {"obs":12s} {"Intern mean (OK)":>20s} {"DS mean (OK)":>20s}')
print('-' * 110)
for probe in PROBES:
    for obs in OBS:
        i_vals = [v18_data[(p, o)] for p, o in v18_data if p == probe and o == obs]
        i_mean = np.mean(i_vals) if i_vals else None
        i_ok = sum(1 for v in i_vals if v > 0.5) if i_vals else 0
        d_vals = v15_data.get((probe, obs), [])
        d_mean = np.mean(d_vals) if d_vals else None
        d_ok = sum(1 for v in d_vals if v > 0.5) if d_vals else 0
        i_s = f'{i_mean:.2f} ({i_ok}/3)' if i_mean is not None else 'skip'
        d_s = f'{d_mean:.2f} ({d_ok}/3)' if d_mean is not None else 'skip'
        print(f'{PROBE_LABELS[probe]:50s} {obs:12s} {i_s:>20s} {d_s:>20s}')
    print()

# Save table JSON
out = {}
for probe in PROBES:
    out[probe] = {}
    for obs in OBS:
        i_vals = [v18_data[(p, o)] for p, o in v18_data if p == probe and o == obs]
        d_vals = v15_data.get((probe, obs), [])
        out[probe][obs] = {
            'intern': {'mean': float(np.mean(i_vals)) if i_vals else None,
                        'ok_n': sum(1 for v in i_vals if v > 0.5),
                        'values': i_vals},
            'ds': {'mean': float(np.mean(d_vals)) if d_vals else None,
                   'ok_n': sum(1 for v in d_vals if v > 0.5),
                   'values': d_vals},
        }
with open(os.path.join(v18_base, 'v18_vs_v15_compare.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f'Saved {v18_base}/v18_vs_v15_compare.json')

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
labels = [PROBE_LABELS[p] for p in PROBES]
x = np.arange(len(PROBES))
w = 0.35
for i, obs in enumerate(OBS):
    ax = axes[i]
    i_means = [out[p][obs]['intern']['mean'] or 0 for p in PROBES]
    d_means = [out[p][obs]['ds']['mean'] or 0 for p in PROBES]
    i_ok = [out[p][obs]['intern']['ok_n'] for p in PROBES]
    d_ok = [out[p][obs]['ds']['ok_n'] for p in PROBES]
    ax.bar(x - w/2, i_means, w, label='Intern-S2-Preview (v18, N=10,G=10)', color='#2A9D8F', alpha=0.85)
    ax.bar(x + w/2, d_means, w, label='DeepSeek-V4-Flash (v15, N=15,G=10 / N=30,G=20)', color='#E76F51', alpha=0.85)
    for j, (im, dm, iok, dok) in enumerate(zip(i_means, d_means, i_ok, d_ok)):
        ax.text(j - w/2, im + 0.03, f'{im:.2f}\n({iok}/3)', ha='center', fontsize=8.5, fontweight='bold')
        ax.text(j + w/2, dm + 0.03, f'{dm:.2f}\n({dok}/3)', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_title(f'{obs} obs', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    if i == 0:
        ax.set_ylabel('mean final cooperation rate', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
fig.suptitle('v18 Intern-S2-Preview (Paratera) vs v15 DeepSeek-V4-Flash — 5 probe × 2 obs × 3 seeds', fontsize=12)
plt.tight_layout()
out_pdf = os.path.join(v18_base, 'v18_vs_v15_compare.pdf')
out_png = os.path.join(v18_base, 'v18_vs_v15_compare.png')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')