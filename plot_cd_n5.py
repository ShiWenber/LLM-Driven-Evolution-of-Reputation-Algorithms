"""Plot n=5 CD sweep: trajectory grid + bar chart, side-by-side with A/B n=5."""
import json, os, statistics
import matplotlib.pyplot as plt
import numpy as np

OBS = ['private', 'partial_0.3', 'partial_0.7', 'full']
COLORS = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']

def load(base, obs, seed):
    d = os.path.join(base, f'{obs}_seed{seed}')
    if not os.path.exists(d): return None
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    if not files: return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    return [g['cooperation_rate_mean'] for g in t['trajectory']]

# --- A/B n=5 (existing data) ---
AB = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_AB_n5'
# --- cooperate/defect n=5 (new data) ---
CD = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5'
# --- donate n=3 (original data) ---
DON = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp1_method_n10'

def get_stats(base, n_seeds):
    means, stds, finals_list = [], [], []
    for obs in OBS:
        finals = []
        for seed in range(n_seeds):
            c = load(base, obs, seed)
            if c: finals.append(c[-1])
        finals_list.append(finals)
        means.append(statistics.mean(finals) if finals else 0)
        stds.append(statistics.stdev(finals) if len(finals) > 1 else 0)
    return means, stds, finals_list

m_don, s_don, fl_don = get_stats(DON, 3)
m_AB,  s_AB,  fl_AB  = get_stats(AB, 5)
m_CD,  s_CD,  fl_CD  = get_stats(CD, 5)

# ====================================================================
# Plot 1: Trajectory grid — 3 rows (donate n=3, A/B n=5, CD n=5) x 4 cols (obs)
# ====================================================================
def get_traj(base, obs, seed):
    d = os.path.join(base, f'{obs}_seed{seed}')
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    return t  # full trajectory dict

fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True, sharey=True)
for col, obs in enumerate(OBS):
    for row, (base, label, n_seeds) in enumerate([
        (DON, 'donate/not_donate (n=3)', 3),
        (AB,  'A/B neutral (n=5)',       5),
        (CD,  'cooperate/defect (n=5)',  5),
    ]):
        ax = axes[row, col]
        for seed in range(n_seeds):
            try:
                t = get_traj(base, obs, seed)
                coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
                gens = list(range(len(coops)))
                ax.plot(gens, coops, color=COLORS[col], alpha=0.4, linewidth=1.2)
            except Exception as e:
                pass
        # mean over seeds
        all_traj = []
        for seed in range(n_seeds):
            try:
                t = get_traj(base, obs, seed)
                coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
                all_traj.append(coops)
            except: pass
        if all_traj:
            L = min(len(c) for c in all_traj)
            arr = np.array([c[:L] for c in all_traj])
            mean = arr.mean(axis=0)
            ax.plot(range(L), mean, color='black', linewidth=2.5, label='mean over seeds')
            if col == 0: ax.set_ylabel(label, fontsize=10)
            if row == 0: ax.set_title(obs, fontsize=11, fontweight='bold')
            if row == 2: ax.set_xlabel('generation', fontsize=9)
            if col == 3: ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

fig.suptitle('Evolution of mean cooperation rate — 3 label conditions x 4 observability levels', fontsize=13)
plt.tight_layout()
out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5\n5_CD_AB_donate_trajectory.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5\n5_CD_AB_donate_trajectory.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')

# ====================================================================
# Plot 2: Bar chart — 3 label conditions x 4 obs
# ====================================================================
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(OBS))
w = 0.27
b1 = ax.bar(x - w, m_don, w, yerr=s_don, capsize=4, color='#264653', label='donate/not_donate (n=3)', alpha=0.85)
b2 = ax.bar(x,     m_AB,  w, yerr=s_AB,  capsize=4, color='#06A77D', label='A/B neutral (n=5)',         alpha=0.85)
b3 = ax.bar(x + w, m_CD,  w, yerr=s_CD,  capsize=4, color='#E63946', label='cooperate/defect (n=5)',   alpha=0.85)
for b, m, s, fl, total in [(b1, m_don, s_don, fl_don, 3),
                             (b2, m_AB,  s_AB,  fl_AB,  5),
                             (b3, m_CD,  s_CD,  fl_CD,  5)]:
    for bar, mm, ss, fins in zip(b, m, s, fl):
        n_ok = sum(1 for f in fins if f > 0.5)
        ax.text(bar.get_x() + bar.get_width()/2, mm + ss + 0.03,
                f'{mm:.2f}\n({n_ok}/{total})',
                ha='center', fontsize=8.5, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['private', 'partial 0.3', 'partial 0.7', 'full'], fontsize=11)
ax.set_ylabel('Mean final cooperation rate', fontsize=11)
ax.set_title('Action-label ablation — 3 label styles, 4 observability conditions', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5\n5_CD_AB_donate_bar.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5\n5_CD_AB_donate_bar.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')
print('done')