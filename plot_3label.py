"""Plot 3-way label comparison: donate (n=3) vs A/B (n=5) vs cooperate/defect (n=5)."""
import json, os, statistics
import matplotlib.pyplot as plt
import numpy as np

OBS = ['private', 'partial_0.3', 'partial_0.7', 'full']
def load(base, obs, seed):
    d = os.path.join(base, f'{obs}_seed{seed}')
    if not os.path.exists(d): return None
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    if not files: return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    return [g['cooperation_rate_mean'] for g in t['trajectory']]

def get_stats(base, n_seeds):
    means, stds, noks = [], [], []
    for obs in OBS:
        finals = []
        for seed in range(n_seeds):
            c = load(base, obs, seed)
            if c: finals.append(c[-1])
        means.append(statistics.mean(finals) if finals else 0)
        stds.append(statistics.stdev(finals) if len(finals) > 1 else 0)
        noks.append(sum(1 for f in finals if f > 0.5))
    return means, stds, noks

m_donate, s_donate, ok_donate = get_stats(
    r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp1_method_n10', 3)
m_AB,    s_AB,    ok_AB    = get_stats(
    r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_AB_n5', 5)
m_CD,    s_CD,    ok_CD    = get_stats(
    r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5', 5)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(OBS))
w = 0.27
b1 = ax.bar(x - w, m_donate, w, yerr=s_donate, capsize=4, color='#264653', label='donate/not_donate (n=3)', alpha=0.85)
b2 = ax.bar(x,     m_AB,    w, yerr=s_AB,    capsize=4, color='#06A77D', label='A/B neutral (n=5)',         alpha=0.85)
b3 = ax.bar(x + w, m_CD,    w, yerr=s_CD,    capsize=4, color='#E63946', label='cooperate/defect (n=5)',   alpha=0.85)
for b, m, s, ok, total in [(b1, m_donate, s_donate, ok_donate, 3),
                             (b2, m_AB,    s_AB,    ok_AB,    5),
                             (b3, m_CD,    s_CD,    ok_CD,    5)]:
    for bar, mm, ss, oo in zip(b, m, s, ok):
        ax.text(bar.get_x() + bar.get_width()/2, mm + ss + 0.03, f'{mm:.2f}\n({oo}/{total})',
                ha='center', fontsize=8.5, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['private', 'partial 0.3', 'partial 0.7', 'full'], fontsize=11)
ax.set_ylabel('Mean final cooperation rate', fontsize=11)
ax.set_title('Action-label ablation — 3 label styles, 4 observability conditions',
             fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5\three_label_comparison.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5\three_label_comparison.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')