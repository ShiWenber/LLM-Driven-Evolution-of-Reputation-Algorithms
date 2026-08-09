"""Plot 5-seed sweep: 4 obs x 5 seeds trajectories + bar chart comparison vs donate n=3."""
import json, os, statistics
import matplotlib.pyplot as plt

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_AB_n5'
orig_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp1_method_n10'
obs_levels = ['private', 'partial_0.3', 'partial_0.7', 'full']
obs_labels = ['private ($p=0.0$)', 'partial ($p=0.3$)', 'partial ($p=0.7$)', 'full ($p=1.0$)']

def load_trial(base_dir, obs, seed):
    d = os.path.join(base_dir, f'{obs}_seed{seed}')
    if not os.path.exists(d): return None
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    if not files: return None
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    return [g['cooperation_rate_mean'] for g in t['trajectory']]

# Trajectory grid: A/B (5 seeds, top) vs donate (3 seeds, bottom)
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey=True)
for col, obs in enumerate(obs_levels):
    # A/B 5 seeds
    finals_AB = []
    per_seed_AB = []
    for seed in range(5):
        coops = load_trial(base, obs, seed)
        if coops is None: continue
        per_seed_AB.append(coops)
        finals_AB.append(coops[-1])
    # donate 3 seeds
    finals_orig = []
    per_seed_orig = []
    for seed in range(3):
        coops = load_trial(orig_base, obs, seed)
        if coops is None: continue
        per_seed_orig.append(coops)
        finals_orig.append(coops[-1])

    # Top: A/B
    ax = axes[0, col]
    for coops in per_seed_AB:
        ax.plot(range(len(coops)), coops, '-', color='#06A77D', alpha=0.35, linewidth=1.2)
    if per_seed_AB:
        max_len = max(len(c) for c in per_seed_AB)
        mean_curve = [sum(c[i] for c in per_seed_AB if i < len(c)) / sum(1 for c in per_seed_AB if i < len(c)) for i in range(max_len)]
        ax.plot(range(len(mean_curve)), mean_curve, 'o-', color='#06A77D', linewidth=3, markersize=7)
    mean_AB = statistics.mean(finals_AB) if finals_AB else 0
    std_AB = statistics.stdev(finals_AB) if len(finals_AB) > 1 else 0
    n_ok = sum(1 for f in finals_AB if f > 0.5)
    ax.set_title(f'{obs_labels[col]}\nA/B (5 seeds): {mean_AB:.2f}±{std_AB:.2f} ({n_ok}/5 OK)', fontsize=10)
    ax.set_xticks(range(0, 10, 2))
    if col == 0: ax.set_ylabel('Cooperation rate\n(A/B label, 5 seeds)', fontsize=10)
    ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.15)

    # Bottom: donate
    ax = axes[1, col]
    for coops in per_seed_orig:
        ax.plot(range(len(coops)), coops, '-', color='#2E86AB', alpha=0.5, linewidth=1.2)
    if per_seed_orig:
        max_len = max(len(c) for c in per_seed_orig)
        mean_curve = [sum(c[i] for c in per_seed_orig if i < len(c)) / sum(1 for c in per_seed_orig if i < len(c)) for i in range(max_len)]
        ax.plot(range(len(mean_curve)), mean_curve, 's-', color='#2E86AB', linewidth=3, markersize=7)
    mean_orig = statistics.mean(finals_orig) if finals_orig else 0
    std_orig = statistics.stdev(finals_orig) if len(finals_orig) > 1 else 0
    n_ok_orig = sum(1 for f in finals_orig if f > 0.5)
    ax.set_title(f'donate (3 seeds): {mean_orig:.2f}±{std_orig:.2f} ({n_ok_orig}/3 OK)', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xticks(range(0, 10, 2))
    if col == 0: ax.set_ylabel('Cooperation rate\n(donate label, 3 seeds)', fontsize=10)
    ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.15)

plt.suptitle('Exp 6 sweep v2 — A/B (5 seeds, top) vs donate (3 seeds, bottom)',
             fontsize=12, y=1.00)
plt.tight_layout()

out_pdf = base + '/n5_AB_vs_donate_trajectory.pdf'
out_png = base + '/n5_AB_vs_donate_trajectory.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')

# Bar chart
fig2, ax2 = plt.subplots(figsize=(10, 5.5))
import numpy as np
x = np.arange(len(obs_levels))
width = 0.35
means_AB = []; stds_AB = []; noks_AB = []
means_orig = []; stds_orig = []; noks_orig = []
for obs in obs_levels:
    finals_AB = []; finals_orig = []
    for seed in range(5):
        c = load_trial(base, obs, seed)
        if c: finals_AB.append(c[-1])
    for seed in range(3):
        c = load_trial(orig_base, obs, seed)
        if c: finals_orig.append(c[-1])
    means_AB.append(statistics.mean(finals_AB) if finals_AB else 0)
    stds_AB.append(statistics.stdev(finals_AB) if len(finals_AB) > 1 else 0)
    noks_AB.append(sum(1 for f in finals_AB if f > 0.5))
    means_orig.append(statistics.mean(finals_orig) if finals_orig else 0)
    stds_orig.append(statistics.stdev(finals_orig) if len(finals_orig) > 1 else 0)
    noks_orig.append(sum(1 for f in finals_orig if f > 0.5))

b1 = ax2.bar(x - width/2, means_AB, width, yerr=stds_AB, capsize=5,
             color='#06A77D', label='A/B label (n=5)', alpha=0.85)
b2 = ax2.bar(x + width/2, means_orig, width, yerr=stds_orig, capsize=5,
             color='#2E86AB', label='donate label (n=3)', alpha=0.85)
# OK counts
for i, (b, m, s, n_ok, n_total) in enumerate(zip(b1, means_AB, stds_AB, noks_AB, [5]*4)):
    label = f'{m:.2f}\n({n_ok}/5 OK)'
    ax2.text(b.get_x() + b.get_width()/2, m + s + 0.05, label,
             ha='center', fontsize=9, color='#06A77D', fontweight='bold')
for b, m, s, n_ok in zip(b2, means_orig, stds_orig, noks_orig):
    label = f'{m:.2f}\n({n_ok}/3 OK)'
    ax2.text(b.get_x() + b.get_width()/2, m + s + 0.05, label,
             ha='center', fontsize=9, color='#2E86AB', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['private', 'partial 0.3', 'partial 0.7', 'full'], fontsize=11)
ax2.set_ylabel('Mean final cooperation rate', fontsize=11)
ax2.set_title('A/B (n=5) vs donate (n=3) — final-generation mean ± std',
              fontsize=12)
ax2.legend(fontsize=10, loc='upper right')
ax2.set_ylim(0, 1.15)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()

out_pdf2 = base + '/n5_AB_vs_donate_bar.pdf'
out_png2 = base + '/n5_AB_vs_donate_bar.png'
fig2.savefig(out_pdf2, bbox_inches='tight')
fig2.savefig(out_png2, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf2}')
