"""Plot exp6_sweep_AB results: 4-panel obs x 3-seeds trajectory grid."""
import json, os
import matplotlib.pyplot as plt
import statistics

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_AB'
obs_levels = ['private', 'partial_0.3', 'partial_0.7', 'full']
obs_labels = ['private (p=0.0)', 'partial_0.3 (p=0.3)', 'partial_0.7 (p=0.7)', 'full (p=1.0)']

# Compare with original (donate) data
orig_base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp1_method_n10'

fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey=True)

for col, obs in enumerate(obs_levels):
    # A/B trials
    finals_AB = []
    coops_per_seed_AB = []
    for seed in range(3):
        d = os.path.join(base, f'{obs}_seed{seed}')
        if not os.path.exists(d):
            continue
        evo_files = [f for f in os.listdir(d) if f.startswith('evo_')]
        if not evo_files:
            continue
        with open(os.path.join(d, evo_files[0])) as f:
            t = json.load(f)
        coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
        coops_per_seed_AB.append(coops)
        finals_AB.append(coops[-1])

    # Original (donate) n=3 trials
    finals_orig = []
    coops_per_seed_orig = []
    if obs == 'private':
        orig_obs = 'private'
    else:
        orig_obs = obs
    for seed in range(3):
        d = os.path.join(orig_base, f'{orig_obs}_seed{seed}')
        if not os.path.exists(d):
            continue
        evo_files = [f for f in os.listdir(d) if f.startswith('evo_')]
        if not evo_files:
            continue
        with open(os.path.join(d, evo_files[0])) as f:
            t = json.load(f)
        coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
        coops_per_seed_orig.append(coops)
        finals_orig.append(coops[-1])

    # Top row: A/B
    ax = axes[0, col]
    for i, coops in enumerate(coops_per_seed_AB):
        ax.plot(range(len(coops)), coops, '-', color='#06A77D', alpha=0.5, linewidth=1.5)
    if coops_per_seed_AB:
        # Mean curve
        max_len = max(len(c) for c in coops_per_seed_AB)
        mean_curve = []
        for i in range(max_len):
            vals = [c[i] for c in coops_per_seed_AB if i < len(c)]
            mean_curve.append(sum(vals) / len(vals))
        ax.plot(range(len(mean_curve)), mean_curve, 'o-', color='#06A77D', linewidth=3, markersize=8, label='mean of 3 seeds')
    mean_AB = statistics.mean(finals_AB) if finals_AB else 0
    std_AB = statistics.stdev(finals_AB) if len(finals_AB) > 1 else 0
    ax.set_title(f'{obs_labels[col]}\nA/B label: gen-10 = {mean_AB:.2f} ± {std_AB:.2f}', fontsize=10)
    ax.set_xticks(range(0, 10, 2))
    if col == 0:
        ax.set_ylabel('Cooperation rate\n(A/B label, this work)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    if col == 0:
        ax.legend(loc='lower right', fontsize=8)

    # Bottom row: original (donate)
    ax = axes[1, col]
    for i, coops in enumerate(coops_per_seed_orig):
        ax.plot(range(len(coops)), coops, '-', color='#2E86AB', alpha=0.5, linewidth=1.5)
    if coops_per_seed_orig:
        max_len = max(len(c) for c in coops_per_seed_orig)
        mean_curve = []
        for i in range(max_len):
            vals = [c[i] for c in coops_per_seed_orig if i < len(c)]
            mean_curve.append(sum(vals) / len(vals))
        ax.plot(range(len(mean_curve)), mean_curve, 'o-', color='#2E86AB', linewidth=3, markersize=8, label='mean of 3 seeds')
    mean_orig = statistics.mean(finals_orig) if finals_orig else 0
    std_orig = statistics.stdev(finals_orig) if len(finals_orig) > 1 else 0
    ax.set_title(f'gen-10 = {mean_orig:.2f} ± {std_orig:.2f}', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.set_xticks(range(0, 10, 2))
    if col == 0:
        ax.set_ylabel('Cooperation rate\n(donate label, original)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    if col == 0:
        ax.legend(loc='lower right', fontsize=8)

plt.suptitle('Exp 6 (A/B label, top) vs original (donate label, bottom) — 3 seeds per condition', fontsize=13, y=1.00)
plt.tight_layout()

out_pdf = base + '/sweep_AB_vs_donate_3seeds.pdf'
out_png = base + '/sweep_AB_vs_donate_3seeds.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')
print(f'Saved {out_png}')

# Also save a summary CSV
import csv
with open(base + '/summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['observability', 'label', 'mean_final_coop', 'std_final_coop', 'n_seeds'])
    for obs in obs_levels:
        d = os.path.join(base, f'{obs}_seed0')
        if not os.path.exists(d):
            continue
        evo_files = [f for f in os.listdir(d) if f.startswith('evo_')]
        if not evo_files:
            continue
        # Compute mean over 3 seeds
        finals = []
        for seed in range(3):
            dd = os.path.join(base, f'{obs}_seed{seed}')
            ef = [x for x in os.listdir(dd) if x.startswith('evo_')]
            if not ef: continue
            with open(os.path.join(dd, ef[0])) as ff:
                tt = json.load(ff)
            finals.append(tt['trajectory'][-1]['cooperation_rate_mean'])
        w.writerow([obs, 'A/B', round(statistics.mean(finals), 3), round(statistics.stdev(finals), 3) if len(finals) > 1 else 0, len(finals)])
    for obs in obs_levels:
        orig_obs = obs
        d = os.path.join(orig_base, f'{orig_obs}_seed0')
        if not os.path.exists(d):
            continue
        finals = []
        for seed in range(3):
            dd = os.path.join(orig_base, f'{orig_obs}_seed{seed}')
            if not os.path.exists(dd): continue
            ef = [x for x in os.listdir(dd) if x.startswith('evo_')]
            if not ef: continue
            with open(os.path.join(dd, ef[0])) as ff:
                tt = json.load(ff)
            finals.append(tt['trajectory'][-1]['cooperation_rate_mean'])
        w.writerow([obs, 'donate', round(statistics.mean(finals), 3), round(statistics.stdev(finals), 3) if len(finals) > 1 else 0, len(finals)])
print('Saved summary.csv')
