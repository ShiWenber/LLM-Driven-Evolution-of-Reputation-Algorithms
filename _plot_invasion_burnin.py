"""Plot invasion threshold scaling: burn-in vs no-burn-in, N=15,30,100.

Key story: original no-burn-in showed n* ≈ 13-20% (BISTABILITY claim).
Proper burn-in shows n* >> 50%, scaling with N. Old result was a
cold-start artifact.
"""
import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
base = ROOT / 'results' / 'quantitative_baseline' / 'invasion'

LEADING_EIGHT = ['IS', 'SS', 'SJ', 'SC', 'SH', 'IS_PLUS', 'SS_PLUS', 'SJ_PLUS']
INVADERS = ['llm_winner', 'ALLD']
N_VALUES = list(range(1, 16))


def load_traj(invader, ess, n, pop_size, burn_in, seed=42):
    sub = f'N{pop_size}_bi{burn_in}'
    p = base / sub / f'{invader}_vs_{ess}_n{n}_seed{seed}' / 'invasion.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))['trajectory']


# ============================================================
# Plot 1: With-burn-in threshold curves for N=15, 30, 100
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

pop_sizes = [15, 30, 100]
colors_N = {15: '#1f77b4', 30: '#2ca02c', 100: '#d62728'}
markers_N = {15: 'o', 30: 's', 100: '^'}

for N in pop_sizes:
    for invader in INVADERS:
        n_means = []
        n_stds = []
        n_labels = []
        for n in range(1, min(N, 100) + 1):
            finals = []
            for ess in LEADING_EIGHT:
                traj = load_traj(invader, ess, n, N, burn_in=10)
                if traj is not None:
                    finals.append(traj[-1]['invader_freq'])
            if finals:
                n_means.append(statistics.mean(finals))
                n_stds.append(statistics.stdev(finals) if len(finals) > 1 else 0)
                n_labels.append(n)
        if n_means:
            ls = '-' if invader == 'llm_winner' else '--'
            label = f'N={N}, {invader.replace("_", " ")}'
            ax.plot(n_labels, n_means, ls, marker=markers_N[N], markersize=4,
                    color=colors_N[N], alpha=0.7, label=label, linewidth=1.5)

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
ax.set_xlabel('Initial invader count n', fontsize=11)
ax.set_ylabel('Final invader frequency (mean across 8 ESS)', fontsize=11)
ax.set_title('Invasion threshold WITH burn-in=10 (all-good initial state): '
             'n* scales with N', fontsize=12)
ax.set_ylim(-0.05, 1.10)
ax.grid(alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
out1_png = base / 'invasion_burnin_N_scaling.png'
out1_pdf = base / 'invasion_burnin_N_scaling.pdf'
fig.savefig(out1_png, dpi=150)
fig.savefig(out1_pdf)
plt.close(fig)
print(f'saved: {out1_png} ({out1_png.stat().st_size/1024:.1f} KB)')


# ============================================================
# Plot 2: Side-by-side — NO burn-in vs WITH burn-in (N=15)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Subplot 1: no burn-in (data from old run, stored at base root)
for ax_i, (burn_in, title_suffix, subdir) in enumerate([
    (0, 'NO burn-in (cold start, rep=0.5 default)', ''),
    (10, 'WITH burn-in=10 (all-good rep initial state)', 'N15_bi10'),
]):
    ax = axes[ax_i]
    for invader in INVADERS:
        n_means = []
        n_stds = []
        n_labels = []
        for n in range(1, 8):
            finals = []
            for ess in LEADING_EIGHT:
                if subdir:
                    p = base / subdir / f'{invader}_vs_{ess}_n{n}_seed42' / 'invasion.json'
                else:
                    p = base / f'{invader}_vs_{ess}_n{n}_seed42' / 'invasion.json'
                if p.exists():
                    traj = json.loads(p.read_text(encoding='utf-8'))['trajectory']
                    finals.append(traj[-1]['invader_freq'])
            if finals:
                n_means.append(statistics.mean(finals))
                n_stds.append(statistics.stdev(finals) if len(finals) > 1 else 0)
                n_labels.append(n)
        if n_means:
            color = '#1f77b4' if invader == 'llm_winner' else '#d62728'
            marker = 'o' if invader == 'llm_winner' else 's'
            label = invader.replace('_', ' ')
            ax.errorbar(n_labels, n_means, yerr=n_stds, marker=marker,
                        markersize=8, linewidth=2, capsize=4,
                        color=color, label=label, alpha=0.85)

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Initial invader count n (out of 15)', fontsize=11)
    ax.set_title(f'N=15, {title_suffix}', fontsize=12)
    ax.set_xticks(range(1, 8))
    ax.set_ylim(-0.05, 1.10)
    ax.grid(alpha=0.3)
    ax.legend(loc='center right', fontsize=10)
    if ax_i == 0:
        ax.set_ylabel('Final invader frequency (mean ± std across 8 ESS)', fontsize=11)
        # Add n* annotation
        ax.text(0.5, 0.05, 'n* ∈ [2, 3] (13-20%)\nARTIFACT from cold start\n(rep default 0.5)',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    else:
        ax.text(0.5, 0.05, 'n* ∈ [12, 13] (80-87%)\nTRUE ESS stability\nwith all-good burn-in',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

fig.suptitle('Burn-in matters: cold-start rep default=0.5 artificially lowers ESS threshold\n'
             '(data: ALLD and LLM_winner overlap completely in BOTH conditions)',
             fontsize=11, y=1.02)
fig.tight_layout()
out2_png = base / 'invasion_burnin_vs_noburnin.png'
out2_pdf = base / 'invasion_burnin_vs_noburnin.pdf'
fig.savefig(out2_png, dpi=150, bbox_inches='tight')
fig.savefig(out2_pdf, bbox_inches='tight')
plt.close(fig)
print(f'saved: {out2_png} ({out2_png.stat().st_size/1024:.1f} KB)')


# ============================================================
# Summary
# ============================================================
print('\n=== Threshold summary (with burn-in=10) ===')
for N in pop_sizes:
    for invader in INVADERS:
        for n in range(1, min(N, 100) + 1):
            finals = []
            for ess in LEADING_EIGHT:
                traj = load_traj(invader, ess, n, N, burn_in=10)
                if traj is not None:
                    finals.append(traj[-1]['invader_freq'])
            if finals:
                m = statistics.mean(finals)
                if m >= 0.9:
                    print(f'  N={N}, {invader}: n={n}/{N} ({n/N*100:.0f}%) -> fixate (g_end={m:.3f})')
                    break  # found first fixate
        else:
            # no fixate found
            max_n = min(N, 100)
            traj = load_traj(invader, LEADING_EIGHT[0], max_n, N, burn_in=10)
            if traj:
                m = traj[-1]['invader_freq']
                print(f'  N={N}, {invader}: n* > {max_n}/{N} ({max_n/N*100:.0f}%) '
                      f'(max g_end at n={max_n}: {m:.3f})')
