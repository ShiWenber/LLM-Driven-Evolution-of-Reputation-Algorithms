"""Plot invasion comparison: ALLD vs LLM_winner across 8 leading-eight ESS.

Two plots:
  1. Trajectory grid: 2x4 subplots, one per ESS, lines for each n, two
     line styles (solid = LLM_winner, dashed = ALLD). Shows dynamics.
  2. Threshold chart: final invader_freq vs n, for both invaders, averaged
     across ESS. Shows the n* threshold clearly.
"""
import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
base = ROOT / 'results' / 'quantitative_baseline' / 'invasion'

LEADING_EIGHT = ['IS', 'SS', 'SJ', 'SC', 'SH', 'IS_PLUS', 'SS_PLUS', 'SJ_PLUS']
ESS_DISPLAY = {'IS_PLUS': 'IS+', 'SS_PLUS': 'SS+', 'SJ_PLUS': 'SJ+'}
INVADERS = ['llm_winner', 'ALLD']
DISPLAY = {'llm_winner': 'LLM_winner', 'ALLD': 'ALLD'}
N_VALUES = [1, 2, 3, 4, 5, 6, 7]


def load_traj(invader, ess, n, seed=42):
    p = base / f'{invader}_vs_{ess}_n{n}_seed{seed}' / 'invasion.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))['trajectory']


# ============================================================
# Plot 1: Trajectory grid (2x4 subplots, one per ESS)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
n_colors = plt.cm.viridis([i / (len(N_VALUES) - 1) for i in range(len(N_VALUES))])

for ax, ess in zip(axes.flat, LEADING_EIGHT):
    for invader_idx, invader in enumerate(INVADERS):
        for n_idx, n in enumerate(N_VALUES):
            traj = load_traj(invader, ess, n)
            if traj is None:
                continue
            gens = [t['generation'] for t in traj]
            freqs = [t['invader_freq'] for t in traj]
            ls = '-' if invader == 'llm_winner' else '--'
            ax.plot(gens, freqs, ls, color=n_colors[n_idx],
                    alpha=0.7 + 0.3 * invader_idx,
                    linewidth=1.0 + 0.5 * invader_idx,
                    label=f'{invader[:3]} n={n}' if ess == 'IS' else None)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
    ax.set_title(f'vs {ESS_DISPLAY.get(ess, ess)}', fontsize=11)
    ax.set_xlabel('Generation')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    if ess == 'IS':
        ax.set_ylabel('Invader frequency')

# Build a single legend (one entry per invader × n)
legend_handles = []
for n_idx, n in enumerate(N_VALUES):
    for invader_idx, invader in enumerate(INVADERS):
        ls = '-' if invader == 'LLM_winner' else '--'
        label = f'{invader.replace("_", " ")} n={n}'
        h, = plt.plot([], [], ls, color=n_colors[n_idx],
                      linewidth=1.0 + 0.5 * invader_idx)
        legend_handles.append((label, h))
# Sort by (n, invader) for clean ordering
legend_handles.sort(key=lambda x: (
    int(x[0].split('n=')[1]),
    0 if 'LLM' in x[0] else 1,
))
labels = [l for l, h in legend_handles]
handles = [h for l, h in legend_handles]
fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Invasibility test: ALLD vs LLM_winner across 8 leading-eight ESS\n'
             '(solid = LLM_winner, dashed = ALLD; 50 generations, seed=42)',
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
out1_png = base / 'invasion_compare_trajectories.png'
out1_pdf = base / 'invasion_compare_trajectories.pdf'
fig.savefig(out1_png, dpi=150, bbox_inches='tight')
fig.savefig(out1_pdf, bbox_inches='tight')
plt.close(fig)
print(f'saved: {out1_png} ({out1_png.stat().st_size/1024:.1f} KB)')


# ============================================================
# Plot 2: Threshold chart (final invader_freq vs n, averaged across ESS)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Compute per-n mean ± std across ESS, for each invader
for invader_idx, invader in enumerate(INVADERS):
    n_means = []
    n_stds = []
    n_labels = []
    for n in N_VALUES:
        finals = []
        for ess in LEADING_EIGHT:
            traj = load_traj(invader, ess, n)
            if traj is not None:
                finals.append(traj[-1]['invader_freq'])
        if finals:
            n_means.append(statistics.mean(finals))
            n_stds.append(statistics.stdev(finals) if len(finals) > 1 else 0)
            n_labels.append(n)

    color = '#1f77b4' if invader == 'llm_winner' else '#d62728'
    marker = 'o' if invader == 'llm_winner' else 's'
    label = DISPLAY[invader]
    ax.errorbar(n_labels, n_means, yerr=n_stds, marker=marker, markersize=8,
                linewidth=2, capsize=4, color=color, label=label, alpha=0.85)

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
ax.axvline(2.5, color='black', linestyle='--', alpha=0.5)
ax.text(2.55, 0.05, 'n* ∈ [2, 3]', fontsize=10, verticalalignment='bottom')
ax.set_xlabel('Initial invader count n (out of 15)', fontsize=11)
ax.set_ylabel('Final invader frequency (mean ± std across 8 ESS)', fontsize=11)
ax.set_title('Invasion threshold: ALLD vs LLM_winner across leading-eight ESS', fontsize=12)
ax.set_xticks(N_VALUES)
ax.set_ylim(-0.05, 1.10)
ax.grid(alpha=0.3)
ax.legend(loc='center right', fontsize=10)
ax.text(0.02, 0.98, 'Both invaders share the same threshold:\n'
        'n=1,2: die in all 8 ESS\n'
        'n=3..7: fixate in all 8 ESS',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.tight_layout()
out2_png = base / 'invasion_compare_threshold.png'
out2_pdf = base / 'invasion_compare_threshold.pdf'
fig.savefig(out2_png, dpi=150)
fig.savefig(out2_pdf)
plt.close(fig)
print(f'saved: {out2_png} ({out2_png.stat().st_size/1024:.1f} KB)')


# ============================================================
# Plot 3: Final frequency heatmap (2 invader × 8 ESS, for fixed n=1,3,7)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
n_show = [1, 3, 7]

for ax, n in zip(axes, n_show):
    matrix = np.zeros((len(INVADERS), len(LEADING_EIGHT)))
    for i, invader in enumerate(INVADERS):
        for j, ess in enumerate(LEADING_EIGHT):
            traj = load_traj(invader, ess, n)
            matrix[i, j] = traj[-1]['invader_freq'] if traj else np.nan
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(LEADING_EIGHT)))
    ax.set_xticklabels([ESS_DISPLAY.get(e, e) for e in LEADING_EIGHT])
    ax.set_yticks(range(len(INVADERS)))
    ax.set_yticklabels(['LLM_winner', 'ALLD'])
    ax.set_title(f'n={n} ({n}/15 = {n/15*100:.0f}%)', fontsize=11)
    for i in range(len(INVADERS)):
        for j in range(len(LEADING_EIGHT)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    color='black' if 0.2 < val < 0.8 else 'white',
                    fontsize=10, fontweight='bold')
    if n == 1:
        ax.set_ylabel('Invader type')
    ax.set_xlabel('Resident ESS')
fig.suptitle('Final invader frequency at gen 50 (red=fix, green=die, yellow=drift)',
             fontsize=12)
fig.tight_layout()
out3_png = base / 'invasion_compare_heatmap.png'
out3_pdf = base / 'invasion_compare_heatmap.pdf'
fig.savefig(out3_png, dpi=150)
fig.savefig(out3_pdf)
plt.close(fig)
print(f'saved: {out3_png} ({out3_png.stat().st_size/1024:.1f} KB)')

print('\n=== Summary ===')
print(f'Total runs: {len(LEADING_EIGHT) * len(N_VALUES) * len(INVADERS)} = '
      f'{len(LEADING_EIGHT)} ESS × {len(N_VALUES)} n values × {len(INVADERS)} invaders')
for invader in INVADERS:
    print(f'\n{invader}:')
    for n in N_VALUES:
        finals = []
        for ess in LEADING_EIGHT:
            traj = load_traj(invader, ess, n)
            if traj is not None:
                finals.append(traj[-1]['invader_freq'])
        if finals:
            m = statistics.mean(finals)
            verdict = 'all fixate' if m >= 0.9 else ('all die' if m <= 0.1 else 'mixed')
            print(f'  n={n}: mean_final={m:.3f} ({verdict})')
