"""Comparison plot: 100 gen x 1 seed x 210 inter (M4 baseline) vs
100 gen x 3 seeds x 1000 inter (production).

The 1000-inter production run is still in progress when this is
first run; the script handles missing files gracefully and plots
whatever's available. Re-run after the 3 seeds finish to update.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
QB = ROOT / 'results' / 'quantitative_baseline'
PLOT_DIR = QB / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_traj(json_path):
    """Return (gens, coop, fitness) numpy arrays."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    gens = [t['generation'] for t in data['trajectory']]
    coop = [t['cooperation_rate_mean'] for t in data['trajectory']]
    fit = [t['fitness_mean'] for t in data['trajectory']]
    return np.array(gens), np.array(coop), np.array(fit)


# 100 gen x 1 seed x 210 inter (M4 baseline, 30 rounds)
m4_path = QB / 'LLM_v3_g100_thinking_off_seed0' / 'evolutionary.json'
m4 = load_traj(m4_path)
print(f"M4 (210 inter): {'OK' if m4 else 'MISSING'}")

# 100 gen x 3 seeds x 1000 inter (production, in progress)
prod_seeds = []
for s in [0, 1, 2]:
    p = QB / f'LLM_v3_g100_1000inter_seed{s}' / 'evolutionary.json'
    t = load_traj(p)
    if t is not None:
        prod_seeds.append((s, t))
        print(f"  prod seed {s} (1000 inter): OK")
    else:
        print(f"  prod seed {s} (1000 inter): not yet")

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

# Left panel: 210 inter (M4)
ax = axes[0]
if m4:
    gens, coop, fit = m4
    ax.plot(gens, coop, color='#1f77b4', linewidth=1.5, label='100 gen × 210 inter (seed 0)')
    # 4-phase annotations
    ax.axvline(15, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(30, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(60, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(95, color='gray', linestyle=':', alpha=0.4)
    ax.text(7, 0.92, 'high coop', fontsize=8, ha='center', color='gray')
    ax.text(22, 0.92, 'collapse', fontsize=8, ha='center', color='gray')
    ax.text(45, 0.92, 'defect basin', fontsize=8, ha='center', color='gray')
    ax.text(80, 0.92, 'rebound', fontsize=8, ha='center', color='gray')
ax.set_title('100 gen × 210 inter (baseline)', fontsize=12)
ax.set_xlabel('Generation')
ax.set_ylabel('Cooperation rate')
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

# Right panel: 1000 inter (production, possibly partial)
ax = axes[1]
if prod_seeds:
    # Stack seeds by length (assume same len, but trim to min)
    min_len = min(len(t[1][0]) for t in prod_seeds)
    coop_arr = np.array([t[1][1][:min_len] for t in prod_seeds])
    gens_arr = np.array([t[1][0][:min_len] for t in prod_seeds])
    fit_arr = np.array([t[1][2][:min_len] for t in prod_seeds])
    mean_coop = coop_arr.mean(axis=0)
    std_coop = coop_arr.std(axis=0)
    # Use first seed's gens as x-axis
    xs = gens_arr[0]
    ax.plot(xs, mean_coop, color='#d62728', linewidth=1.5,
            label=f'mean (n={len(prod_seeds)} seeds)')
    if len(prod_seeds) > 1:
        ax.fill_between(xs, mean_coop - std_coop, mean_coop + std_coop,
                        color='#d62728', alpha=0.2, label='± std')
    # Also plot individual seeds thin
    cmap = plt.cm.Reds
    for i, (s, (g, c, f)) in enumerate(prod_seeds):
        ax.plot(g, c, color=cmap(0.4 + 0.2 * i), linewidth=0.7, alpha=0.5,
                label=f'seed {s}')
    # Reference line at M4 final (0.743)
    ax.axhline(0.743, color='#1f77b4', linestyle='--', alpha=0.5,
               label='M4 210 inter final (0.743)')
ax.set_title('100 gen × 1000 inter (production)', fontsize=12)
ax.set_xlabel('Generation')
ax.set_ylabel('Cooperation rate')
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=8)

fig.suptitle('v3 type-2 LLM evolution: interactions per gen effect', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out_path = PLOT_DIR / 'g100_1000inter_vs_210inter.png'
fig.savefig(out_path, dpi=150)
out_pdf = PLOT_DIR / 'g100_1000inter_vs_210inter.pdf'
fig.savefig(out_pdf)
plt.close(fig)
print(f"\nsaved: {out_path}")
print(f"saved: {out_pdf}")

# Print quick stats
print(f"\n=== Quick stats ===")
if m4:
    print(f"M4 210 inter:  gen 0 coop = {m4[1][0]:.3f}, final coop = {m4[1][-1]:.3f}")
if prod_seeds:
    final_coops = [t[1][1][-1] for t in prod_seeds]
    final_fits = [t[1][2][-1] for t in prod_seeds]
    gen0_coops = [t[1][1][0] for t in prod_seeds]
    print(f"Prod 1000 inter (n={len(prod_seeds)} seeds so far):")
    print(f"  gen 0 coop: {[f'{c:.3f}' for c in gen0_coops]}")
    print(f"  final coop: {[f'{c:.3f}' for c in final_coops]}")
    if len(prod_seeds) > 1:
        print(f"  final coop mean ± std: {np.mean(final_coops):.3f} ± {np.std(final_coops):.3f}")
        print(f"  final fitness mean ± std: {np.mean(final_fits):.1f} ± {np.std(final_fits):.1f}")
