"""Aggregate the 3-seed 100 gen x 1000 inter run + plot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
QB = ROOT / 'results' / 'quantitative_baseline'
PLOT_DIR = QB / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load 3 seeds of 1000 inter
seeds_1000 = []
for s in [0, 1, 2]:
    p = QB / f'LLM_v3_g100_1000inter_seed{s}' / 'evolutionary.json'
    if p.exists():
        d = json.loads(p.read_text())
        seeds_1000.append((s, d))
        print(f"seed {s}: gen0={d['trajectory'][0]['cooperation_rate_mean']:.3f}, "
              f"final={d['trajectory'][-1]['cooperation_rate_mean']:.3f}, "
              f"fitness_final={d['trajectory'][-1]['fitness_mean']:.1f}, "
              f"FALLBACK init={d['config']['fallback_init_count']}, "
              f"mut={d['config']['fallback_mutation_count']}")
    else:
        print(f"seed {s}: MISSING")

if not seeds_1000:
    print("No 1000-inter data found")
    raise SystemExit

# M4 baseline (210 inter, 1 seed)
m4_path = QB / 'LLM_v3_g100_thinking_off_seed0' / 'evolutionary.json'
m4 = None
if m4_path.exists():
    m4 = json.loads(m4_path.read_text())

# Final stats
final_coops_1000 = [d['trajectory'][-1]['cooperation_rate_mean'] for s, d in seeds_1000]
final_fits_1000 = [d['trajectory'][-1]['fitness_mean'] for s, d in seeds_1000]
print()
print(f"=== 1000 inter / 3 seeds / 100 gen ===")
print(f"  final coop: {final_coops_1000}")
print(f"  mean ± std: {np.mean(final_coops_1000):.3f} ± {np.std(final_coops_1000):.3f}")
print(f"  fitness:    {final_fits_1000}")
print(f"  mean ± std: {np.mean(final_fits_1000):.1f} ± {np.std(final_fits_1000):.1f}")

if m4:
    final_m4 = m4['trajectory'][-1]['cooperation_rate_mean']
    print(f"\n  M4 210 inter 1 seed: final coop = {final_m4:.3f}")
    print(f"  1000 inter 3 seeds:  final coop = {np.mean(final_coops_1000):.3f} ± {np.std(final_coops_1000):.3f}")
    print(f"  delta: 1000 inter spreads variance {np.std(final_coops_1000):.3f} "
          f"vs M4 single-seed deterministic-ish {0.0:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: 3 seeds overlaid
ax = axes[0]
cmap = plt.cm.coolwarm
for i, (s, d) in enumerate(seeds_1000):
    gens = [t['generation'] for t in d['trajectory']]
    coop = [t['cooperation_rate_mean'] for t in d['trajectory']]
    fit = [t['fitness_mean'] for t in d['trajectory']]
    ax.plot(gens, coop, color=cmap(0.0 + 0.4 * i), linewidth=1.0, alpha=0.7,
            label=f'seed {s} (final={coop[-1]:.3f})')
# Mean
min_len = min(len(d['trajectory']) for s, d in seeds_1000)
gens_min = [t['generation'] for t in seeds_1000[0][1]['trajectory'][:min_len]]
coop_arr = np.array([[t['cooperation_rate_mean'] for t in d['trajectory'][:min_len]]
                     for s, d in seeds_1000])
mean_coop = coop_arr.mean(axis=0)
std_coop = coop_arr.std(axis=0)
ax.plot(gens_min, mean_coop, color='black', linewidth=2.0, label='mean (n=3)')
ax.fill_between(gens_min, mean_coop - std_coop, mean_coop + std_coop,
                color='gray', alpha=0.25, label='± std')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)
ax.set_title('100 gen × 1000 inter × 3 seeds (type-2 LLM)', fontsize=12)
ax.set_xlabel('Generation')
ax.set_ylabel('Cooperation rate')
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

# Right: M4 210 inter vs 1000 inter mean
ax = axes[1]
if m4:
    m4_gens = [t['generation'] for t in m4['trajectory']]
    m4_coop = [t['cooperation_rate_mean'] for t in m4['trajectory']]
    ax.plot(m4_gens, m4_coop, color='#1f77b4', linewidth=1.5, alpha=0.7,
            label=f'210 inter seed 0 (final={m4_coop[-1]:.3f})')
ax.plot(gens_min, mean_coop, color='#d62728', linewidth=1.5,
        label=f'1000 inter mean (final={mean_coop[-1]:.3f}, std={std_coop[-1]:.3f})')
if len(seeds_1000) > 1:
    ax.fill_between(gens_min, mean_coop - std_coop, mean_coop + std_coop,
                    color='#d62728', alpha=0.2)
# Annotate basin
ax.text(50, 0.05, 'seed 0: defection basin', color='red', fontsize=9)
ax.text(85, 0.85, 'seed 1/2: mid-coop basin', color='red', fontsize=9)
ax.set_title('210 inter vs 1000 inter', fontsize=12)
ax.set_xlabel('Generation')
ax.set_ylabel('Cooperation rate')
ax.set_ylim(-0.02, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

fig.suptitle('v3 type-2 LLM evolution: ESS multi-basin dynamics', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out_path = PLOT_DIR / 'g100_3seed_1000inter.png'
fig.savefig(out_path, dpi=150)
out_pdf = PLOT_DIR / 'g100_3seed_1000inter.pdf'
fig.savefig(out_pdf)
plt.close(fig)
print(f"\nsaved: {out_path}")
print(f"saved: {out_pdf}")
