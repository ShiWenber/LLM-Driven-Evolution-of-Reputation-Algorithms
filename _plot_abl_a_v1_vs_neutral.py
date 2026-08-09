"""Plot: Ablation A v1 (adversarial) vs NEUTRAL main run, 3 seeds each, 100 gen."""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QB = Path(r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline')
PLOT_DIR = QB / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def load_traj(name_prefix, n_seeds=3):
    out = []
    for s in range(n_seeds):
        p = QB / f'{name_prefix}_seed{s}' / 'evolutionary.json'
        d = json.loads(p.read_text())
        out.append(d['trajectory'])
    return out

v1_traj = load_traj('LLM_v3_g100_1000inter_ADVERSARIAL')
n_traj  = load_traj('LLM_v3_g100_1000inter')

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

def panel(ax, trajs, label, color):
    min_len = min(len(t) for t in trajs)
    gens = [t['generation'] for t in trajs[0][:min_len]]
    coop = np.array([[t['cooperation_rate_mean'] for t in tr[:min_len]] for tr in trajs])
    mean = coop.mean(0); std = coop.std(0)
    for i, tr in enumerate(trajs):
        ax.plot([t['generation'] for t in tr[:min_len]],
                [t['cooperation_rate_mean'] for t in tr[:min_len]],
                color=color, alpha=0.25, linewidth=1.0)
    ax.plot(gens, mean, color=color, linewidth=2.5, label=f'{label} mean')
    ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2, label=f'{label} ± std')
    # final annotation
    finals = [tr[-1]['cooperation_rate_mean'] for tr in trajs]
    ax.text(0.98, 0.05 if label == 'v1 ADVERSARIAL' else 0.95,
            f'{label}\nfinal: {finals[0]:.3f}, {finals[1]:.3f}, {finals[2]:.3f}\nmean={np.mean(finals):.3f}±{np.std(finals):.3f}',
            transform=ax.transAxes, ha='right', va='bottom' if label == 'v1 ADVERSARIAL' else 'top',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor=color))

# left: v1 ADVERSARIAL
panel(axes[0], v1_traj, 'v1 ADVERSARIAL', 'tab:red')
axes[0].set_title('Ablation A v1: adversarial prompt (PD + canonical hints)\n3 seeds, 100 gen × 1000 inter, LLM type-2')
axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Cooperation rate')
axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.4)
axes[0].set_ylim(-0.05, 1.05); axes[0].legend(loc='upper left', fontsize=9); axes[0].grid(alpha=0.3)

# right: NEUTRAL
panel(axes[1], n_traj, 'NEUTRAL', 'tab:blue')
axes[1].set_title('Main run: neutral prompt (no PD framing)\n3 seeds, 100 gen × 1000 inter, LLM type-2')
axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Cooperation rate')
axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.4)
axes[1].set_ylim(-0.05, 1.05); axes[1].legend(loc='upper left', fontsize=9); axes[1].grid(alpha=0.3)

fig.suptitle('Ablation A v1: prompt-induced basin attraction (low start → high end)\n'
             'v1 ADVERSARIAL gen0≈0.27, final≈0.83±0.21  |  NEUTRAL gen0≈0.85, final≈0.44±0.35',
             fontsize=12, y=1.02)
plt.tight_layout()
out_png = PLOT_DIR / 'abl_a_v1_vs_neutral.png'
out_pdf = PLOT_DIR / 'abl_a_v1_vs_neutral.pdf'
fig.savefig(out_png, dpi=150, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f'saved: {out_png}')
print(f'saved: {out_pdf}')
