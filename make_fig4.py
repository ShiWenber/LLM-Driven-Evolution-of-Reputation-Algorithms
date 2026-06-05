"""Generate Figure 4: side-by-side trajectory comparison (with vs without selection).

We compare LLM-driven evolution (with selection) vs static (no selection) at
the same observability level. The static trajectories are flat (no drift)
while the evolutionary trajectories show selection pressure killing
cooperation over generations.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation')
RES = ROOT / 'results'
OUT = RES / 'figures'

# Sort key
def sort_key(n):
    if n == 'private': return 0
    if n == 'full': return 1
    if n.startswith('partial_'):
        try: return float(n.split('_')[1])
        except: return 99
    return 99

OBS_LABELS = {
    'private': 'private (p=0)',
    'partial_0.3': 'partial (p=0.3)',
    'full': 'full (p=1.0)',
}
OBS_COLORS_EVO = {
    'private': '#1f77b4',
    'partial_0.3': '#ff7f0e',
    'full': '#2ca02c',
}
OBS_COLORS_STATIC = {
    'private': '#aec7e8',
    'partial_0.3': '#ffbb78',
    'full': '#98df8a',
}

# Load LLM-evolutionary trajectories (Experiment 1: 3 seeds, G=10, T=30)
evo = defaultdict(list)
for sd in (RES / 'exp1_method').iterdir():
    if not sd.is_dir(): continue
    for f in sd.glob('evo_*.json'):
        d = json.loads(f.read_text(encoding='utf-8'))
        traj = d.get('trajectory', [])
        if not traj: continue
        # obs from path
        obs = sd.name.rsplit('_seed', 1)[0]
        evo[obs].append(traj)

# Load static trajectories (Experiment 3: 2 seeds, G=5, T=30)
static = defaultdict(list)
for sd in (RES / 'exp3_static').iterdir():
    if not sd.is_dir(): continue
    for f in sd.glob('static_control_*.json'):
        d = json.loads(f.read_text(encoding='utf-8'))
        # The aggregate file has trials_summary, but per-trial files have trajectory directly
        if 'trajectory' in d:
            traj = d['trajectory']
        elif 'trials_summary' in d:
            # take the first trial
            traj = d['trials_summary'][0].get('trajectory', [])
        else:
            continue
        if not traj: continue
        obs = sd.name.rsplit('_seed', 1)[0]
        static[obs].append(traj)

print('Evo obs:', sorted(evo.keys(), key=sort_key))
print('Static obs:', sorted(static.keys(), key=sort_key))
for obs in ['private', 'partial_0.3', 'full']:
    print(f'  {obs}: evo n={len(evo.get(obs,[]))}, static n={len(static.get(obs,[]))}')

# Compute mean trajectories
def mean_traj(trajs):
    if not trajs: return None
    n_gens = min(len(t) for t in trajs)
    return np.array([np.mean([t[g]['cooperation_rate_mean'] for t in trajs]) for g in range(n_gens)])

# Build figure: 3 panels, one per obs
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, obs in zip(axes, ['private', 'partial_0.3', 'full']):
    evo_t = mean_traj(evo.get(obs, []))
    sta_t = mean_traj(static.get(obs, []))
    if evo_t is not None:
        ax.plot(np.arange(len(evo_t)), evo_t, marker='o', markersize=4,
                color=OBS_COLORS_EVO[obs], linewidth=2,
                label=f'LLM-evo (G=10, n={len(evo[obs])} seeds)')
    if sta_t is not None:
        ax.plot(np.arange(len(sta_t)), sta_t, marker='s', markersize=4,
                color=OBS_COLORS_STATIC[obs], linewidth=2,
                label=f'Static (no selection, G=5, n={len(static[obs])} seeds)')
    ax.set_title(f'{OBS_LABELS[obs]}')
    ax.set_xlabel('Generation')
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

axes[0].set_ylabel('Mean cooperation rate')
fig.suptitle('Selection pressure comparison: with vs without LLM-driven evolution\n'
             '(Static trajectories are flat — no selection, no drift; '
             'LLM-evo trajectories show selection pressure gradually eliminating cooperation)',
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = OUT / 'fig4_selection_comparison.png'
fig.savefig(out, dpi=120)
plt.close(fig)
print(f'Figure 4: {out}')

# Also export PDF version
from PIL import Image
img = Image.open(out).convert('RGB')
out_pdf = out.with_suffix('.pdf')
img.save(out_pdf, 'PDF', resolution=150)
print(f'Figure 4 PDF: {out_pdf}')
