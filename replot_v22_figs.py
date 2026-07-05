"""Update fig3 (control comparison) and fig4 (trajectory comparison) with v22 data.
Reads static trials from results/exp3_static_g10_n10, LLM-evo from
results/exp1_method_n10, random mutation from results/exp4_random_mut.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

def collect_final_coop(base, obs, n_seeds=10):
    """Pull final-gen cooperation for all (obs, seed) dirs."""
    finals = []
    for seed in range(n_seeds):
        d = base / f'{obs}_seed{seed}'
        if not d.exists(): continue
        # static_control_* or evo_* or evolutionary_*
        jsons = [f for f in d.iterdir()
                 if (f.name.startswith('evo_') or f.name.startswith('static_control_') or f.name.startswith('evolutionary_'))
                 and f.name.endswith('.json')]
        if not jsons: continue
        # Pick evo_*.json if exists (full per-trial), else static_control_*.json
        evos = [j for j in jsons if j.name.startswith('evo_')]
        if evos:
            jsons = evos
        jsons.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        try:
            with open(jsons[0]) as f:
                t = json.load(f)
            if 'trials_summary' in t and t['trials_summary']:
                traj = t['trials_summary'][0].get('trajectory', [])
                coop = traj[-1]['cooperation_rate_mean'] if traj else None
            else:
                traj = t.get('trajectory', [])
                coop = traj[-1]['cooperation_rate_mean'] if traj else None
            if coop is not None:
                finals.append(coop)
        except Exception:
            pass
    return finals

def collect_trajectories(base, obs, n_seeds=10):
    """Pull full trajectories per seed."""
    all_traj = []
    for seed in range(n_seeds):
        d = base / f'{obs}_seed{seed}'
        if not d.exists(): continue
        jsons = [f for f in d.iterdir()
                 if (f.name.startswith('evo_') or f.name.startswith('static_control_'))
                 and f.name.endswith('.json')]
        if not jsons: continue
        evos = [j for j in jsons if j.name.startswith('evo_')]
        if evos:
            jsons = evos
        jsons.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        try:
            with open(jsons[0]) as f:
                t = json.load(f)
            if 'trials_summary' in t and t['trials_summary']:
                traj = t['trials_summary'][0].get('trajectory', [])
            else:
                traj = t.get('trajectory', [])
            if len(traj) >= 2:
                all_traj.append([g['cooperation_rate_mean'] for g in traj])
        except Exception:
            pass
    return all_traj

OBS = ['private', 'partial_0.3', 'partial_0.7', 'full']
STATIC_BASE = REPO / 'results' / 'exp3_static_g10_n10'
EVOL_BASE = REPO / 'results' / 'exp1_method_n10'
RAND_BASE = REPO / 'results' / 'exp4_random_mut'

# === fig3: control comparison (static vs LLM-evo vs random mutation) ===
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(OBS))
w = 0.27

# LLM-evo
for i, obs in enumerate(OBS):
    finals = collect_final_coop(EVOL_BASE, obs, n_seeds=10)
    print(f'LLM-evo {obs}: n={len(finals)} mean={np.mean(finals):.3f}')

statics = [np.mean(collect_final_coop(STATIC_BASE, obs, n_seeds=10)) for obs in OBS]
evol = [np.mean(collect_final_coop(EVOL_BASE, obs, n_seeds=10)) for obs in OBS]
rands = [np.mean(collect_final_coop(RAND_BASE, obs, n_seeds=5)) for obs in OBS if obs != 'partial_0.7']  # no random p0.7
rands = [rands[0], rands[1], 0, rands[2]]  # insert 0 for partial_0.7
# LLM-evo with std
evol_stds = [np.std(collect_final_coop(EVOL_BASE, obs, n_seeds=10)) for obs in OBS]
statics_stds = [np.std(collect_final_coop(STATIC_BASE, obs, n_seeds=10)) for obs in OBS]

ax.bar(x - w, evol, w, yerr=evol_stds, capsize=4, label='LLM-driven evolution (n=10)', color='#264653', alpha=0.85)
ax.bar(x, statics, w, yerr=statics_stds, capsize=4, label='Static control (n=10)', color='#E76F51', alpha=0.85)
ax.bar(x + w, rands, w, label='Random mutation (n=5)', color='#2A9D8F', alpha=0.85)

for i, (e, s, r) in enumerate(zip(evol, statics, rands)):
    ax.text(i - w, e + 0.03, f'{e:.2f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.text(i, s + 0.03, f'{s:.2f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.text(i + w, r + 0.03, f'{r:.2f}', ha='center', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['private ($p$=0)', 'partial 0.3', 'partial 0.7', 'full ($p$=1)'], fontsize=10)
ax.set_ylabel('Final-generation mean cooperation rate', fontsize=11)
ax.set_title('Control comparison (v22 data): LLM-driven vs. static vs. random mutation', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 0.85)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out_pdf = REPO / 'results' / 'figures' / 'fig3_control_comparison.pdf'
out_png = REPO / 'results' / 'figures' / 'fig3_control_comparison.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')

# === fig4: trajectory comparison ===
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
labels = ['private ($p$=0)', 'partial 0.3', 'partial 0.7', 'full ($p$=1)']
for i, obs in enumerate(OBS):
    ax = axes[i]
    # Static
    static_traj = collect_trajectories(STATIC_BASE, obs, n_seeds=10)
    for tr in static_traj:
        ax.plot(range(len(tr)), tr, color='#E76F51', alpha=0.3, linewidth=1)
    if static_traj:
        L = min(len(t) for t in static_traj)
        mean_static = np.mean([t[:L] for t in static_traj], axis=0)
        ax.plot(range(L), mean_static, color='#E76F51', linewidth=2.5, label='static mean (n=10)')
    # LLM-evo
    evol_traj = collect_trajectories(EVOL_BASE, obs, n_seeds=10)
    for tr in evol_traj:
        ax.plot(range(len(tr)), tr, color='#264653', alpha=0.3, linewidth=1)
    if evol_traj:
        L = min(len(t) for t in evol_traj)
        mean_evol = np.mean([t[:L] for t in evol_traj], axis=0)
        ax.plot(range(L), mean_evol, color='#264653', linewidth=2.5, label='LLM-evo mean (n=10)')
    ax.set_title(labels[i], fontsize=11)
    ax.set_xlabel('generation', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    if i == 0: ax.set_ylabel('mean cooperation rate', fontsize=10)
axes[0].legend(fontsize=9, loc='lower right')
fig.suptitle('Trajectory comparison (v22 data): LLM-driven evolution vs. static control', fontsize=12)
plt.tight_layout()
out_pdf = REPO / 'results' / 'figures' / 'fig4_selection_comparison.pdf'
out_png = REPO / 'results' / 'figures' / 'fig4_selection_comparison.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')
print('done')