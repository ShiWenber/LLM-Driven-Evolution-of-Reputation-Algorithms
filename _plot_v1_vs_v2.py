"""Plot v1 vs v2 trajectory comparison (3-seed mean ± std)."""
import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

base = Path(r'results/quantitative_baseline')

def load_traj(s, label):
    p = base / f'{label}_seed{s}' / 'evolutionary.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    return [(t['generation'], t['cooperation_rate_mean']) for t in d['trajectory']]

v1, v2 = {}, {}
for s in [0, 1, 2]:
    v1[s] = load_traj(s, 'LLM_v3_fermi_z_g100_1000inter')
    v2[s] = load_traj(s, 'LLM_v3_fermi_z_v2_g100_1000inter')

gens_v1 = [g for g, _ in v1[0]]
gens_v2 = [g for g, _ in v2[0]]

def mean_std(traj_dict, gens):
    arr = []
    for g in gens:
        vals = [v for (gg, v) in traj_dict.items() if gg == g or True]
        # align by gen
    means, stds = [], []
    for g in gens:
        vals = [dict(traj_dict[s])[g] for s in traj_dict]
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return means, stds

v1_dict = {s: dict(v1[s]) for s in v1}
v2_dict = {s: dict(v2[s]) for s in v2}

m1, s1 = mean_std(v1_dict, gens_v1)
m2, s2 = mean_std(v2_dict, gens_v2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(gens_v1, m1, '-o', color='#2c7fb8', label='v1: over-constrained prompt (M8)', markersize=4)
ax.fill_between(gens_v1, [a - b for a, b in zip(m1, s1)], [a + b for a, b in zip(m1, s1)], color='#2c7fb8', alpha=0.2)
ax.plot(gens_v2, m2, '-s', color='#d7301f', label='v2: loosened prompt (M9)', markersize=4)
ax.fill_between(gens_v2, [a - b for a, b in zip(m2, s2)], [a + b for a, b in zip(m2, s2)], color='#d7301f', alpha=0.2)
ax.set_xlabel('Generation')
ax.set_ylabel('Cooperation rate (3-seed mean)')
ax.set_title('Fermi Z-like: prompt looseness reveals true evolutionary dynamics\n'
             'v1: mean 0.892±0.098   v2: mean 0.360±0.492   (M9)')
ax.legend(loc='best', frameon=True)
ax.set_ylim(-0.02, 1.05)
ax.grid(alpha=0.3)
fig.tight_layout()

out_png = base / 'fermi_z_v1_vs_v2.png'
out_pdf = base / 'fermi_z_v1_vs_v2.pdf'
fig.savefig(out_png, dpi=150)
fig.savefig(out_pdf)
print(f'saved: {out_png} ({out_png.stat().st_size/1024:.1f} KB)')
print(f'saved: {out_pdf}')
