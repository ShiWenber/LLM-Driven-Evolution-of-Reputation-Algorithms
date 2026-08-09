"""Plot the top-3 final-cooperation trajectories across all exp7 trials."""
import json, os, glob
import matplotlib.pyplot as plt
import numpy as np

# Collect all trials, rank by final coop
base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling'
also = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5'

all_trials = []
for src_base, src_label in [(base, 'exp7'), (also, 'CD_n5')]:
    for subdir in os.listdir(src_base):
        d = os.path.join(src_base, subdir)
        if not os.path.isdir(d): continue
        for obs_dir in os.listdir(d):
            d2 = os.path.join(d, obs_dir)
            if not os.path.isdir(d2): continue
            files = [f for f in os.listdir(d2) if f.startswith('evo_') and f.endswith('.json')]
            if not files: continue
            files.sort(key=lambda f: os.path.getmtime(os.path.join(d2, f)), reverse=True)
            with open(os.path.join(d2, files[0])) as f:
                t = json.load(f)
            if not t.get('trajectory'): continue
            coop = t['trajectory'][-1]['cooperation_rate_mean']
            all_trials.append({
                'src': src_label, 'subdir': subdir, 'obs_dir': obs_dir,
                'coop': coop, 'traj': t['trajectory'],
                'gens': len(t['trajectory']),
            })

# Sort by final coop
all_trials.sort(key=lambda x: -x['coop'])

# Top 3
top3 = all_trials[:3]
print('=== TOP 3 trials by final cooperation ===')
for i, t in enumerate(top3):
    print(f'  #{i+1}: coop={t["coop"]:.3f} | {t["src"]}/{t["subdir"]}/{t["obs_dir"]} | {t["gens"]} gens')

# Plot top 3 as side-by-side
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for i, (ax, t) in enumerate(zip(axes, top3)):
    coops = [g['cooperation_rate_mean'] for g in t['traj']]
    gens = list(range(len(coops)))
    ax.plot(gens, coops, 'o-', color='#E63946', linewidth=2, markersize=5)
    ax.fill_between(gens, 0, coops, alpha=0.15, color='#E63946')
    ax.set_title(f'#{i+1}: {t["src"]}/{t["subdir"]}\nobs={t["obs_dir"]} | final coop={t["coop"]:.3f}',
                 fontsize=10)
    ax.set_xlabel('generation', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.set_ylabel('mean cooperation rate', fontsize=10)
fig.suptitle('Top 3 final-cooperation trajectories across all exp7 + CD_n5 trials',
             fontsize=12)
plt.tight_layout()
out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\top3_trajectories.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\top3_trajectories.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')

# Also: top 1 alone, larger
fig, ax = plt.subplots(figsize=(8, 5))
t = top3[0]
coops = [g['cooperation_rate_mean'] for g in t['traj']]
gens = list(range(len(coops)))
ax.plot(gens, coops, 'o-', color='#E63946', linewidth=2.5, markersize=7)
ax.fill_between(gens, 0, coops, alpha=0.2, color='#E63946')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold (high-coop attractor)')
ax.set_title(f'Highest-cooperation trajectory\n{t["src"]}/{t["subdir"]}/{t["obs_dir"]} | final coop = {t["coop"]:.3f}',
             fontsize=12)
ax.set_xlabel('generation', fontsize=11)
ax.set_ylabel('mean cooperation rate', fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\top1_trajectory.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling\top1_trajectory.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')