"""Plot comparison: 3 exp6 trials with different action labels.
1. 'donate'/'not_donate' (original)
2. 'option_0'/'option_1' (previous attempt)
3. 'A'/'B' (current)
"""
import json
import os
import matplotlib.pyplot as plt

# Find all 3 trial files
base = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'
files = [f for f in os.listdir(base) if f.startswith('evo_full_deepseek-v4-flash_') and f.endswith('.json')]
files.sort()
print('files:', files)

# Find the corresponding run in exp1_method_n10 (original interface)
orig_d = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp1_method_n10/full_seed0'

datasets = []
with open(os.path.join(orig_d, 'evo_full_deepseek-v4-flash_20260610_052333.json')) as f:
    t_orig = json.load(f)
coops_orig = [g['cooperation_rate_mean'] for g in t_orig['trajectory']]
datasets.append(('original (donate/not_donate)\nseed 0, full obs', coops_orig, '#2E86AB', '-'))

# New exp6 trials
for fp in files:
    with open(os.path.join(base, fp)) as f:
        t = json.load(f)
    coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
    # Parse timestamp
    ts = fp.split('_')[-1].replace('.json', '')
    # Heuristic: 20260616_115153 = A/B
    if '20260616_115153' in fp:
        label = '"A"/"B" (neutral)\nseed 0, full obs, G=10'
        color = '#06A77D'
        style = '-'
    elif '20260616_105442' in fp:
        label = '"option_0"/"option_1" (neutral)\nseed 0, full obs, G=10'
        color = '#E63946'
        style = '--'
    else:
        continue
    datasets.append((label, coops, color, style))

fig, ax = plt.subplots(figsize=(10, 6))
for label, coops, color, style in datasets:
    gens = list(range(len(coops)))
    ax.plot(gens, coops, 'o' + style, color=color, linewidth=2.5, markersize=8, label=label)
    # Annotate final value
    ax.annotate(f'{coops[-1]:.2f}', xy=(gens[-1], coops[-1]),
                xytext=(gens[-1]+0.3, coops[-1]+0.05), fontsize=11,
                color=color, fontweight='bold')

ax.set_xlabel('Generation', fontsize=12)
ax.set_ylabel('Mean cooperation rate (population)', fontsize=12)
ax.set_title('Effect of action-label naming on LLM-evolved cooperation\n'
             '(full observability, seed 0, N=15, G=10, deepseek-v4-flash)',
             fontsize=12)
ax.set_ylim(-0.05, 1.15)
ax.set_xticks(list(range(10)))
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)

plt.tight_layout()
out_pdf = base + '/action_label_comparison.pdf'
out_png = base + '/action_label_comparison.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')
print(f'Saved {out_png}')
