"""Plot the A/B trial trajectory with full annotations."""
import json
import matplotlib.pyplot as plt

d = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'
with open(d + '/evo_full_deepseek-v4-flash_20260616_115153.json') as f:
    t = json.load(f)

coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
gens = list(range(len(coops)))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(gens, coops, 'o-', color='#06A77D', linewidth=3.0, markersize=10,
        markerfacecolor='#06A77D', markeredgecolor='white', markeredgewidth=2,
        label='"A" / "B" (neutral action label)')

# Annotate each point
for i, c in enumerate(coops):
    if c >= 0.5:
        ax.annotate(f'{c:.2f}', xy=(i, c), xytext=(i, c+0.05),
                    fontsize=10, ha='center', color='#06A77D', fontweight='bold')
    else:
        ax.annotate(f'{c:.2f}', xy=(i, c), xytext=(i, c-0.08),
                    fontsize=10, ha='center', color='#666')

# Annotate final value with arrow
ax.annotate(f'gen-10 = {coops[-1]:.2f}\n(15/15 Hybrid strategies)',
            xy=(gens[-1], coops[-1]),
            xytext=(gens[-1]-2.5, coops[-1]-0.25),
            fontsize=11, color='#06A77D', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#06A77D', lw=2, alpha=0.7))

# Highlight the recovery
ax.axvspan(2.5, 4.5, alpha=0.15, color='#06A77D',
           label='selection finds cooperative attractor (gen 3→4)')
ax.text(3.5, 0.05, 'selection finds\ncooperative attractor',
        ha='center', fontsize=9, color='#06A77D', style='italic')

ax.set_xlabel('Generation', fontsize=12)
ax.set_ylabel('Mean cooperation rate (population)', fontsize=12)
ax.set_title('LLM-evolved cooperation with neutral action label\n'
             '("A" / "B" instead of "donate" / "not_donate")\n'
             '(full obs, 1 seed, N=15, G=10, deepseek-v4-flash, 19 min)',
             fontsize=12)
ax.set_ylim(-0.05, 1.15)
ax.set_xticks(gens)
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)

plt.tight_layout()
out_pdf = d + '/AB_trial_annotated.pdf'
out_png = d + '/AB_trial_annotated.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')
print(f'Saved {out_png}')
