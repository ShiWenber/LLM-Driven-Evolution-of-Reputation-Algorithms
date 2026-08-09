"""Plot exp1 (orig) seed 0 vs exp6 (leading-8 interface) seed 0 trajectory."""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

d_orig = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp1_method_n10/full_seed0'
d_new = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'

with open(d_orig + '/evo_full_deepseek-v4-flash_20260610_052333.json') as f:
    orig = json.load(f)
with open(d_new + '/evo_full_deepseek-v4-flash_20260616_105442.json') as f:
    new = json.load(f)

orig_coops = [g['cooperation_rate_mean'] for g in orig['trajectory']]
new_coops = [g['cooperation_rate_mean'] for g in new['trajectory']]
gens = list(range(len(orig_coops)))

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(gens, orig_coops, 'o-', color='#2E86AB', linewidth=2.5, markersize=8,
        label='Original interface\n(observation[\"action\"] only)')
ax.plot(gens, new_coops, 's--', color='#E63946', linewidth=2.5, markersize=8,
        label='Augmented interface\n(+ donor_reputation, recipient_reputation)')

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Initial rep cold-start (0.01)')
ax.set_xlabel('Generation', fontsize=12)
ax.set_ylabel('Mean cooperation rate (population)', fontsize=12)
ax.set_title('Effect of augmenting the observation interface on LLM-evolved cooperation\n'
             '(full observability, 1 seed, N=15, G=10, deepseek-v4-flash)', fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(gens)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)

# Annotate final values
ax.annotate(f'gen-10 = {orig_coops[-1]:.2f}', xy=(gens[-1], orig_coops[-1]),
            xytext=(gens[-1]-3, orig_coops[-1]+0.08), fontsize=10, color='#2E86AB',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', alpha=0.6))
ax.annotate(f'gen-10 = {new_coops[-1]:.2f}', xy=(gens[-1], new_coops[-1]),
            xytext=(gens[-1]-3, new_coops[-1]+0.12), fontsize=10, color='#E63946',
            arrowprops=dict(arrowstyle='->', color='#E63946', alpha=0.6))

plt.tight_layout()

out_pdf = d_new + '/comparison_trajectory.pdf'
out_png = d_new + '/comparison_trajectory.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')
print(f'Saved {out_png}')

# Also: small plot of how many strategies in final pop use each new field
fp = new['final_population']
use_donor = sum(1 for a in fp if 'donor_reputation' in a['code'].split('def decide')[0])
use_recipient = sum(1 for a in fp if 'recipient_reputation' in a['code'].split('def decide')[0])
use_either = sum(1 for a in fp if 'donor_reputation' in a['code'] or 'recipient_reputation' in a['code'])
print(f'\nFinal pop (15 agents) interface usage:')
print(f'  use donor_reputation:    {use_donor}/15 ({100*use_donor/15:.1f}%)')
print(f'  use recipient_reputation: {use_recipient}/15 ({100*use_recipient/15:.1f}%)')
print(f'  use either:              {use_either}/15 ({100*use_either/15:.1f}%)')
