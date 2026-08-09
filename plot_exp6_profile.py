"""Plot the final-pop strategy profile: how many agents use each new field,
and what kind of recipient-conditional logic they implement."""
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

d = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'
with open(d + '/evo_full_deepseek-v4-flash_20260616_105442.json') as f:
    t = json.load(f)
fp = t['final_population']

# Categorize each strategy
def categorize(code):
    eval_part = code.split('def decide')[0] if 'def decide' in code else code
    has_donor_rep = 'donor_reputation' in eval_part
    has_recipient_rep = 'recipient_reputation' in eval_part
    has_action = 'observation["action"]' in eval_part or "observation['action']" in eval_part

    if not has_action and not has_donor_rep and not has_recipient_rep:
        return 'no learning\n(returns constant)'
    if has_recipient_rep and has_action:
        # Look for recipient-conditional structure
        # Heuristic: has explicit if-else on recipient_rep
        has_conditional = re.search(r'if\s+.*recipient_reputation', eval_part) is not None
        has_donate = '"donate"' in eval_part
        has_notdonate = '"not_donate"' in eval_part
        if has_conditional and has_donate and has_notdonate:
            return 'recipient-conditional\n(J / SS / IS+ style)'
        else:
            return 'recipient as scaling\n(Augmented IS)'
    if has_action and not has_recipient_rep:
        return 'IS-style\n(action only)'
    return 'other'

cats = [categorize(a['code']) for a in fp]
from collections import Counter
c = Counter(cats)
labels = list(c.keys())
sizes = list(c.values())
colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653'][:len(labels)]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: bar chart of categories
order = ['IS-style\n(action only)', 'recipient as scaling\n(Augmented IS)',
         'recipient-conditional\n(J / SS / IS+ style)', 'no learning\n(returns constant)']
ordered_sizes = [c.get(o, 0) for o in order]
ordered_labels = [o for o in order if c.get(o, 0) > 0]
ordered_sizes_nonzero = [c.get(o, 0) for o in ordered_labels]
ordered_colors = ['#2A9D8F', '#F4A261', '#E63946', '#264653'][:len(ordered_labels)]

bars = ax1.bar(range(len(ordered_labels)), ordered_sizes_nonzero, color=ordered_colors, alpha=0.85)
ax1.set_xticks(range(len(ordered_labels)))
ax1.set_xticklabels(ordered_labels, fontsize=9, rotation=0)
ax1.set_ylabel('Number of strategies (out of 15)', fontsize=11)
ax1.set_title('Final-population strategy profile\n(Exp 6, full obs, seed 0, n=15)', fontsize=11)
ax1.set_ylim(0, 16)
ax1.grid(axis='y', alpha=0.3)
for bar, size in zip(bars, ordered_sizes_nonzero):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{size}', ha='center', fontsize=11, fontweight='bold')

# Right: pie chart
ax2.pie(ordered_sizes_nonzero, labels=ordered_labels, colors=ordered_colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
ax2.set_title('Same data, as proportions', fontsize=11)

plt.suptitle('How LLM-evolved strategies use the augmented observation fields\n'
             '(donor_reputation + recipient_reputation)',
             fontsize=12, y=1.02)
plt.tight_layout()

out_pdf = d + '/final_pop_profile.pdf'
out_png = d + '/final_pop_profile.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')
print(f'Saved {out_png}')
