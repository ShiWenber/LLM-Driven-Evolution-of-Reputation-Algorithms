"""Plot b/c scan results: 3 obs x 4 ratios x 3 seeds."""
import json, os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

with open(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp9_bc_scan\_manifest.json') as f:
    d = json.load(f)
ok = [m for m in d if m.get('ok')]

# Group by obs, benefit
data = defaultdict(lambda: defaultdict(list))
for m in ok:
    data[m['obs']][m['benefit']].append(m['coop_final'])

RATIOS = [1.5, 2.0, 3.0, 4.0]
OBS = ['private', 'partial_0.7', 'full']

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = {'private': '#264653', 'partial_0.7': '#E76F51', 'full': '#2A9D8F'}
for obs in OBS:
    means = [np.mean(data[obs].get(r, [])) for r in RATIOS]
    stds = [np.std(data[obs].get(r, [])) if data[obs].get(r, []) else 0 for r in RATIOS]
    ax.errorbar(RATIOS, means, yerr=stds, marker='o', linewidth=2, capsize=4,
                label=obs, color=colors[obs])

ax.set_xlabel('Benefit / Cost ratio', fontsize=11)
ax.set_ylabel('Mean final cooperation rate', fontsize=11)
ax.set_title('b/c ratio scan: cooperation by observability x benefit level', fontsize=12)
ax.set_xticks(RATIOS)
ax.set_xticklabels([f'{r:.1f}' for r in RATIOS])
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
out_pdf = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp9_bc_scan\bc_scan.pdf'
out_png = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp9_bc_scan\bc_scan.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, bbox_inches='tight', dpi=200)
print(f'Saved {out_pdf}')

# Print table
print('\n=== b/c scan: final cooperation rates ===')
print(f'{"b/c":>6s}  {"private":>20s}  {"partial_0.7":>20s}  {"full":>20s}')
print('-' * 75)
for r in RATIOS:
    pvals = data['private'].get(r, [])
    p7vals = data['partial_0.7'].get(r, [])
    fvals = data['full'].get(r, [])
    pm = f'{np.mean(pvals):.2f} (n={len(pvals)})' if pvals else 'skip'
    p7m = f'{np.mean(p7vals):.2f} (n={len(p7vals)})' if p7vals else 'skip'
    fm = f'{np.mean(fvals):.2f} (n={len(fvals)})' if fvals else 'skip'
    print(f'{r:>6.1f}  {pm:>20s}  {p7m:>20s}  {fm:>20s}')

# Save table
table = {}
for obs in OBS:
    table[obs] = {}
    for r in RATIOS:
        vals = data[obs].get(r, [])
        table[obs][r] = {'mean': float(np.mean(vals)) if vals else None,
                          'n': len(vals),
                          'values': vals}
with open(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp9_bc_scan\bc_table.json', 'w') as f:
    json.dump(table, f, indent=2)
print('\nSaved table to bc_table.json')