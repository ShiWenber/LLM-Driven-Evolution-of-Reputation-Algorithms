"""Diagnose: why are so many strategies not getting extracted?"""
import json, glob, os, re
base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"
target_dirs = ['exp1_method_n10', 'exp5_robustness', 'exp6_sweep_AB_n5', 'exp6_sweep_CD_n5', 'exp7_algorithmic_ceiling', 'exp8_intern_ceiling_v18', 'exp8_intern_ceiling_v19_A', 'exp9_bc_scan']

all_strats = []
for d in target_dirs:
    pattern = os.path.join(base, d, '**', 'evo_*.json')
    for f in glob.glob(pattern, recursive=True):
        try:
            with open(f, encoding='utf-8') as fp: data = json.load(fp)
        except: continue
        if 'final_population' not in data: continue
        for a in data['final_population']:
            all_strats.append(a.get('code', ''))

# Check: how many strategies have an action-comparison pattern?
n_action_branch = 0
n_coop_lit = 0
n_defect_lit = 0
for c in all_strats:
    if re.search(r"if\s+(action|observation\[.action.\])\s*==", c):
        n_action_branch += 1
    if 'cooperate' in c:
        n_coop_lit += 1
    if 'defect' in c:
        n_defect_lit += 1

print(f"Total: {len(all_strats)}")
print(f"With if action == X branch:  {n_action_branch} ({100*n_action_branch/len(all_strats):.1f}%)")
print(f"Containing 'cooperate' literal:  {n_coop_lit} ({100*n_coop_lit/len(all_strats):.1f}%)")
print(f"Containing 'defect' literal:    {n_defect_lit} ({100*n_defect_lit/len(all_strats):.1f}%)")

# In all_strats with both literals, find the typical pattern of the gap
# between 'cooperate' literal and the first number
for c in all_strats[:5]:
    print("\n---")
    print(c[:500])
