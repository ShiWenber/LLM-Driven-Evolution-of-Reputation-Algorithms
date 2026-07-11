"""For every final-population strategy, check whether evaluate() body
actually USES donor_reputation or recipient_reputation, not just receives it
as a parameter."""
import json, re
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results')

uses_donor_rep = 0
uses_recipient_rep = 0
uses_action = 0
uses_observation = 0
uses_my_history = 0
total_eval = 0
total_with_three_factors = 0  # uses donor_rep + recipient_rep + action

for exp_dir in ['exp1_method_n10', 'exp3_static_g10_n10', 'exp4_random_mut',
                'exp5_robustness', 'exp6_sweep_AB_n5', 'exp6_sweep_CD_n5',
                'exp6_sweep_donate_n3', 'exp7_algorithmic_ceiling',
                'exp8_intern_ceiling_v18', 'exp8_intern_ceiling_v19_A',
                'exp9_bc_scan']:
    base = RES / exp_dir
    if not base.exists(): continue
    for td in base.rglob('*.json'):
        try:
            d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
        except:
            continue
        if not isinstance(d, dict): continue
        fp = d.get('final_population') or []
        if not isinstance(fp, list): continue
        for a in fp:
            if not isinstance(a, dict): continue
            code = a.get('code') or ''
            if 'def evaluate' not in code: continue
            m = re.search(r'def evaluate\([^)]*\):(.*?)(?=\ndef |\Z)', code, re.S)
            if not m: continue
            body = m.group(1)
            total_eval += 1
            d_used = 'donor_reputation' in body
            r_used = 'recipient_reputation' in body
            a_used = bool(re.search(r"observation\[.+\baction\b", body))
            if d_used: uses_donor_rep += 1
            if r_used: uses_recipient_rep += 1
            if a_used: uses_action += 1
            if 'observation' in body: uses_observation += 1
            if 'my_history' in body: uses_my_history += 1
            if d_used and r_used and a_used:
                total_with_three_factors += 1

print(f'Total evaluate() functions across all final populations: {total_eval}')
print(f'  uses donor_reputation:     {uses_donor_rep} ({100*uses_donor_rep/max(1,total_eval):.1f}%)')
print(f'  uses recipient_reputation: {uses_recipient_rep} ({100*uses_recipient_rep/max(1,total_eval):.1f}%)')
print(f'  uses observation[action]:  {uses_action} ({100*uses_action/max(1,total_eval):.1f}%)')
print(f'  uses observation (any):    {uses_observation} ({100*uses_observation/max(1,total_eval):.1f}%)')
print(f'  uses my_history:           {uses_my_history} ({100*uses_my_history/max(1,total_eval):.1f}%)')
print(f'  ===> uses ALL THREE (donor_rep + recipient_rep + action): {total_with_three_factors} ({100*total_with_three_factors/max(1,total_eval):.2f}%)')
