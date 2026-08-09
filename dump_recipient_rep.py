"""Show the 23 strategies that actually use recipient_reputation in their evaluate body."""
import json, re, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results')

found = []
for exp_dir in ['exp1_method_n10', 'exp3_static_g10_n10', 'exp4_random_mut',
                'exp5_robustness', 'exp6_sweep_AB_n5', 'exp6_sweep_CD_n5',
                'exp6_sweep_donate_n3', 'exp7_algorithmic_ceiling',
                'exp8_intern_ceiling_v18', 'exp8_intern_ceiling_v19_A',
                'exp9_bc_scan']:
    base = RES / exp_dir
    if not base.exists(): continue
    for td in base.rglob('evo_*.json'):
        try:
            d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
        except: continue
        if not isinstance(d, dict): continue
        fp = d.get('final_population') or []
        if not isinstance(fp, list): continue
        llm = (d.get('model') or 'unknown')
        if llm == 'unknown':
            for k in ['deepseek-v4-flash', 'deepseek-coder', 'Intern-S2-Preview', 'paratera-intern']:
                if k in str(td): llm = k
        for a in fp:
            if not isinstance(a, dict): continue
            code = a.get('code') or ''
            if 'def evaluate' not in code: continue
            m = re.search(r'def evaluate\([^)]*\):(.*?)(?=\ndef |\Z)', code, re.S)
            if not m: continue
            body = m.group(1)
            if 'recipient_reputation' in body:
                coop = a.get('cooperation_rate', 0) or 0
                fit = a.get('fitness')
                trial = str(td.relative_to(RES))
                aid = a.get('agent_id')
                found.append((llm, exp_dir, trial, aid, coop, fit, code))

print(f'Total strategies using recipient_reputation in evaluate body: {len(found)}')
seen = set()
unique = []
for entry in found:
    if entry[6] not in seen:
        seen.add(entry[6])
        unique.append(entry)
print(f'Unique code variants: {len(unique)}')
print()

for i, (llm, exp_dir, trial, aid, coop, fit, code) in enumerate(unique, 1):
    print('='*78)
    print(f'#{i:>2d} | {llm} | exp={exp_dir}')
    print(f'    trial={trial}  agent_id={aid}  cooperation={coop:.3f}  fitness={fit}')
    print('='*78)
    print(code)
    print()
