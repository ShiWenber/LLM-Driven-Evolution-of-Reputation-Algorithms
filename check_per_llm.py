import json, re
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results')
by_llm = {}
for exp_dir in ['exp1_method_n10', 'exp3_static_g10_n10', 'exp4_random_mut',
                'exp5_robustness', 'exp6_sweep_AB_n5', 'exp6_sweep_CD_n5',
                'exp7_algorithmic_ceiling', 'exp8_intern_ceiling_v18',
                'exp8_intern_ceiling_v19_A', 'exp9_bc_scan']:
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
            m = re.search(r'def evaluate\([^)]*\):(.*?)(?=\ndef |\Z)', code, re.S)
            if not m: continue
            body = m.group(1)
            by_llm.setdefault(llm, {'total': 0, 'uses_recipient_rep': 0,
                                    'uses_donor_rep': 0, 'uses_action': 0})
            by_llm[llm]['total'] += 1
            if 'recipient_reputation' in body: by_llm[llm]['uses_recipient_rep'] += 1
            if 'donor_reputation' in body: by_llm[llm]['uses_donor_rep'] += 1
            if re.search(r'observation\[.+\baction\b', body): by_llm[llm]['uses_action'] += 1

for llm, c in by_llm.items():
    t = max(1, c['total'])
    pct_r = 100 * c['uses_recipient_rep'] / t
    pct_d = 100 * c['uses_donor_rep'] / t
    pct_a = 100 * c['uses_action'] / t
    print(f'{llm}:')
    print(f'  total: {c["total"]}')
    print(f'  recipient_reputation: {c["uses_recipient_rep"]} ({pct_r:.1f}%)')
    print(f'  donor_reputation:     {c["uses_donor_rep"]} ({pct_d:.1f}%)')
    print(f'  action:               {c["uses_action"]} ({pct_a:.1f}%)')
