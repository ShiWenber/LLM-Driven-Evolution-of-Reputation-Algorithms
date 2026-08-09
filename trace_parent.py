"""For each of 23 recipient_rep strategies, find parent strategy code.
This shows LLM's actual mutation decisions: parent_code -> child_code."""
import json, re
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results')

def get_recipient_rep_strategies():
    found = []
    for exp_dir in ['exp1_method_n10', 'exp5_robustness', 'exp6_sweep_AB_n5']:
        base = RES / exp_dir
        if not base.exists(): continue
        for td in base.rglob('evo_*.json'):
            try:
                d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
            except: continue
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
                if 'recipient_reputation' in body:
                    found.append((td, a))
    return found

found = get_recipient_rep_strategies()
print(f'Found {len(found)} recipient_rep strategies. Looking up parents...\n')

for td, target in found:
    d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
    fp = d.get('final_population') or []
    # Index by strategy_id
    by_sid = {a.get('strategy_id'): a for a in fp if isinstance(a, dict)}
    pid = target.get('parent_id')
    print(f'\n--- target: {target.get("agent_id")} sid={target.get("strategy_id")} gen={target.get("generation")} coop={target.get("cooperation_rate"):.3f}')
    if pid is None:
        print('   NO PARENT (this is gen-0 root strategy!)')
        continue
    if pid in by_sid:
        p = by_sid[pid]
        print(f'   parent: sid={pid} gen={p.get("generation")} fit={p.get("fitness")} coop={p.get("cooperation_rate"):.3f}')
        pcode = p.get('code', '')
        pbody_match = re.search(r'def evaluate\([^)]*\):(.*?)(?=\ndef |\Z)', pcode, re.S)
        p_uses_rec = 'recipient_reputation' in pbody_match.group(1) if pbody_match else False
        print(f'   parent uses recipient_reputation: {p_uses_rec}')
        # Show diff: did child keep rec_rep from parent? Or introduce new?
        tcode = target.get('code', '')
        t_has_rec = 'recipient_reputation' in tcode
        if not p_uses_rec and t_has_rec:
            print(f'   --> LLM INTRODUCED recipient_reputation in this mutation')
        elif p_uses_rec and t_has_rec:
            print(f'   --> LLM KEPT recipient_reputation from parent')
        elif p_uses_rec and not t_has_rec:
            print(f'   --> LLM REMOVED recipient_reputation')
    else:
        print(f'   parent {pid} NOT in final_population (must have been from earlier gen that died out)')
