"""Trace mutation chain for the 23 strategies that use recipient_reputation."""
import json, re
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results')

# Find all 23 strategies
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

print(f'Found {len(found)} recipient_rep strategies. Tracing their mutation chains...\n')

# Build strategy_id -> code index per trial
for td, target in found:
    d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
    fp = d.get('final_population') or []
    if not isinstance(fp, list): continue

    # Index all known strategies in this trial by strategy_id
    by_id = {}
    for a in fp:
        by_id[a.get('strategy_id')] = a

    # Also need to walk through trajectory generations
    # trajectory[i] has the population of generation i
    by_gen = {}
    for i, g in enumerate(d.get('trajectory', [])):
        pop = g.get('population', [])
        for a in pop:
            by_gen[(i, a.get('strategy_id'))] = a

    # Walk back parent_id chain
    chain = []
    cur = target
    for hop in range(15):
        chain.append({
            'gen': cur.get('generation', '?'),
            'sid': cur.get('strategy_id'),
            'pid': cur.get('parent_id'),
            'fit': cur.get('fitness'),
            'coop': cur.get('cooperation_rate'),
            'uses_recipient_rep': 'recipient_reputation' in (re.search(r'def evaluate\([^)]*\):(.*?)(?=\ndef |\Z)', cur.get('code',''), re.S).group(1) if 'def evaluate' in cur.get('code','') else ''),
        })
        pid = cur.get('parent_id')
        if pid is None: break
        # Find in trajectory
        prev_gen = cur.get('generation', 0) - 1
        prev = by_gen.get((prev_gen, pid))
        if prev is None:
            chain.append({'NOTE': f'parent {pid} in gen {prev_gen} not in trajectory.population, may be in elites/pool'})
            break
        cur = prev

    print(f'\n{"="*78}')
    print(f'Path: {td.relative_to(RES)} | agent_id={target.get("agent_id")} sid={target.get("strategy_id")} coop={target.get("cooperation_rate")}')
    print('MUTATION CHAIN (target -> ... -> root):')
    for i, hop in enumerate(reversed(chain)):
        marker = '  ROOT' if i == 0 else f'  hop {i}'
        if 'NOTE' in hop:
            print(f'{marker}: {hop["NOTE"]}')
        else:
            ur = 'REC' if hop['uses_recipient_rep'] else '   '
            print(f'{marker}: gen={hop["gen"]:>2} sid={hop["sid"]} pid={hop["pid"]} fit={hop["fit"]} coop={hop["coop"]:.3f} [{ur}]')
