"""Check if private / partial_0.3 differences are robust by examining
specific high-cooperation strategies in those trials."""
import json
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp11_thinking_compare')

# mtime-based pairing
evo_by_obs = {}
for evo in RES.glob('evo_*.json'):
    name = evo.name
    rest = name.replace('evo_', '').replace('.json', '')
    parts_no_ts = rest.rsplit('_', 2)[0]
    obs = parts_no_ts.replace('_deepseek-v4-flash', '')
    evo_by_obs.setdefault(obs, []).append(evo)

results = {}
for obs, evos in evo_by_obs.items():
    r_path = RES / f'{obs}_seed0' / f'reasoning_deepseek-v4-flash_{obs}_seed0.json'
    has_reasoning = r_path.exists() and any(
        e.get('reasoning', '')
        for e in json.loads(r_path.read_text(encoding='utf-8', errors='ignore'))
    )
    if has_reasoning and len(evos) == 2:
        sorted_evos = sorted(evos, key=lambda p: p.stat().st_mtime)
        results[(obs, 'on')] = sorted_evos[0]
        results[(obs, 'off')] = sorted_evos[1]

# Show high-cooperation strategies in private + partial_0.3 (where on > off)
print('=== High-cooperation strategies (cooperation > 0.3) ===')
for obs in ['private', 'partial_0.3']:
    for mode in ['on', 'off']:
        td = results.get((obs, mode))
        if not td: continue
        d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
        fp = d.get('final_population', [])
        high = [a for a in fp if isinstance(a, dict) and (a.get('cooperation_rate', 0) or 0) > 0.3]
        print(f'\n{obs} thinking={mode}: {len(high)} high-coop strategies (out of {len(fp)}):')
        for a in high[:3]:
            code = a.get('code', '')
            print(f'  agent {a["agent_id"]} coop={a.get("cooperation_rate"):.3f}')
            # Show first 20 lines
            lines = code.split('\n')[:15]
            for ln in lines:
                print(f'    {ln}')
            print('    ...')
