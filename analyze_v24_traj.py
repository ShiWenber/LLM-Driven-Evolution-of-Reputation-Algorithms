"""Compare per-generation cooperation trajectory: thinking=on vs thinking=off.
Look at whether thinking changes the TRAJECTORY (faster learning, more
exploration), not just the final endpoint.
"""
import json
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp11_thinking_compare')

# Same mtime-based pairing as analyze_v24
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

# Per-generation cooperation trajectory
print('=== Per-generation mean cooperation rate ===')
print(f'{"obs":>12s} {"mode":>4s} {" ".join(f"g{i:>2d}" for i in range(10))}')
for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
    for mode in ['on', 'off']:
        td = results.get((obs, mode))
        if not td: continue
        d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
        traj = d.get('trajectory', [])
        coops = [t.get('cooperation_rate_mean', 0) or 0 for t in traj]
        coops_str = ' '.join(f'{c:.2f}' for c in coops)
        print(f'{obs:>12s} {mode:>4s} {coops_str}')

# Aggregate: by generation, mean coop across 4 obs
print('\n=== Mean across 4 obs by generation ===')
print(f'{"mode":>4s} {" ".join(f"g{i:>2d}" for i in range(10))}')
for mode in ['on', 'off']:
    coops_by_gen = [[] for _ in range(10)]
    for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
        td = results.get((obs, mode))
        if not td: continue
        d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
        for i, t in enumerate(d.get('trajectory', [])):
            if i < 10:
                coops_by_gen[i].append(t.get('cooperation_rate_mean', 0) or 0)
    means = [sum(c)/len(c) if c else 0 for c in coops_by_gen]
    means_str = ' '.join(f'{m:.3f}' for m in means)
    print(f'{mode:>4s} {means_str}')

# Compare first 3 gens vs last 3 gens (early vs late coop)
print('\n=== Early vs late cooperation (mean over 4 obs) ===')
print(f'{"mode":>4s} {"gen0-2":>15s} {"gen3-6":>15s} {"gen7-9":>15s} {"delta(7-9)-(0-2)":>20s}')
for mode in ['on', 'off']:
    early, mid, late = [], [], []
    for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
        td = results.get((obs, mode))
        if not td: continue
        d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
        traj = d.get('trajectory', [])
        if len(traj) < 10: continue
        for i, t in enumerate(traj):
            c = t.get('cooperation_rate_mean', 0) or 0
            if i < 3: early.append(c)
            elif i < 7: mid.append(c)
            else: late.append(c)
    e_m = sum(early)/max(1,len(early))
    m_m = sum(mid)/max(1,len(mid))
    l_m = sum(late)/max(1,len(late))
    print(f'{mode:>4s} {e_m:>15.3f} {m_m:>15.3f} {l_m:>15.3f} {l_m - e_m:>+20.3f}')
