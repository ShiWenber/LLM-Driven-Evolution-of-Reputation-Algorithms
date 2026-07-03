"""Quick check: which random mutation dirs have actual evo JSON?"""
import os, json
from pathlib import Path
RAND = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp4_random_mut')
ok = []
for d in RAND.iterdir():
    if d.is_dir():
        evos = [f for f in d.iterdir() if f.name.startswith('evo_') and f.name.endswith('.json')]
        if evos:
            try:
                with open(evos[0]) as f:
                    t = json.load(f)
                if t.get('trajectory'):
                    coop = t['trajectory'][-1]['cooperation_rate_mean']
                    ok.append((d.name, coop))
            except Exception as e:
                print(f'err {d.name}: {e}')
print(f'OK: {len(ok)}')
for n, c in sorted(ok):
    print(f'  {n}: coop={c:.3f}')