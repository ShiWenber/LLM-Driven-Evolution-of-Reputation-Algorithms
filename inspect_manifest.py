"""Inspect the new manifest: count fallback / api_err / per-trial coop."""
import json
from pathlib import Path
m = json.loads(Path('results/_manifest.json').read_text())
api_err = 0
fallback = 0
coop = []
for e in m:
    t = e.get('stdout_tail', '') + e.get('stderr_tail', '')
    if 'invalid_api_key' in t or 'invalid_request_error' in t:
        api_err += 1
    if 'using random mutation fallback' in t or 'using fallback strategies' in t:
        fallback += 1
    line = e.get('stdout_tail', '').strip().split('\n')[-1]
    coop.append((e['i'], e['run'], e['obs'], e['seed'], line))

print(f"Trials: {len(m)}")
print(f"API errors: {api_err}")
print(f"Fallback events: {fallback}")
print()
for i, run, obs, seed, line in coop:
    print(f"  #{i:2d} {run:18s} {obs:12s} s={seed}  {line}")
