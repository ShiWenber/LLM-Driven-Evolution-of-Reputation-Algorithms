"""Show b/c scan summary."""
import json
with open(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp9_bc_scan\_manifest.json') as f:
    d = json.load(f)
ok = [m for m in d if m.get('ok')]
notok = [m for m in d if not m.get('ok')]
print(f'OK: {len(ok)}/{len(d)}, fail: {len(notok)}')
for m in ok:
    bs = f'b{m["benefit"]}c{m["cost"]}'
    print(f'  {m["obs"]:10s} {bs} seed{m["seed"]}: coop={m["coop_final"]:.3f} ({m["elapsed_sec"]:.0f}s)')