import json
with open(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_AB\_manifest.json') as f:
    m = json.load(f)
print(f'{"obs":14s} {"seed":5s} {"elapsed_sec":>12s} {"elapsed_min":>12s}')
total = 0
for r in m:
    if 'elapsed_sec' in r:
        print(f"{r['obs']:14s} {r['seed']:5d} {r['elapsed_sec']:12.1f} {r['elapsed_sec']/60:12.1f}")
        total += r['elapsed_sec']
print(f'\nTotal: {total/60:.1f} min, {total/3600:.2f} h, {len(m)} trials')
