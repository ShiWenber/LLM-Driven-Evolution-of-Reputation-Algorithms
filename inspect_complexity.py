"""Look at 5 final-pop strategies from each obs level to see algorithm complexity."""
import json, os
from collections import Counter

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5'

# Look at full obs seed 0 (the one that hit 0.97)
print('=== full_seed0 (final coop 0.97) ===')
d = os.path.join(base, 'full_seed0')
files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
with open(os.path.join(d, files[0])) as f:
    t = json.load(f)

# Number of distinct strategy types by AST surface (unique code patterns)
unique_codes = set()
for a in t['final_population']:
    unique_codes.add(a['code'])
print(f'final_pop: {len(t["final_population"])} agents')
print(f'unique codes: {len(unique_codes)} / {len(t["final_population"])}')
print(f'mean len of code (chars): {sum(len(a["code"]) for a in t["final_population"])/len(t["final_population"]):.0f}')
print(f'max len: {max(len(a["code"]) for a in t["final_population"])}')
print(f'min len: {min(len(a["code"]) for a in t["final_population"])}')

# Count how many use any 'learning' pattern
import re
patterns = {
    'EMA-like': r'0\.\d+\s*\*\s*(rep|current_reputation|target|recipient)',
    'gradient': r'grad|delta|step|update|alpha|beta|gamma',
    'count_history': r'count|sum\(|len\(',
    'threshold_only': r'(>=|<=|>|<)\s*[-\d.]+\s*$',
    'state_machine': r'if\s+(state|phase|mode|stage|epoch)\s*[=:]',
    'discount_factor': r'discount|decay|forget|tau|lambda',
}
for name, pat in patterns.items():
    n = sum(1 for a in t['final_population'] if re.search(pat, a['code'], re.MULTILINE))
    print(f'  {name:20s}: {n} / {len(t["final_population"])} agents')

# Print 1 unique sample
print('\n--- sample 1 ---')
print(list(unique_codes)[0][:2000])
print('\n--- sample 2 (if different) ---')
if len(unique_codes) > 1:
    print(list(unique_codes)[1][:2000])
print('\n--- sample 3 (if different) ---')
if len(unique_codes) > 2:
    print(list(unique_codes)[2][:2000])