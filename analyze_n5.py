"""Analyze 5-seed sweep: per-obs success rate, mean, std."""
import json, os, statistics
from collections import Counter
import re

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_AB_n5'
obs_levels = ['private', 'partial_0.3', 'partial_0.7', 'full']

print('=== Exp 6 sweep v2 (5 seeds x 4 obs, A/B label) ===\n')
print(f'{"obs":12s} {"mean":>8s} {"std":>8s} {"min":>8s} {"max":>8s} {"OK/total":>10s} {"per-seed finals"}')
print('-' * 100)

for obs in obs_levels:
    finals = []
    for seed in range(5):
        d = os.path.join(base, f'{obs}_seed{seed}')
        if not os.path.exists(d): continue
        files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
        if not files: continue
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        with open(os.path.join(d, files[0])) as f:
            t = json.load(f)
        coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
        finals.append(coops[-1])
    if not finals: continue
    mean = statistics.mean(finals)
    std = statistics.stdev(finals) if len(finals) > 1 else 0
    n_ok = sum(1 for f in finals if f > 0.5)
    print(f'{obs:12s} {mean:8.3f} {std:8.3f} {min(finals):8.3f} {max(finals):8.3f} {n_ok}/{len(finals):>3d}      {[round(f, 3) for f in finals]}')

print()
print('=== Strategy distribution (5 trials x 15 = 75 per obs, 300 total) ===')
OBS_D = 'observation["action"]'
OBS_S = "observation['action']"
def classify(code):
    if not code: return 'Other'
    has_action = (OBS_D in code) or (OBS_S in code)
    has_threshold = ('recipient_reputation' in code) and (re.search(r'>=|>|==|<|<=', code) is not None)
    has_my_history = 'my_history' in code
    has_A = "'A'" in code or '"A"' in code
    has_B = "'B'" in code or '"B"' in code
    has_return_true = re.search(r'return\s+True\b', code) and not has_threshold
    has_return_false = re.search(r'return\s+False\b', code) and not has_threshold
    if has_return_true and not has_threshold: return 'ALLC'
    if has_return_false and not has_threshold: return 'ALLD'
    if has_action and has_threshold and not has_my_history: return 'ImageScoring'
    if has_action and has_threshold and has_my_history: return 'Hybrid'
    if has_threshold and not has_action: return 'ThresholdOnly'
    return 'Other'

classes = Counter()
for obs in obs_levels:
    for seed in range(5):
        d = os.path.join(base, f'{obs}_seed{seed}')
        if not os.path.exists(d): continue
        files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
        if not files: continue
        files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        with open(os.path.join(d, files[0])) as f:
            t = json.load(f)
        for a in t['final_population']:
            classes[classify(a.get('code', ''))] += 1

total = sum(classes.values())
print(f'{"class":20s} {"count":>8s} {"%":>8s}')
print('-' * 40)
for cls, c in classes.most_common():
    print(f'{cls:20s} {c:8d} {100*c/total:7.1f}%')
