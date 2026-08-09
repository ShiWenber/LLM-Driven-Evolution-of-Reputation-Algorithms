import json, os, statistics, re

d = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'
with open(d + '/evo_full_deepseek-v4-flash_20260616_115153.json') as f:
    t = json.load(f)
print('top keys:', list(t.keys()))
if 'trajectory' in t:
    coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
elif 'trials_summary' in t:
    trial = t['trials_summary'][0]
    coops = [g['cooperation_rate_mean'] for g in trial['trajectory']]
else:
    print('No trajectory')
    coops = []

print('=== Exp 6 (A/B labels, 1 trial) trajectory ===')
for i, c in enumerate(coops):
    print(f'  gen {i}: {c:.3f}')
print(f'gen-0: {coops[0]:.3f}, gen-10: {coops[-1]:.3f}')

# Final pop analysis
fp = t['final_population']
print(f'\nFinal pop: {len(fp)} agents')

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
    if has_return_true and not has_threshold: return 'ALways_True'
    if has_return_false and not has_threshold: return 'Always_False'
    if has_action and has_threshold and not has_my_history: return 'ImageScoring'
    if has_action and has_threshold and has_my_history: return 'Hybrid'
    if has_threshold and not has_action: return 'ThresholdOnly'
    return 'Other'

from collections import Counter
classes = Counter()
for a in fp:
    classes[classify(a.get('code', ''))] += 1
print()
print('=== Classifier distribution ===')
for cls, c in classes.most_common():
    print(f'  {cls:20s} {c:3d}  ({100*c/len(fp):.1f}%)')

# Count uses of A vs B
n_use_A = sum(1 for a in fp if "'A'" in a.get('code', '') or '"A"' in a.get('code', ''))
n_use_B = sum(1 for a in fp if "'B'" in a.get('code', '') or '"B"' in a.get('code', ''))
print(f'\nStrategies using "A": {n_use_A}/{len(fp)}')
print(f'Strategies using "B": {n_use_B}/{len(fp)}')

# Show 1-2 short examples
print('\n=== Sample strategies ===')
for i, a in enumerate(fp[:3]):
    code = a.get('code', '')
    aid = a.get('agent_id')
    print(f'\n--- Agent {aid} (len={len(code)}) ---')
    print(code[:1200])
