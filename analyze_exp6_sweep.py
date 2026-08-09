"""Compute summary statistics for exp6_sweep_AB (3 seeds x 4 obs = 12 trials)."""
import json, os, statistics
from collections import Counter
import re

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_AB'
obs_levels = ['private', 'partial_0.3', 'partial_0.7', 'full']
print('=== Exp 6 sweep (A/B labels, 3 seeds x 4 obs) ===\n')
print(f'{"observability":20s} {"mean":>8s} {"std":>8s} {"min":>8s} {"max":>8s} {"per-seed finals":40s}')
print('-' * 96)
for obs in obs_levels:
    finals = []
    for seed in range(3):
        d = os.path.join(base, f'{obs}_seed{seed}')
        if not os.path.exists(d):
            continue
        # Find the evo_*.json file — pick the most recent one (concurrent run
        # may leave multiple evo_*.json files in the same dir)
        evo_files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
        if not evo_files:
            continue
        # Pick most recent
        evo_files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
        with open(os.path.join(d, evo_files[0])) as f:
            t = json.load(f)
        coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
        finals.append(coops[-1])
    if not finals:
        continue
    mean = statistics.mean(finals)
    std = statistics.stdev(finals) if len(finals) > 1 else 0
    print(f'{obs:20s} {mean:8.3f} {std:8.3f} {min(finals):8.3f} {max(finals):8.3f}  {[round(f,3) for f in finals]}')

# Strategy classification across all 12 trials
print()
print('=== Strategy classification (12 trials x 15 = 180 strategies) ===')
print()
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
    has_random = 'random' in code and 'return' in code
    has_round = 'round_num' in code
    if has_return_true and not has_threshold and not has_random: return 'ALLC'
    if has_return_false and not has_threshold and not has_random: return 'ALLD'
    if has_action and has_threshold and not has_my_history: return 'ImageScoring'
    if has_action and has_threshold and has_my_history: return 'Hybrid'
    if has_random and not has_threshold: return 'RandomStrategy'
    if has_threshold and not has_action: return 'ThresholdOnly'
    if has_my_history and not has_threshold: return 'DirectExperience'
    if has_round and not has_threshold: return 'RoundDependent'
    return 'Other'

classes = Counter()
per_obs = {obs: Counter() for obs in obs_levels}
for obs in obs_levels:
    for seed in range(3):
        d = os.path.join(base, f'{obs}_seed{seed}')
        if not os.path.exists(d):
            continue
        evo_files = [f for f in os.listdir(d) if f.startswith('evo_')]
        if not evo_files:
            continue
        with open(os.path.join(d, evo_files[0])) as f:
            t = json.load(f)
        for a in t['final_population']:
            cls = classify(a.get('code', ''))
            classes[cls] += 1
            per_obs[obs][cls] += 1

print(f'{"class":20s} {"total":>8s} {"%":>8s}')
print('-' * 40)
total = sum(classes.values())
for cls, c in classes.most_common():
    print(f'{cls:20s} {c:8d} {100*c/total:7.1f}%')

print()
print('=== Per-condition classifier distribution ===')
for obs in obs_levels:
    n = sum(per_obs[obs].values())
    print(f'\n{obs} (n={n} agents):')
    for cls, c in per_obs[obs].most_common():
        print(f'  {cls:20s} {c:3d}  ({100*c/n:.1f}%)')
