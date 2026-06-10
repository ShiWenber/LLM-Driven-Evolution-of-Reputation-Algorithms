import json, re
from pathlib import Path
from collections import Counter

def my_classify(code):
    if not code: return 'NoCode'
    has_image = bool(re.search(r"observation\s*\[\s*['\"]action['\"]\s*\]\s*==\s*['\"]donate['\"]", code))
    has_history = bool(re.search(r"my_history", code))
    has_threshold = bool(re.search(r"recipient_reputation\s*[><=!]+", code))
    has_random = bool(re.search(r"random\.", code))
    if re.search(r"return\s+True\s*$", code.strip(), re.MULTILINE) and not has_threshold and not has_random:
        return 'ALLC'
    if re.search(r"return\s+False\s*$", code.strip(), re.MULTILINE) and not has_threshold and not has_random:
        return 'ALLD'
    if has_random and not has_image and not has_history:
        return 'RandomStrategy'
    if has_image and has_threshold and not has_history:
        return 'ImageScoring'
    if has_image and has_threshold and has_history:
        return 'Hybrid'
    if has_threshold and not has_image and not has_history:
        return 'ThresholdOnly'
    if has_history and not has_threshold:
        return 'DirectExperience'
    if re.search(r"round_num", code):
        if not has_threshold:
            return 'RoundDependent'
    if has_image:
        return 'ImageScoringUnstructured'
    return 'Other'

EXP1 = Path('results/exp1_method')
d = json.load(open('results/exp1_method/full_seed0/evo_full_deepseek-v4-flash_20260606_143421.json'))
fp = d['final_population']
n_gens = max((a.get('generation', 0) or 0) for a in fp)
final_agents = [a for a in fp if a.get('generation') == n_gens]
print(f'final_agents n_gens = {n_gens}, n = {len(final_agents)}')
classes = Counter()
for a in final_agents:
    code = a.get('code', '')
    classes[my_classify(code)] += 1
print('Per-agent classes:', classes)
print()
print('=== First agent code (first 500 chars) ===')
print(final_agents[0].get('code', '')[:500])
