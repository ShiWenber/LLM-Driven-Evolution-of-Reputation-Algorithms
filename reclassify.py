import os, json, re
from collections import Counter

OBS_KEY_DOUBLE = 'observation["action"]'
OBS_KEY_SINGLE = "observation['action']"

def classify(code):
    if not code: return 'Other'
    has_observe_action = (OBS_KEY_DOUBLE in code) or (OBS_KEY_SINGLE in code)
    has_threshold = ('recipient_reputation' in code) and (re.search(r'>=|>|==|<|<=', code) is not None)
    has_my_history = 'my_history' in code
    has_return_true = re.search(r'return\s+True\b', code) and not has_threshold and ('random' not in code)
    has_return_false = re.search(r'return\s+False\b', code) and not has_threshold and ('random' not in code)
    has_random = 'random' in code
    has_round = 'round_num' in code
    if has_return_true and not has_threshold and not has_random:
        return 'ALLC'
    if has_return_false and not has_threshold and not has_random:
        return 'ALLD'
    if has_observe_action and has_threshold and not has_my_history:
        return 'ImageScoring'
    if has_observe_action and has_threshold and has_my_history:
        return 'Hybrid'
    if has_random and not has_threshold:
        return 'RandomStrategy'
    if has_threshold and not has_observe_action:
        return 'ThresholdOnly'
    if has_my_history and not has_threshold:
        return 'DirectExperience'
    if has_round and not has_threshold:
        return 'RoundDependent'
    return 'Other'

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results'
sources = []

# n=10 LLM-evo
for cond in ['private', 'partial_0.3', 'partial_0.7', 'full']:
    for seed in range(10):
        d = os.path.join(base, 'exp1_method_n10', f'{cond}_seed{seed}')
        if not os.path.exists(d): continue
        evo_files = [f for f in os.listdir(d) if f.startswith('evo_')]
        if not evo_files: continue
        with open(os.path.join(d, evo_files[0])) as fh:
            t = json.load(fh)
        for a in t.get('final_population', []):
            sources.append(('n10', cond, seed, a.get('code','')))

# exp2 (n=2)
for d_name in os.listdir(os.path.join(base, 'exp2_threshold')):
    path = os.path.join(base, 'exp2_threshold', d_name)
    if not os.path.isdir(path): continue
    for f in os.listdir(path):
        if not f.endswith('.json'): continue
        try:
            with open(os.path.join(path, f)) as fh:
                t = json.load(fh)
        except: continue
        for a in t.get('final_population', []):
            sources.append(('exp2', d_name, -1, a.get('code','')))

# robustness
for d_name in os.listdir(os.path.join(base, 'exp5_robustness')):
    path = os.path.join(base, 'exp5_robustness', d_name)
    if not os.path.isdir(path): continue
    for f in os.listdir(path):
        if not f.endswith('.json'): continue
        try:
            with open(os.path.join(path, f)) as fh:
                t = json.load(fh)
        except: continue
        for a in t.get('final_population', []):
            sources.append(('robust', d_name, -1, a.get('code','')))

print('total strategies:', len(sources))
classes = {}
for src, cond, seed, code in sources:
    cls = classify(code)
    classes.setdefault(cls, []).append((src, cond, seed))

for cls in classes:
    print(f'{cls:20s} {len(classes[cls]):4d}  ({100*len(classes[cls])/len(sources):.1f}%)')

print()
print('=== per-condition ===')
cond_dist = {}
for src, cond, seed, code in sources:
    cond_dist.setdefault(cond, Counter())[classify(code)] += 1
for cond in sorted(cond_dist.keys()):
    total = sum(cond_dist[cond].values())
    print(f'{cond:20s} (n={total})')
    for cls in ['Hybrid', 'ALLD', 'ALLC', 'ImageScoring', 'Other']:
        c = cond_dist[cond].get(cls, 0)
        if c: print(f'  {cls:20s} {c:3d}  ({100*c/total:.1f}%)')
