"""Classify decide() strategies in n=5 cooperate/defect sweep."""
import json, os, re
from collections import Counter, defaultdict

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_CD_n5'
obs_levels = ['private', 'partial_0.3', 'partial_0.7', 'full']

def split_decide(code):
    if 'def decide' in code:
        return code.split('def decide')[0], code.split('def decide')[1]
    return code, ''

def classify_decide(decide_code):
    d = decide_code
    has_threshold = re.search(r'recipient_reputation\s*[><=!]+\s*[-\d.]+', d) is not None
    has_dynamic_thr = bool(re.search(r'\bthreshold\s*=', d))
    has_my_history = 'my_history' in d
    has_round = 'round_num' in d
    has_return_true = re.search(r'return\s+True\b', d) is not None
    has_return_false = re.search(r'return\s+False\b', d) is not None
    n_if = len(re.findall(r'\bif\b', d))

    if has_return_true and not has_threshold and not has_dynamic_thr and not has_my_history:
        return 'F5a: always-True'
    if has_return_false and not has_threshold and not has_dynamic_thr and not has_my_history:
        return 'F5b: always-False'
    if has_threshold and has_dynamic_thr and has_my_history:
        return 'F4: dyn-threshold + my_history'
    if has_threshold and has_dynamic_thr:
        return 'F3: dyn-threshold + round'
    if has_threshold and has_my_history:
        return 'F2: threshold + my_history'
    if has_threshold:
        return 'F1: simple threshold'
    if not has_threshold and has_my_history:
        return 'F2b: my_history only'
    if n_if >= 2:
        return 'F7: complex multi-condition'
    return 'F8: other'

all_strategies = []
per_obs = defaultdict(list)
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
            code = a.get('code', '')
            eval_part, decide_part = split_decide(code)
            fam = classify_decide(decide_part)
            all_strategies.append((obs, seed, a.get('agent_id'), fam, decide_part.strip()))
            per_obs[obs].append(all_strategies[-1])

print('=== CD n=5 overall decide() family distribution (300 strategies) ===')
fc = Counter(e[3] for e in all_strategies)
for fam, c in fc.most_common():
    print(f'  {fam:42s} {c:4d}  ({100*c/len(all_strategies):5.1f}%)')

print()
print('=== Per-observability distribution (CD) ===')
for obs in obs_levels:
    print(f'\n{obs} (n={len(per_obs[obs])}):')
    fc = Counter(e[3] for e in per_obs[obs])
    for fam, c in fc.most_common():
        print(f'  {fam:42s} {c:4d}  ({100*c/len(per_obs[obs]):5.1f}%)')