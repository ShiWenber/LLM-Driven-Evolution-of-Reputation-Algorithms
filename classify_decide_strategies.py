"""Classify all decide() strategies in the n=5 A/B sweep final populations.
Group by family:
  F1: simple threshold (recipient_reputation >= constant)
  F2: threshold + recency (uses my_history actions or count)
  F3: round-dependent threshold
  F4: recipient_reputation + dynamic threshold from my_history
  F5: always True / always False (constant)
  F6: standing-like (recipient_rep conditional on rep)
  F7: complex multi-condition
"""
import json, os, re
from collections import Counter, defaultdict

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp6_sweep_AB_n5'
obs_levels = ['private', 'partial_0.3', 'partial_0.7', 'full']

def split_decide(code):
    """Split code into evaluate-part and decide-part."""
    if 'def decide' in code:
        return code.split('def decide')[0], code.split('def decide')[1]
    return code, ''

def classify_decide(decide_code):
    """Return (family_name, short_description)."""
    d = decide_code
    has_threshold = re.search(r'recipient_reputation\s*[><=!]+\s*[-\d.]+', d) is not None
    has_dynamic_thr = bool(re.search(r'\bthreshold\s*=', d))
    has_my_history = 'my_history' in d
    has_round = 'round_num' in d
    has_random = 'random' in d and 'return' in d
    has_return_true = re.search(r'return\s+True\b', d) is not None
    has_return_false = re.search(r'return\s+False\b', d) is not None
    has_constant = re.search(r'recipient_reputation\s*[><=!]+\s*-?\d+\.?\d*\s*$', d.strip()) is not None
    n_if = len(re.findall(r'\bif\b', d))

    if has_return_true and not has_threshold and not has_dynamic_thr and not has_my_history:
        return ('F5a: always-True', 'return True regardless of anything')
    if has_return_false and not has_threshold and not has_dynamic_thr and not has_my_history:
        return ('F5b: always-False', 'return False regardless of anything')

    if has_threshold and has_dynamic_thr and has_my_history:
        return ('F4: dyn-threshold + my_history', 'threshold = f(round, my_history) AND recipient > thr')
    if has_threshold and has_dynamic_thr:
        return ('F3: dyn-threshold + round', 'threshold = f(round) AND recipient > thr')
    if has_threshold and has_my_history:
        return ('F2: threshold + my_history', 'recipient > const AND uses my_history')
    if has_threshold:
        return ('F1: simple threshold', 'recipient_reputation > constant')

    if not has_threshold and has_my_history:
        return ('F2b: my_history only', 'uses history without explicit threshold')

    if n_if >= 2:
        return ('F7: complex multi-condition', f'{n_if} if branches')
    return ('F8: other', '')

# Collect all strategies across 20 trials
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
            family, desc = classify_decide(decide_part)
            entry = (obs, seed, a.get('agent_id'), family, desc, decide_part.strip())
            all_strategies.append(entry)
            per_obs[obs].append(entry)

print('=== Overall decide() family distribution (5 seeds x 4 obs x 15 = 300 strategies) ===')
family_counter = Counter(e[3] for e in all_strategies)
total = len(all_strategies)
for fam, c in family_counter.most_common():
    print(f'  {fam:42s} {c:4d}  ({100*c/total:5.1f}%)')

print()
print('=== Per-observability distribution ===')
for obs in obs_levels:
    print(f'\n{obs} (n={len(per_obs[obs])}):')
    fc = Counter(e[3] for e in per_obs[obs])
    for fam, c in fc.most_common():
        print(f'  {fam:42s} {c:4d}  ({100*c/len(per_obs[obs]):5.1f}%)')

# Show 2-3 real examples per major family
print()
print('=== Real strategy samples (1 per family) ===')
seen = set()
for entry in all_strategies:
    fam = entry[3]
    if fam in seen: continue
    seen.add(fam)
    print(f'\n--- {fam} (obs={entry[0]}, seed={entry[1]}, agent={entry[2]}) ---')
    code = entry[5]
    # show first 600 chars
    print(code[:600] + ('...' if len(code) > 600 else ''))