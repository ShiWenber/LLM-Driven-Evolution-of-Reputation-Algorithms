import json, re

with open(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\gen0_inspection\gen0_population.json') as f:
    d = json.load(f)

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
    if has_return_true and not has_threshold and not has_random: return 'ALLC'
    if has_return_false and not has_threshold and not has_random: return 'ALLD'
    if has_observe_action and has_threshold and not has_my_history: return 'ImageScoring'
    if has_observe_action and has_threshold and has_my_history: return 'Hybrid'
    if has_random and not has_threshold: return 'RandomStrategy'
    if has_threshold and not has_observe_action: return 'ThresholdOnly'
    if has_my_history and not has_threshold: return 'DirectExperience'
    if has_round and not has_threshold: return 'RoundDependent'
    return 'Other'

# Classify and predict cooperation rate
def predict_coop_rate(code, donor_history=None):
    """Simulate gen-0 against itself: estimate mean cooperation rate."""
    if not code: return 0.0
    # Crude: ALLC=1.0, ALLD=0.0, RandomStrategy=0.5,
    # Hybrid/ImageScoring/ThresholdOnly: depends on recipient_reputation distribution
    has_threshold = ('recipient_reputation' in code) and (re.search(r'>=|>|==|<|<=', code) is not None)
    has_observe = 'observation["action"]' in code or "observation['action']" in code
    has_return_true = re.search(r'return\s+True\b', code)
    has_return_false = re.search(r'return\s+False\b', code)
    has_random = 'random' in code and 'return' in code
    if has_return_true and not has_threshold and not has_random: return 1.0
    if has_return_false and not has_threshold and not has_random: return 0.0
    if has_random and not has_threshold: return 0.5
    if has_threshold and has_observe:
        # Image-Scoring style: depends on initial rep
        # Most initial reps are 0.01 (default), so threshold >= 0.0 means donate
        # threshold > 0.0 or >= 0.5 means not donate
        m = re.search(r'recipient_reputation\s*([><=!]+)\s*([-\d.]+)', code)
        if m:
            op, val = m.group(1), float(m.group(2))
            # If threshold is 0.0, almost always donate (recipients start at 0.01)
            if op in ('>=', '>') and val <= 0.0: return 0.95
            if op in ('>=', '>') and val <= 0.5: return 0.6
            if op in ('>=', '>') and val > 0.5: return 0.2
            if op in ('==',) and val == 0.0: return 0.05
            if op in ('<', '<=') and val >= 0.0: return 0.05
        return 0.5
    if has_threshold and not has_observe:
        # ThresholdOnly: same
        m = re.search(r'recipient_reputation\s*([><=!]+)\s*([-\d.]+)', code)
        if m:
            op, val = m.group(1), float(m.group(2))
            if op in ('>=', '>') and val <= 0.0: return 0.95
            if op in ('>=', '>') and val <= 0.5: return 0.6
            if op in ('>=', '>') and val > 0.5: return 0.2
        return 0.5
    return 0.5

# Classify all
from collections import Counter
classes = Counter()
coops = []
for s in d['strategies']:
    cls = classify(s['code'])
    cr = predict_coop_rate(s['code'])
    classes[cls] += 1
    coops.append((s['agent_id'], cls, cr, s['code'][:50].replace('\n', ' ')))

print('=== GEN-0 strategy distribution (n=15) ===')
for cls, c in classes.most_common():
    print(f'  {cls:20s} {c:3d}  ({100*c/15:.1f}%)')

print()
print('=== Per-agent predicted cooperation ===')
mean_coop = sum(c for _, _, c, _ in coops) / len(coops)
for aid, cls, cr, snippet in coops:
    print(f'  Agent {aid:2d}: {cls:18s} pred_coop={cr:.2f}  | {snippet[:60]}')
print(f'\n  Mean predicted cooperation: {mean_coop:.3f}')
print(f'  (Compare to trajectory[0] observed mean: 0.49-0.60 across conditions)')
