import json, os, re

d = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'
with open(os.path.join(d, 'evo_full_deepseek-v4-flash_20260616_105442.json')) as f:
    t = json.load(f)
print('keys:', list(t.keys()))
fp = t['final_population']
print(f'final_pop: {len(fp)} agents')

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

# Try to detect leading-eight style in evaluate()
def detect_norm(code):
    """Detect which leading-eight norm this looks like."""
    eval_part = code.split('def decide')[0] if 'def decide' in code else code
    has_donor_rep = 'donor_reputation' in eval_part
    has_recipient_rep = 'recipient_reputation' in eval_part
    has_action = 'observation["action"]' in eval_part or "observation['action']" in eval_part
    if not has_action and not has_donor_rep and not has_recipient_rep:
        return 'N/A (no learning)'
    if not has_recipient_rep and not has_donor_rep:
        return 'IS-like (action only)'
    # Check for recipient-conditional logic
    if has_recipient_rep:
        # Look for asymmetric handling of donate vs not_donate
        has_donate_block = '"donate"' in eval_part
        has_notdonate_block = '"not_donate"' in eval_part
        has_conditional = re.search(r'if.*recipient_reputation', eval_part) is not None
        if has_conditional and has_donate_block and has_notdonate_block:
            return 'Leading-8 (recipient-conditional)'
    return 'Augmented IS (uses rep field, but standard)'

from collections import Counter
classes = Counter()
norms = Counter()
for a in fp:
    code = a.get('code', '')
    cls = classify(code)
    nrm = detect_norm(code)
    classes[cls] += 1
    norms[nrm] += 1

print()
print('=== Final-pop classifier (9-archetype) ===')
for cls, c in classes.most_common():
    print(f'  {cls:20s} {c:3d}  ({100*c/len(fp):.1f}%)')
print()
print('=== Leading-eight style ===')
for nrm, c in norms.most_common():
    print(f'  {nrm:50s} {c:3d}  ({100*c/len(fp):.1f}%)')

# Show 1-2 leading-8 strategies if any
print()
print('=== Sample leading-8 strategies ===')
shown = 0
for a in fp:
    nrm = detect_norm(a.get('code', ''))
    if 'Leading-8' in nrm:
        aid = a.get('agent_id')
        print(f'\n--- Agent {aid} (length={len(a["code"])}) ---')
        print(a['code'][:1000])
        shown += 1
        if shown >= 2:
            break

if shown == 0:
    print('No full leading-8 strategies in final pop. Showing the most recipient-conditional ones:')
    candidates = []
    for a in fp:
        nrm = detect_norm(a.get('code', ''))
        if 'recipient' in nrm.lower() or 'Augmented' in nrm:
            candidates.append((a, nrm))
    for a, nrm in candidates[:2]:
        aid = a.get('agent_id')
        print(f'\n--- Agent {aid} (length={len(a["code"])}, nrm={nrm}) ---')
        print(a['code'][:1000])
