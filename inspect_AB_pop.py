"""Inspect the A/B trial's final population: show all 15 strategies, classify them,
and report which 'family' they belong to."""
import json, re

d = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'
with open(d + '/evo_full_deepseek-v4-flash_20260616_115153.json') as f:
    t = json.load(f)
fp = t['final_population']
print(f'final_pop: {len(fp)} agents\n')

for i, a in enumerate(fp):
    code = a.get('code', '')
    aid = a.get('agent_id')
    has_action = "'A'" in code or '"A"' in code
    has_B = "'B'" in code or '"B"' in code
    has_recipient_rep = 'recipient_reputation' in code.split('def decide')[0]
    has_my_history = 'my_history' in code
    has_round = 'round_num' in code
    has_random = 'random' in code and 'return' in code
    has_return_true = re.search(r'return\s+True\b', code) and 'recipient_reputation' not in code
    has_return_false = re.search(r'return\s+False\b', code) and 'recipient_reputation' not in code
    has_threshold = re.search(r'recipient_reputation\s*[><=!]+\s*[-\d.]+', code) is not None
    has_decay = re.search(r'0\.\d+\s*\*\s*current', code) or 'decay' in code.lower()
    has_ema = 'sum' in code and 'my_history' in code
    has_conditional = re.search(r'if\s+.*recipient_reputation', code) is not None
    n_if = len(re.findall(r'\bif\b', code))

    # Print a one-line summary
    summary = f'Agent {aid:2d} (len={len(code):4d}): '
    if has_return_true:
        summary += 'always-True-decide '
    elif has_return_false:
        summary += 'always-False-decide '
    elif has_threshold:
        summary += f'threshold-decide '
    summary += f'  [A={has_action}, B={has_B}, rep={has_recipient_rep}, my_hist={has_my_history}, round={has_round}, rand={has_random}, decay={has_decay}, ema={has_ema}, cond={has_conditional}, n_if={n_if}]'
    print(summary)
