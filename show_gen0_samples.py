import json
with open(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\gen0_inspection\gen0_population.json') as f:
    d = json.load(f)

# Find first ALLC, first ALLD, first big Hybrid
def short_label(code):
    if 'return True' in code and 'recipient_reputation' not in code:
        return 'ALLC'
    if 'return False' in code and 'recipient_reputation' not in code:
        return 'ALLD'
    if 'observation' in code and 'recipient_reputation' in code and 'my_history' in code:
        return 'Hybrid'
    if 'recipient_reputation' in code:
        return 'ThresholdOnly'
    return 'Other'

for s in d['strategies']:
    code = s['code']
    aid = s['agent_id']
    label = short_label(code)
    print(f'=== Agent {aid}: {label} (len={len(code)}) ===')
    print(code[:800])
    print()
