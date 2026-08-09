import json
with open(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\gen0_with_reputation_fields\gen0_population.json') as f:
    d = json.load(f)
for s in d['strategies']:
    code = s['code']
    if 'recipient_reputation' in code and 'def decide' in code:
        eval_part = code.split('def decide')[0]
        aid = s['agent_id']
        print(f'=== Agent {aid} (full evaluate()) ===')
        print(eval_part)
        print()
