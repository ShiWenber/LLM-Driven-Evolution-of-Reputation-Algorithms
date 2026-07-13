"""Find and show interesting reasoning entries from v23 (6 trials, 336 entries).
Pick:
  1. An init-population generation where LLM thinks about reputation design
  2. A mutation where LLM explicitly decides to use/drop recipient_reputation
  3. A mutation where LLM considers 'image scoring' or 'standing'
"""
import json
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp10_reasoning_trace')

all_log = []
for f in sorted(RES.rglob('reasoning_*.json')):
    log = json.loads(f.read_text(encoding='utf-8', errors='ignore'))
    for e in log:
        e['trial'] = str(f.relative_to(RES.parent.parent))
        all_log.append(e)

# 1. Find an init-population entry that discusses image scoring or standing
print('=' * 78)
print('EXAMPLE 1: INIT-POPULATION generation where LLM mentions "image scoring"')
print('=' * 78)
for e in all_log:
    if e.get('prompt_kind') != 'init': continue
    r = (e.get('reasoning') or '').lower()
    if 'image scoring' in r or 'standing' in r or 'reputation' in r:
        print(f'\n--- {e["trial"]} ---')
        print(f'  reasoning length: {len(e["reasoning"])}')
        print(f'  code length: {len(e.get("content") or "")}')
        print('  REASONING:')
        print(e['reasoning'])
        print('  CODE:')
        print(e.get('content'))
        break

# 2. Find a mutation entry where LLM thinks about recipient_reputation design
print('\n\n' + '=' * 78)
print('EXAMPLE 2: MUTATION where LLM designs recipient_reputation logic')
print('=' * 78)
# Save to file rather than print to avoid truncation
output_lines = []
for e in all_log:
    if e.get('prompt_kind') != 'mutation': continue
    r = (e.get('reasoning') or '')
    if 'recipient_reputation' in r and 'based on' in r.lower() and 'action' in r.lower():
        if len(r) > 3000 and len(r) < 7000:
            output_lines.append(f'\n--- {e["trial"]} ---')
            output_lines.append(f'  reasoning length: {len(r)}')
            output_lines.append(f'  code length: {len(e.get("content") or "")}')
            output_lines.append('  REASONING:')
            output_lines.append(e['reasoning'])
            output_lines.append('  CODE:')
            output_lines.append(e.get('content', ''))
            break

with open('reasoning_example2.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))
print(f'Wrote reasoning_example2.txt ({sum(len(s) for s in output_lines)} chars)')
print('First 800 chars:')
print(''.join(output_lines)[:800])

# 3. Find a mutation where LLM considers 'good standing' as concept
print('\n\n' + '=' * 78)
print('EXAMPLE 3: MUTATION where LLM uses "good standing" as concept')
print('=' * 78)
shown = 0
for e in all_log:
    if e.get('prompt_kind') != 'mutation': continue
    r = (e.get('reasoning') or '')
    if 'good standing' in r.lower() or 'bad standing' in r.lower():
        print(f'\n--- {e["trial"]} ---')
        print(f'  reasoning length: {len(r)}')
        print(f'  code length: {len(e.get("content") or "")}')
        print('  REASONING (excerpt around "standing"):')
        # find context
        idx = r.lower().find('standing') - 300
        if idx < 0: idx = 0
        print('  ...' + r[max(0,idx):idx+1500] + '...')
        print('  CODE:')
        print(e.get('content'))
        shown += 1
        if shown >= 1: break
