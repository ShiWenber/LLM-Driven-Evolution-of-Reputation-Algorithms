"""Find mutation reasoning entries that show LLM designing:
  - EMA decay (exponential moving average)
  - Asymmetric delta (different reward vs penalty)
  - Threshold / time-decay
Print full reasoning + code for each.
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

# 1. EMA decay example
print('=' * 78)
print('EXAMPLE A: MUTATION where LLM designs EMA decay')
print('=' * 78)
output_a = []
for e in all_log:
    if e.get('prompt_kind') != 'mutation': continue
    r = (e.get('reasoning') or '').lower()
    code = e.get('content') or ''
    # Look for "exponential" / "EMA" / "moving average" / "decay" + "alpha" in reasoning
    if ('exponential' in r or 'moving average' in r or 'ema' in r) and 'alpha' in r:
        # Check that the produced code has alpha-based decay
        if 'alpha' in code.lower() and ('* ' in code or '* ' in code):
            output_a.append(f'\n--- {e["trial"]} ---')
            output_a.append(f'  reasoning length: {len(e["reasoning"])}')
            output_a.append(f'  code length: {len(code)}')
            output_a.append('  REASONING:')
            output_a.append(e['reasoning'])
            output_a.append('  CODE:')
            output_a.append(code)
            break

if output_a:
    with open('reasoning_example_ema.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_a))
    print(f'Wrote reasoning_example_ema.txt ({sum(len(s) for s in output_a)} chars)')
    print('First 600 chars:')
    print(''.join(output_a)[:600])
else:
    print('No EMA example found; will search by alpha')

# 2. Asymmetric delta example
print('\n\n' + '=' * 78)
print('EXAMPLE B: MUTATION where LLM designs asymmetric delta')
print('=' * 78)
output_b = []
for e in all_log:
    if e.get('prompt_kind') != 'mutation': continue
    r = (e.get('reasoning') or '').lower()
    code = e.get('content') or ''
    if ('asymmetric' in r or ('cooperate' in r and 'defect' in r and ('reward' in r or 'punish' in r))):
        # check if code has different magnitude for cooperate vs defect
        import re
        m_coop = re.search(r"['\"]cooperate['\"].*?[+]\s*([\d.]+)", code, re.S)
        m_def = re.search(r"['\"]defect['\"].*?[-=]\s*([\d.]+)", code, re.S)
        if m_coop and m_def:
            coop_val = float(m_coop.group(1))
            def_val = float(m_def.group(1))
            if abs(coop_val) != abs(def_val):  # asymmetric
                output_b.append(f'\n--- {e["trial"]} ---')
                output_b.append(f'  reasoning length: {len(e["reasoning"])}')
                output_b.append(f'  code length: {len(code)}')
                output_b.append('  REASONING:')
                output_b.append(e['reasoning'])
                output_b.append('  CODE:')
                output_b.append(code)
                break

if output_b:
    with open('reasoning_example_asym.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_b))
    print(f'Wrote reasoning_example_asym.txt ({sum(len(s) for s in output_b)} chars)')
    print('First 600 chars:')
    print(''.join(output_b)[:600])
else:
    print('No asymmetric example found')

# 3. Time-decay threshold
print('\n\n' + '=' * 78)
print('EXAMPLE C: MUTATION where LLM designs time-decay threshold')
print('=' * 78)
output_c = []
for e in all_log:
    if e.get('prompt_kind') != 'mutation': continue
    r = (e.get('reasoning') or '').lower()
    code = e.get('content') or ''
    if 'round_num' in r and ('threshold' in r or 'time' in r) and ('decay' in r or 'increase' in r or 'decrease' in r):
        if 'round_num' in code and ('threshold' in code.lower() or '0.01' in code or '0.02' in code):
            output_c.append(f'\n--- {e["trial"]} ---')
            output_c.append(f'  reasoning length: {len(e["reasoning"])}')
            output_c.append(f'  code length: {len(code)}')
            output_c.append('  REASONING:')
            output_c.append(e['reasoning'])
            output_c.append('  CODE:')
            output_c.append(code)
            break

if output_c:
    with open('reasoning_example_time.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_c))
    print(f'Wrote reasoning_example_time.txt ({sum(len(s) for s in output_c)} chars)')
    print('First 600 chars:')
    print(''.join(output_c)[:600])
else:
    print('No time-decay example found')
