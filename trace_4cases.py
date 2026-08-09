"""Look at 4 representative cases."""
import json, re
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results')

cases = [
    ('exp1_method_n10/partial_0.7_seed4/evo_partial_0.7_deepseek-v4-flash_20260610_090723.json', '6233d49d3c11', 'INTRODUCED in #2 (target coop=0)'),
    ('exp5_robustness/partial_0.7_seed2/evo_partial_0.7_deepseek-coder_20260608_231307.json', '8859a5bad443', 'INTRODUCED in #5 (target coop=0)'),
    ('exp1_method_n10/partial_0.7_seed3/evo_partial_0.7_deepseek-v4-flash_20260610_080750.json', 'c3e816a0b856', '#1 SUCCESS 0.53 - parent died'),
    ('exp5_robustness/partial_0.7_seed0/evo_partial_0.7_deepseek-coder_20260608_225137.json', '063f944a0fbc', '#3 SUCCESS 0.57 - parent died'),
]
for path, sid, label in cases:
    td = RES / path
    d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
    fp = d.get('final_population') or []
    target = next((a for a in fp if a.get('strategy_id') == sid), None)
    if not target: continue
    print('\n' + '='*78)
    print('CASE:', label)
    print('Path:', path)
    print('target sid=', sid, 'gen=', target.get('generation'), 'coop=', target.get('cooperation_rate'))
    print('PARENT_ID:', target.get('parent_id'), '(NOT IN final_population - parent died in earlier gen)')
    print('TARGET CODE:')
    print(target.get('code'))
