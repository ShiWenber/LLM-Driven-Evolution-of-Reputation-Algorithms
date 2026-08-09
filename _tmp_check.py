import json
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp11_thinking_compare')
d = json.loads((RES / 'evo_private_deepseek-v4-flash_20260713_170331.json').read_text(encoding='utf-8', errors='ignore'))
fp = d.get('final_population', [])
for a in fp:
    code = a.get('code', '')
    if 'recipient_reputation' in code:
        # find context
        idx = code.find('recipient_reputation')
        print(f'agent {a["agent_id"]}: ...{code[max(0,idx-50):idx+200]}...')
        print('---')
        break
