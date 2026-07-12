"""Find mutation reasoning entries where LLM explicitly discusses recipient_reputation
in design decisions (rather than just mentioning it)."""
import json, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp10_reasoning_trace')

# Look for reasoning that USES recipient_reputation in a design rationale
# Pattern: "recipient_reputation" within ~200 chars of "use|using|consider|because|since|so" or "weight|threshold|score"
all_log = []
for f in sorted(RES.rglob('reasoning_*.json')):
    log = json.loads(f.read_text(encoding='utf-8', errors='ignore'))
    for e in log:
        e['trial'] = str(f.relative_to(RES.parent.parent))
        all_log.append(e)

# Find mutation entries where the reasoning discusses how to use recipient_reputation
# in a design context
shown = 0
for e in all_log:
    if e.get('prompt_kind') != 'mutation': continue
    r = (e.get('reasoning') or '').lower()
    if 'recipient_reputation' not in r: continue
    # heuristic: find sentences containing "recipient_reputation" and any of
    # use/using/consider/because/so that
    sentences = re.split(r'[.\n]', (e.get('reasoning') or ''))
    design_snippets = []
    for s in sentences:
        s_low = s.lower()
        if 'recipient_reputation' in s_low and any(w in s_low for w in ['use', 'using', 'consider', 'because', 'since', 'so that', 'based on', 'depending', 'if', 'when', 'weight', 'threshold', 'score', 'scale', 'factor', 'modulate', 'adjust', 'account']):
            design_snippets.append(s.strip())
    if not design_snippets: continue
    print('='*78)
    print(f'TRIAL: {e["trial"]}')
    print(f'  reasoning length: {len(e["reasoning"])}')
    print(f'  code length: {len(e.get("content") or "")}')
    for s in design_snippets[:5]:
        print(f'  • {s[:300]}')
    shown += 1
    if shown >= 3: break

# Also look for "leading eight" / "standing" / "judging" mentions in full reasoning
print('\n\n=== Entries mentioning standing / judging / image scoring ===')
for e in all_log:
    r = (e.get('reasoning') or '').lower()
    for kw in ['standing', 'judging', 'image scoring', 'leading eight', 'good standing', 'bad standing']:
        if kw in r:
            # show 200 chars of context around it
            idx = r.find(kw)
            ctx = (e.get('reasoning') or '')[max(0, idx-100):idx+200]
            print(f'\n[TRIAL: {e["trial"]}] keyword: {kw}')
            print(f'  context: {ctx[:300]}')
            break
