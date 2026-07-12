"""Analyze reasoning content for evidence of leading-eight thinking (or absence)."""
import json, re
from pathlib import Path
from collections import Counter
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp10_reasoning_trace')

# Keywords relevant to leading-eight / standing / recipient_reputation
LE8_KEYWORDS = {
    'recipient_reputation': r'recipient_reputation|recipient.{0,5}reputation|recipient_rep',
    'standing': r'\bstanding\b',
    'judging': r'\bjudging\b|\bjudge\b',
    'simple_standing': r'simple.?standing|standing.{0,10}norm',
    'image_scoring': r'image.?scoring|image.{0,3}score',
    'good_rep_look': r'good.{0,5}(reputation|rep|standing)|good.{0,5}agent',
    'bad_rep_look': r'bad.{0,5}(reputation|rep|standing)|bad.{0,5}agent',
    'four_case': r'four.?case|four.{0,10}case|8.{0,3}case|all.{0,5}case',
    'donor_rep': r'donor.{0,5}reputation|donor.{0,5}rep',
    'third_order': r'third.?order|3rd.?order|higher.?order',
    'leading_eight': r'leading.?eight|leading.?8|8.{0,3}norm|8.{0,3}social',
}

# Per-trial aggregate
all_log = []
for f in sorted(RES.rglob('reasoning_*.json')):
    log = json.loads(f.read_text(encoding='utf-8', errors='ignore'))
    for e in log:
        e['trial'] = str(f.relative_to(RES.parent.parent))
        all_log.append(e)

print(f'Total reasoning entries: {len(all_log)}')

# Per-keyword counts
print('\n=== Keyword mention frequency ===')
for label, pattern in LE8_KEYWORDS.items():
    cnt = sum(1 for e in all_log if re.search(pattern, (e.get('reasoning') or '').lower()))
    print(f'  {label:>20s}: {cnt} / {len(all_log)} ({100*cnt/len(all_log):.1f}%)')

# Per-keyword per-prompt_kind
print('\n=== Per keyword, split by prompt_kind (init vs mutation) ===')
for label, pattern in LE8_KEYWORDS.items():
    init_cnt = sum(1 for e in all_log if e.get('prompt_kind') == 'init' and re.search(pattern, (e.get('reasoning') or '').lower()))
    mut_cnt = sum(1 for e in all_log if e.get('prompt_kind') == 'mutation' and re.search(pattern, (e.get('reasoning') or '').lower()))
    print(f'  {label:>20s}: init={init_cnt} mut={mut_cnt}')

# How many entries have ZERO recipient_reputation mentions at all?
no_rec = sum(1 for e in all_log if 'recipient_reputation' not in (e.get('reasoning') or '') and 'recipient_rep' not in (e.get('reasoning') or '').lower())
print(f'\nEntries with NO recipient_reputation mention: {no_rec} / {len(all_log)} ({100*no_rec/len(all_log):.1f}%)')

# How many have donor_reputation mention but NOT recipient_reputation?
donor_only = sum(1 for e in all_log if 'donor_reputation' in (e.get('reasoning') or '') and 'recipient_reputation' not in (e.get('reasoning') or ''))
print(f'Entries with donor_reputation but NOT recipient_reputation: {donor_only} / {len(all_log)} ({100*donor_only/len(all_log):.1f}%)')

# Show 1 example reasoning snippet from init that mentions recipient_reputation
print('\n=== Example init reasoning that mentions recipient_reputation ===')
shown = 0
for e in all_log:
    if e.get('prompt_kind') == 'init' and 'recipient_reputation' in (e.get('reasoning') or ''):
        print(f'\n--- {e["trial"]} (reasoning len={len(e["reasoning"])}) ---')
        print(e['reasoning'][:1500])
        shown += 1
        if shown >= 1: break

# Show 1 example mutation reasoning that explicitly considers whether to use recipient_reputation
print('\n\n=== Example mutation reasoning where LLM explicitly decides to use/drop recipient_reputation ===')
shown = 0
for e in all_log:
    if e.get('prompt_kind') == 'mutation' and 'recipient_reputation' in (e.get('reasoning') or ''):
        print(f'\n--- {e["trial"]} (reasoning len={len(e["reasoning"])}) ---')
        print(e['reasoning'][:1500])
        shown += 1
        if shown >= 1: break
