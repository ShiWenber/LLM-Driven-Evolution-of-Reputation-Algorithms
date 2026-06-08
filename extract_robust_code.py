"""Extract representative Hybrid code from deepseek-coder final populations."""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ROBUST = Path('results/exp5_robustness')


def classify(code):
    if not code: return 'NoCode'
    has_image = bool(re.search(r"observation\s*\[\s*['\"]action['\"]\s*\]\s*==\s*['\"]donate['\"]", code))
    has_history = bool(re.search(r"my_history", code))
    has_threshold = bool(re.search(r"recipient_reputation\s*[><=!]+", code))
    has_random = bool(re.search(r"random\.", code))
    if has_image and has_threshold and has_history:
        return 'Hybrid'
    return 'Other'


# Top 3 by coop, Hybrid only, distinct code
seen = set()
shown = 0
for trial_dir in sorted(ROBUST.iterdir()):
    if not trial_dir.is_dir():
        continue
    evo_files = list(trial_dir.glob('evo_*.json'))
    if not evo_files:
        continue
    d = json.loads(evo_files[0].read_text())
    fp = d.get('final_population', [])
    fp.sort(key=lambda a: a.get('cooperation_rate', 0), reverse=True)
    for a in fp:
        if a.get('cooperation_rate', 0) < 0.5:
            continue
        if classify(a.get('code', '')) != 'Hybrid':
            continue
        code = a.get('code', '')
        if code in seen:
            continue
        seen.add(code)
        print(f"### {trial_dir.name} agent={a.get('agent_id')} coop={a.get('cooperation_rate'):.3f} fit={a.get('fitness'):.1f}\n")
        print("```python")
        print(code)
        print("```\n")
        shown += 1
        if shown >= 4:
            break
    if shown >= 4:
        break
