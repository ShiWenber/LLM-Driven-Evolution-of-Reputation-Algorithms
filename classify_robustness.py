"""Classify robustness strategies using the same classifier as strategy_analysis.py."""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

ROBUST = Path('results/exp5_robustness')


def classify(code: str) -> str:
    if not code: return 'NoCode'
    has_image = bool(re.search(r"observation\s*\[\s*['\"]action['\"]\s*\]\s*==\s*['\"]donate['\"]", code))
    has_history = bool(re.search(r"my_history", code))
    has_round = bool(re.search(r"round_num", code))
    has_threshold = bool(re.search(r"recipient_reputation\s*[><=!]+", code))
    has_random = bool(re.search(r"random\.", code))
    if re.search(r"return\s+True\s*$", code.strip(), re.MULTILINE) and not has_threshold and not has_random:
        return 'ALLC'
    if re.search(r"return\s+False\s*$", code.strip(), re.MULTILINE) and not has_threshold and not has_random:
        return 'ALLD'
    if has_random and not has_image and not has_history:
        return 'RandomStrategy'
    if has_image and has_threshold and not has_history:
        return 'ImageScoring'
    if has_image and has_threshold and has_history:
        return 'Hybrid'
    if has_threshold and not has_image and not has_history:
        return 'ThresholdOnly'
    if has_history and not has_threshold:
        return 'DirectExperience'
    if has_round and not has_threshold:
        return 'RoundDependent'
    if has_image:
        return 'ImageScoringUnstructured'
    return 'Other'


# Collect all final-population strategies
all_agents = []
high_coop = []
for trial_dir in sorted(ROBUST.iterdir()):
    if not trial_dir.is_dir():
        continue
    evo_files = list(trial_dir.glob('evo_*.json'))
    if not evo_files:
        continue
    d = json.loads(evo_files[0].read_text())
    fp = d.get('final_population', [])
    for a in fp:
        code = a.get('code', '')
        coop = a.get('cooperation_rate', 0)
        cls = classify(code)
        a2 = dict(a)
        a2['classification'] = cls
        a2['trial'] = trial_dir.name
        all_agents.append(a2)
        if coop > 0.05:
            high_coop.append(a2)

# Distribution
counter = Counter(a['classification'] for a in all_agents)
total = len(all_agents)
print(f"=== Strategy classification across {total} final-population strategies (deepseek-coder) ===\n")
for cls, n in counter.most_common():
    pct = 100 * n / total
    print(f"  {cls:<25s} {n:3d}  ({pct:.1f}%)")

print()
print(f"  Agents with coop > 0.05: {len(high_coop)} / {total} ({100*len(high_coop)/total:.1f}%)")
print()
print("=== Top 8 by cooperation rate ===")
high_coop.sort(key=lambda a: a.get('cooperation_rate', 0), reverse=True)
for a in high_coop[:8]:
    print(f"  {a['trial']:<25s} agent={a.get('agent_id'):<3} coop={a.get('cooperation_rate'):.3f} fit={a.get('fitness'):.1f} class={a['classification']}")
