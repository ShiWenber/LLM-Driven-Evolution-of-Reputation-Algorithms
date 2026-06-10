"""Strategy-level analysis of the LLM-evo Standard plan.

Use ALL final_population agents (not just last gen) per trial - 12 trials
x 15 agents = 180 strategies. This is the same sampling the original
strategy_analysis.py used (75% Hybrid etc).

Comparisons:
(A) Per-obs archetype distribution
(B) All-obs aggregate (compare to strategy_analysis.md 75% Hybrid)
(C) Hybrid diversity: which sub-classes of Hybrid dominate
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

EXP1 = Path('results/exp1_method')
EXP2 = Path('results/exp2_threshold')
EXP4 = Path('results/exp4_random_mut')


def classify(code: str) -> str:
    if not code: return 'NoCode'
    has_image = bool(re.search(r"observation\s*\[\s*['\"]action['\"]\s*\]\s*==\s*['\"]donate['\"]", code))
    has_history = bool(re.search(r"my_history", code))
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
    if re.search(r"round_num", code):
        if not has_threshold:
            return 'RoundDependent'
    if has_image:
        return 'ImageScoringUnstructured'
    return 'Other'


# Collect ALL final_population agents from Exp 1 + Exp 2 + Exp 4 (30 trials)
all_agents = []
trial_dirs_to_scan = []
for root in (EXP1, EXP2, EXP4):
    for td in root.iterdir():
        if td.is_dir():
            trial_dirs_to_scan.append(td)
for trial_dir in sorted(trial_dirs_to_scan):
    evo_files = list(trial_dir.glob('evo_*.json'))
    if not evo_files: continue
    d = json.loads(evo_files[0].read_text())
    fp = d.get('final_population', [])
    obs = d.get('config', {}).get('observability', '?')
    for a in fp:
        all_agents.append({
            'obs': obs,
            'trial': trial_dir.name,
            'class': classify(a.get('code', '')),
            'coop': a.get('cooperation_rate', 0),
            'fit': a.get('fitness'),
            'gen': a.get('generation'),
        })

print(f"=== LLM-evo Standard plan, all 180 final-population agents ===\n")
counter = Counter(a['class'] for a in all_agents)
total = len(all_agents)
for cls, n in counter.most_common():
    pct = 100 * n / total
    print(f"  {cls:<25s} {n:3d}  ({pct:.1f}%)")

# By obs
print("\n=== Per-obs distribution ===")
by_obs = defaultdict(list)
for a in all_agents:
    by_obs[a['obs']].append(a['class'])
for obs, classes in sorted(by_obs.items()):
    c = Counter(classes)
    tot = sum(c.values())
    n_hyb = c.get('Hybrid', 0)
    n_allc = c.get('ALLC', 0)
    n_alld = c.get('ALLD', 0)
    n_other = tot - n_hyb - n_allc - n_alld
    print(f"  {obs:<14}  n={tot:3d}  Hybrid:{n_hyb:2d} ({100*n_hyb/tot:5.1f}%)  ALLC:{n_allc:2d} ({100*n_allc/tot:5.1f}%)  ALLD:{n_alld:2d} ({100*n_alld/tot:5.1f}%)  Other:{n_other:2d}")

# high coop subset
print("\n=== High-cooperation agents (coop > 0.05) ===")
high_coop = [a for a in all_agents if a['coop'] > 0.05]
print(f"  total: {len(high_coop)}/{total} ({100*len(high_coop)/total:.1f}%)")
hc_classes = Counter(a['class'] for a in high_coop)
for cls, n in hc_classes.most_common():
    print(f"    {cls:<25s} {n:3d}")
