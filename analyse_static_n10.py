"""Strategy-level analysis of the static n=10 control.

For each of the 30 static trials (3 obs x 10 seeds), extract the
final-population strategies and classify them. Compare to the
LLM-evolved Hybrid-dominant distribution.

Also: trajectory-level drift analysis (gen 0 -> gen 9) to see
which archetypes are most affected by play (no selection, no
mutation).
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

STATIC = Path('results/exp3_static_g10_n10')


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
    if has_round := bool(re.search(r"round_num", code)):
        if not has_threshold:
            return 'RoundDependent'
    if has_image:
        return 'ImageScoringUnstructured'
    return 'Other'


# --- (3) final-pop classification ---
all_agents = []
for trial_dir in sorted(STATIC.iterdir()):
    if not trial_dir.is_dir():
        continue
    evo_files = list(trial_dir.glob('static_control_*.json'))
    if not evo_files:
        continue
    d = json.loads(evo_files[0].read_text())
    fp = d.get('final_population', [])
    if not fp:
        # try trials_summary[0].final_population
        ts = d.get('trials_summary', [{}])[0]
        fp = ts.get('final_population', [])
    for a in fp:
        code = a.get('code', '')
        coop = a.get('cooperation_rate', 0)
        all_agents.append({
            'trial': trial_dir.name,
            'class': classify(code),
            'coop': coop,
            'fit': a.get('fitness'),
        })

# breakdown
counter = Counter(a['class'] for a in all_agents)
total = len(all_agents)
print(f"=== Static n=10 final-population: {total} strategies (30 trials x 15 agents) ===\n")
for cls, n in counter.most_common():
    pct = 100 * n / total
    print(f"  {cls:<25s} {n:3d}  ({pct:.1f}%)")
high_coop = [a for a in all_agents if a['coop'] > 0.05]
print(f"\n  High-coop (>0.05): {len(high_coop)}/{total} ({100*len(high_coop)/total:.1f}%)")

# breakdown by obs
print("\n=== Distribution per observability level ===")
by_obs = defaultdict(list)
for trial_dir in sorted(STATIC.iterdir()):
    if not trial_dir.is_dir():
        continue
    obs = trial_dir.name.rsplit('_seed', 1)[0]
    evo_files = list(trial_dir.glob('static_control_*.json'))
    if not evo_files: continue
    d = json.loads(evo_files[0].read_text())
    fp = d.get('final_population', [])
    if not fp:
        ts = d.get('trials_summary', [{}])[0]
        fp = ts.get('final_population', [])
    for a in fp:
        by_obs[obs].append(classify(a.get('code', '')))
for obs, classes in sorted(by_obs.items()):
    c = Counter(classes)
    tot = sum(c.values())
    line = f"  {obs:<14}  n={tot:3d}  " + "  ".join(f"{k}:{c.get(k,0)}" for k in ['Hybrid','ALLC','ALLD','Other'])
    print(line)
print()
print("Recall LLM-evo (Standard plan, n=3 per obs) was: Hybrid 75%, ALLC 14%, ALLD 2%.")
print("Recall deepseek-coder (Robustness, n=3 per obs) was: Hybrid 97.8%, ALLC 0%, ALLD 0%.")
print()

# --- (4) gen0 vs gen9 archetype drift ---
print("=== Trajectory-level drift: gen-0 vs gen-9 archetype distribution ===\n")
# This requires the initial population - need to find it from the trajectory
# The trajectory only gives mean cooperation per gen, not the actual agent code per gen.
# But we have final_population (gen 9/10). We do NOT have gen-0 individual agent code.
# We can estimate gen-0 archetype distribution from the initial cooperation rate:
# 0.55-0.65 = mix of ALLC + Hybrid + ImageScoring. But this is inferential.
# Let's just present what we have: final-pop only.

# Actually, let me check if there's an init_population field
sample_dir = STATIC / 'private_seed0'
files = list(sample_dir.glob('*.json'))
print(f"Files in sample trial: {[f.name for f in files]}")
for f in files:
    d = json.loads(f.read_text())
    keys = list(d.keys())
    print(f"  {f.name}: top-level keys = {keys}")
    if 'initial_population' in d:
        ip = d['initial_population']
        print(f"    initial_population has {len(ip)} agents")
    ts = d.get('trials_summary', [{}])
    if ts:
        ts0 = ts[0]
        ts0_keys = list(ts0.keys())
        print(f"    trials_summary[0] keys = {ts0_keys}")
