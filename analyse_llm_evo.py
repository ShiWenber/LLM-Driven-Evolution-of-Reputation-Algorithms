"""Strategy-level analysis of the LLM-evo Standard plan trajectories.

Two angles:

(A) Generation-0 archetype distribution: what does the LLM propose
    *before* any selection or mutation acts? This is the LLM's raw
    prior over strategies.

(B) Generation-trajectory archetype distribution: which archetypes
    survive / die across the 10 generations? This shows what selection
    + LLM-mutation does to the initial mix.

Compare to the static control's no-selection, no-mutation behaviour
(approx 0.52-0.59 mean cooperation across 10 generations of play only).
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

EXP1 = Path('results/exp1_method')


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


# --- (A) gen 0 distribution ---
print("=" * 70)
print("(A) Generation-0 initial population: LLM's raw prior")
print("=" * 70)

gen0_class_count = Counter()
gen0_total = 0
for trial_dir in sorted(EXP1.iterdir()):
    if not trial_dir.is_dir(): continue
    evo_files = list(trial_dir.glob('evo_*.json'))
    if not evo_files: continue
    d = json.loads(evo_files[0].read_text())
    fp = d.get('final_population', [])
    for a in fp:
        if a.get('generation') == 0:
            gen0_class_count[classify(a.get('code', ''))] += 1
            gen0_total += 1

print(f"\n  Total gen-0 strategies across {len(list(EXP1.iterdir()))} trials: {gen0_total}")
for cls, n in gen0_class_count.most_common():
    pct = 100 * n / gen0_total
    print(f"    {cls:<25s} {n:3d}  ({pct:.1f}%)")

# --- (B) final-pop (gen 9/10) vs gen 0 archetype distribution ---
print()
print("=" * 70)
print("(B) Archetype distribution by generation (last generation only)")
print("=" * 70)

# gen-9 or final-generation
final_class_count = Counter()
final_total = 0
for trial_dir in sorted(EXP1.iterdir()):
    if not trial_dir.is_dir(): continue
    evo_files = list(trial_dir.glob('evo_*.json'))
    if not evo_files: continue
    d = json.loads(evo_files[0].read_text())
    fp = d.get('final_population', [])
    n_gens = max((a.get('generation', 0) or 0) for a in fp) if fp else 0
    for a in fp:
        if a.get('generation') == n_gens:
            final_class_count[classify(a.get('code', ''))] += 1
            final_total += 1

print(f"\n  Total final-pop strategies: {final_total}")
for cls, n in final_class_count.most_common():
    pct = 100 * n / final_total
    print(f"    {cls:<25s} {n:3d}  ({pct:.1f}%)")

# Compare
print()
print("=" * 70)
print("Comparison: gen-0 (LLM's prior) -> final (after 10 gens of LLM-mut + tournament)")
print("=" * 70)
classes = sorted(set(gen0_class_count) | set(final_class_count), key=lambda c: -gen0_class_count.get(c, 0))
print(f"\n  {'Class':<25s}  {'gen-0 %':>8s}  {'final %':>8s}  {'Δ':>8s}")
for cls in classes:
    g0 = gen0_class_count.get(cls, 0) * 100 / gen0_total
    fn = final_class_count.get(cls, 0) * 100 / final_total
    print(f"  {cls:<25s}  {g0:7.1f}%  {fn:7.1f}%  {fn-g0:+7.1f}%")

# --- (C) Per-obs-level final-pop distribution (using Exp 1 data) ---
print()
print("=" * 70)
print("(C) Final-pop archetype distribution by observability level (LLM-evo)")
print("=" * 70)
import re
by_obs = defaultdict(list)
for trial_dir in sorted(EXP1.iterdir()):
    if not trial_dir.is_dir(): continue
    m = re.match(r'([a-z_0-9.]+)_seed\d+', trial_dir.name)
    if not m: continue
    obs = m.group(1)
    evo_files = list(trial_dir.glob('evo_*.json'))
    if not evo_files: continue
    d = json.loads(evo_files[0].read_text())
    fp = d.get('final_population', [])
    n_gens = max((a.get('generation', 0) or 0) for a in fp) if fp else 0
    for a in fp:
        if a.get('generation') == n_gens:
            by_obs[obs].append(classify(a.get('code', '')))
for obs, classes in sorted(by_obs.items()):
    c = Counter(classes)
    tot = sum(c.values())
    n_hyb = c.get('Hybrid', 0)
    n_allc = c.get('ALLC', 0)
    n_alld = c.get('ALLD', 0)
    print(f"  {obs:<14}  n={tot:3d}  Hybrid:{n_hyb:2d} ({100*n_hyb/tot:.0f}%)  ALLC:{n_allc:2d} ({100*n_allc/tot:.0f}%)  ALLD:{n_alld:2d} ({100*n_alld/tot:.0f}%)")
