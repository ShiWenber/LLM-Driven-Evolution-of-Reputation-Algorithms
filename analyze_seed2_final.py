"""Deep-dive analysis of seed 2's final generation (gen 29) population."""
import json
import re
from collections import Counter

d = json.load(open("results/quantitative_baseline/LLM_evolution_seed2/evolutionary.json", encoding="utf-8"))
last = d["trajectory"][-1]
pop = last["population"]

# Extract key features from each agent's code
def features(code):
    decide = re.search(r"return\s+opponent_reputation\s*([><=!]+)\s*\(?(-?[0-9.]+)?", code)
    thresh = float(decide.group(2)) if decide and decide.group(2) else None
    op = decide.group(1) if decide else "?"
    # my_reputation presence
    has_my_rep = "my_reputation" in code
    # step sizes
    steps = [float(s) for s in re.findall(r"step\s*=\s*([0-9.]+)", code)]
    # evaluate structure
    has_interp = re.search(r"\(\s*target\s*-\s*\w+_reputation\s*\)", code) is not None
    has_hard_step = re.search(r"donor_reputation\s*[+\-]=\s*[0-9.]+", code) is not None
    return {
        "decide_op": op,
        "decide_thresh": thresh,
        "has_my_rep": has_my_rep,
        "steps": steps,
        "has_interp": has_interp,
        "has_hard_step": has_hard_step,
    }

# Print all 15 agents
print(f"=== seed 2 final gen (gen 29) — {len(pop)} agents ===")
print(f"pop coop mean = {last['cooperation_rate_mean']:.3f}, fitness mean = {last['fitness_mean']:.2f}, max = {last['fitness_max']:.2f}")
print()

# Sort by fitness
pop_sorted = sorted(pop, key=lambda a: -a.get("fitness", -1e9))
print("Top 5 by fitness:")
for i, a in enumerate(pop_sorted[:5]):
    f = features(a["code"])
    print(f"  #{i+1}  fit={a['fitness']:.1f}  coop={a['cooperation_rate']:.2f}  rep={a['self_reputation']:.2f}")
    print(f"        decide: opponent_reputation {f['decide_op']} {f['decide_thresh']}  | my_rep: {f['has_my_rep']}  | interp: {f['has_interp']}  | steps: {f['steps']}")

print()
print("Bottom 5 by fitness:")
for i, a in enumerate(pop_sorted[-5:]):
    f = features(a["code"])
    print(f"  #{i+1}  fit={a['fitness']:.1f}  coop={a['cooperation_rate']:.2f}  rep={a['self_reputation']:.2f}")
    print(f"        decide: opponent_reputation {f['decide_op']} {f['decide_thresh']}  | my_rep: {f['has_my_rep']}  | interp: {f['has_interp']}  | steps: {f['steps']}")

# Diversity stats
print()
print("=== Population diversity ===")
thresh_counter = Counter()
hard_step_count = 0
interp_count = 0
my_rep_count = 0
for a in pop:
    f = features(a["code"])
    key = f"op {f['decide_op']} {f['decide_thresh']}"
    thresh_counter[key] += 1
    if f["has_hard_step"]:
        hard_step_count += 1
    if f["has_interp"]:
        interp_count += 1
    if f["has_my_rep"]:
        my_rep_count += 1
print(f"decide-threshold distribution: {dict(thresh_counter)}")
print(f"agents with hard-step evaluate: {hard_step_count}/{len(pop)}")
print(f"agents with interp-step evaluate: {interp_count}/{len(pop)}")
print(f"agents using my_reputation: {my_rep_count}/{len(pop)}")

# Trajectory context
print()
print("=== Trajectory of gen 15 (peak 1.0) → gen 17 (collapse 0.456) → gen 29 (final 0.767) ===")
for g in [14, 15, 16, 17, 18, 19, 28, 29]:
    e = d["trajectory"][g]
    best = max(e["population"], key=lambda a: a.get("fitness", -1e9))
    f = features(best["code"])
    print(f"gen {g:2d}  coop={e['cooperation_rate_mean']:.3f}  best_fit={best['fitness']:.1f}  decide {f['decide_op']} {f['decide_thresh']}  my_rep={f['has_my_rep']}  steps={f['steps']}")
