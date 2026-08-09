"""Dump several concrete seed 2 final-gen strategies."""
import json

d = json.load(open("results/quantitative_baseline/LLM_evolution_seed2/evolutionary.json", encoding="utf-8"))
last = d["trajectory"][-1]
pop = last["population"]

# Sort: best by fitness, then by cooperation (high to low)
pop_sorted = sorted(pop, key=lambda a: (-a.get("fitness", -1e9), -a.get("cooperation_rate", -1)))

# Pick a diverse set: top 1, top 2, mid, lowest-fit, a cooperator near 0
picks = []
picks.append(("TOP 1 (best fitness)", pop_sorted[0]))
picks.append(("TOP 2", pop_sorted[1]))
# Find mid (around fitness mean)
fits = [a.get("fitness", 0) for a in pop]
mid_fit = sorted(fits)[len(fits) // 2]
for a in pop:
    if abs(a.get("fitness", 0) - mid_fit) < 2:
        picks.append((f"MID (fit~{mid_fit})", a))
        break
# Lowest
picks.append(("LOWEST fitness", pop_sorted[-1]))
# Defector (low coop, moderate fitness)
defectors = [a for a in pop if a.get("cooperation_rate", 1) < 0.3]
if defectors:
    picks.append(("DEFECTOR (low coop)", max(defectors, key=lambda a: a.get("fitness", -1e9))))

for tag, a in picks:
    print("=" * 70)
    print(f"  {tag}")
    print(f"  fitness={a['fitness']:.1f}  coop={a['cooperation_rate']:.2f}  self_rep={a['self_reputation']:.2f}")
    print("=" * 70)
    print(a["code"])
    print()
