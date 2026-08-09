"""Dump one representative strategy (highest-fitness) per generation for LLM seed 2."""
import json

d = json.load(open("results/quantitative_baseline/LLM_evolution_seed2/evolutionary.json", encoding="utf-8"))
print(f"gens: {len(d['trajectory'])}")
print(f"schema_version: {d.get('config', {}).get('schema_version', 1)}")
print()

for entry in d["trajectory"]:
    g = entry["generation"]
    coop = entry["cooperation_rate_mean"]
    pop = entry.get("population", [])
    if not pop:
        print(f"gen {g}: coop={coop:.3f} (no population data)")
        continue
    best = max(pop, key=lambda a: a.get("fitness", -1e9))
    code = best.get("code", "").strip()
    fit = best.get("fitness", 0)
    rep = best.get("self_reputation", 0)
    own_coop = best.get("cooperation_rate", 0)
    aid = best.get("agent_id", "?")
    print(f"--- gen {g}  coop(pop)={coop:.3f}  best(agent {aid}, fit={fit:.2f}, own_coop={own_coop:.2f}, self_rep={rep:.2f}) ---")
    print(code)
    print()
