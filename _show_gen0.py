"""Show gen 0 + gen 99 coop/fitness for 3-seed 1000-inter run."""
import json
import os

base = r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline"

print(f"{'seed':<6} {'gen0_coop':<10} {'gen0_fit':<10} {'gen99_coop':<10} {'gen99_fit':<10}")
print("-" * 50)
for s in [0, 1, 2]:
    p = os.path.join(base, f"LLM_v3_g100_1000inter_seed{s}", "evolutionary.json")
    j = json.load(open(p))
    g0 = j["trajectory"][0]
    gF = j["trajectory"][-1]
    print(f"{s:<6} {g0['cooperation_rate_mean']:<10.3f} {g0['fitness_mean']:<10.2f} {gF['cooperation_rate_mean']:<10.3f} {gF['fitness_mean']:<10.2f}")

# Also show strategy composition at gen 0
print()
print("=== Gen 0 strategy snapshots ===")
for s in [0, 1, 2]:
    p = os.path.join(base, f"LLM_v3_g100_1000inter_seed{s}", "evolutionary.json")
    j = json.load(open(p))
    g0 = j["trajectory"][0]
    pops = g0.get("population", [])
    if not pops:
        print(f"seed={s}: no population data in gen 0")
        continue
    print(f"\nseed={s} gen 0: {len(pops)} agents")
    # Show fitness distribution
    fits = [a.get("fitness", 0) for a in pops]
    coops = [a.get("history", []) for a in pops]
    if coops and coops[0]:
        total_games = sum(len(h) for h in coops)
        total_coop = sum(sum(1 for x in h if x) for h in coops)
        print(f"  per-agent fitness: min={min(fits):.1f} max={max(fits):.1f} mean={sum(fits)/len(fits):.2f}")
        print(f"  total coop actions / total games = {total_coop}/{total_games} = {total_coop/total_games:.3f}")
