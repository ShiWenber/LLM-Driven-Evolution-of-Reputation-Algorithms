"""Dump the actual code of TOP 1 strategy per seed (v3)."""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline")

print("=" * 80)
print("V3 LLM TOP 1 STRATEGY PER SEED (raw code)")
print("=" * 80)

for seed in [0, 1, 2]:
    f = ROOT / f"LLM_evolution_seed{seed}" / "evolutionary.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    sorted_pop = sorted(j["final_population"], key=lambda a: a["fitness"], reverse=True)
    a = sorted_pop[0]
    print(f"\n{'#'*80}")
    print(f"# SEED {seed}  TOP 1  (fit={a['fitness']:.0f}, coop={a['cooperation_rate']:.2f}, self_rep={a['self_reputation']:.3f})")
    print(f"# final coop = {j['trajectory'][-1]['cooperation_rate_mean']}")
    print(f"{'#'*80}")
    print()
    print(a["code"])

# Also dump one example from the BOTTOM of seed 2 (the failed seed)
print(f"\n{'='*80}")
print("BONUS: SEED 2 agent 159 (a 'failed' IS strategy with self_rep=-0.23):")
print("="*80)
f = ROOT / "LLM_evolution_seed2" / "evolutionary.json"
j = json.loads(f.read_text(encoding="utf-8"))
sorted_pop = sorted(j["final_population"], key=lambda a: a["agent_id"])
for a in sorted_pop[:3]:
    if a["self_reputation"] < -0.1:
        print(f"\n[agent {a['agent_id']}, self_rep={a['self_reputation']:.3f}, coop={a['cooperation_rate']:.2f}]")
        print(a["code"])
        break
