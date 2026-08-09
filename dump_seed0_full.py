"""Dump full code of LLM seed0 final strategies."""
import json
from pathlib import Path

D = json.loads(Path("results/quantitative_baseline/LLM_evolution_seed0/evolutionary.json").read_text(encoding="utf-8"))
fp = D["final_population"]

# Show the highest-fitness agent's full code as a representative
fp_sorted = sorted(fp, key=lambda a: -a["fitness"])
for a in fp_sorted[:3]:
    print(f"=== agent (fitness={a['fitness']}, coop={a['cooperation_rate']}, self_rep={a['self_reputation']:.3f}) ===")
    print(a["code"])
    print()
