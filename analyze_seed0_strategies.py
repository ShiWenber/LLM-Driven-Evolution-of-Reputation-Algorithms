"""Inspect the final strategies evolved in LLM seed 0 (v2 quantitative baseline)."""
import json
from pathlib import Path

D = json.loads(Path("results/quantitative_baseline/LLM_evolution_seed0/evolutionary.json").read_text(encoding="utf-8"))
fp = D["final_population"]
traj = D["trajectory"]
print(f"Final cooperation: {traj[-1]['cooperation_rate_mean']:.3f}")
print(f"Final population size: {len(fp)}")
print()
print("=== Per-agent summary ===")
for i, a in enumerate(fp):
    code = a["code"].strip()
    short = code[:300].replace("\n", " | ")
    print(f"agent {i:2d}  coop={a['cooperation_rate']:.3f}  self_rep={a['self_reputation']:.3f}  fit={a['fitness']:.1f}")
    print(f"         code ({len(code)} chars): {short}")
    print()
