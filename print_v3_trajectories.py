"""Print trajectory summaries for v3 LLM seeds."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline")

for seed in [0, 1, 2]:
    f = OUT / f"LLM_evolution_seed{seed}" / "evolutionary.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    print(f"=== seed{seed} trajectory (v3) ===")
    for t in j["trajectory"]:
        gen = t["generation"]
        coop = t["cooperation_rate_mean"]
        # Quick visual bar
        bar = "#" * int(coop * 40)
        print(f"  gen {gen:2d}: coop={coop:.3f}  {bar}")
    print()
