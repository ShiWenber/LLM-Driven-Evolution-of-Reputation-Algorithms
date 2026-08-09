import json
from pathlib import Path
p = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline\LLM_evolution_seed0\evolutionary.json")
r = json.loads(p.read_text(encoding="utf-8"))
print("top-level keys:", list(r.keys()))
print()
print("trajectory[0] keys:", list(r["trajectory"][0].keys()))
print()
print("trajectory[0]:", r["trajectory"][0])
