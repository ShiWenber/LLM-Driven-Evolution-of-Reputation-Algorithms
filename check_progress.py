"""Check which G=30 trials are complete."""
import os, json
from pathlib import Path
OUT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp12_g30_n15")
for d in sorted(OUT.iterdir()):
    if not d.is_dir() or d.name == "logs":
        continue
    evo = d / "evolutionary.json"
    if evo.exists():
        try:
            data = json.loads(evo.read_text(encoding="utf-8"))
            n = len(data.get("trajectory", []))
            final = data["trajectory"][-1].get("cooperation_rate_mean", "?") if n else "?"
            print(f"{d.name}: {n} gens, final coop={final:.3f}" if isinstance(final, float) else f"{d.name}: {n} gens, final coop={final}")
        except Exception as e:
            print(f"{d.name}: ERROR {e}")
    else:
        print(f"{d.name}: NO FILE")
