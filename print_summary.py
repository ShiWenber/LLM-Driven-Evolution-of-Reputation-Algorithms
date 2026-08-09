"""Print v2 baseline summary for verification."""
import json
from pathlib import Path

OUT = Path("results/quantitative_baseline")
print("=== Per-trial summary ===")
for d in sorted(OUT.iterdir()):
    if not d.is_dir():
        continue
    j = d / "evolutionary.json"
    if not j.exists():
        print(f"{d.name}: no data")
        continue
    data = json.loads(j.read_text(encoding="utf-8"))
    traj = data.get("trajectory", [])
    if not traj:
        print(f"{d.name}: empty trajectory")
        continue
    final = traj[-1]["cooperation_rate_mean"]
    mean = sum(x["cooperation_rate_mean"] for x in traj) / len(traj)
    std = (sum((x["cooperation_rate_mean"] - mean) ** 2 for x in traj) / len(traj)) ** 0.5
    elapsed = data.get("elapsed_sec", 0) / 60
    print(f"{d.name}: n_gens={len(traj):2d}  final={final:.3f}  mean={mean:.3f}  std={std:.3f}  time={elapsed:.1f}min")
