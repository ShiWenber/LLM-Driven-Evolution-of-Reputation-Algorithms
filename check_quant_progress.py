"""Check progress of v2 quantitative experiment."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline")
print(f"Output dir: {OUT}")
print()
print(f"{'name':<25} {'seed':<5} {'n_gens':<7} {'final_coop':<12} {'mean_coop':<10}")
print("-" * 70)
for d in sorted(OUT.iterdir()):
    if not d.is_dir():
        continue
    evo = d / "evolutionary.json"
    if not evo.exists():
        print(f"{d.name:<25} - no data")
        continue
    try:
        data = json.loads(evo.read_text(encoding="utf-8"))
        traj = data.get("trajectory", [])
        n_gens = len(traj)
        if n_gens == 0:
            print(f"{d.name:<25} empty trajectory")
            continue
        final = traj[-1].get("cooperation_rate_mean", 0)
        mean = sum(g["cooperation_rate_mean"] for g in traj) / n_gens
        # Parse name and seed from dir name
        parts = d.name.rsplit("_seed", 1)
        name = parts[0]
        seed = parts[1] if len(parts) > 1 else "?"
        print(f"{name:<25} {seed:<5} {n_gens:<7} {final:<12.3f} {mean:<10.3f}")
    except Exception as e:
        print(f"{d.name:<25} ERROR: {e}")
