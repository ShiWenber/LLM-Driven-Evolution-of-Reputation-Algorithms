"""Find the highest cooperation-rate trial in v15 main plan."""
import json, glob, os, sys

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"

# Find all evo_*.json files (per-trial logs)
all_files = glob.glob(os.path.join(base, "**", "evo_*.json"), recursive=True)
print(f"Found {len(all_files)} trial files")

# For each, find the max cooperation_rate_mean across all generations
# AND the final generation's mean
records = []
for f in all_files:
    rel = os.path.relpath(f, base)
    try:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
    except Exception as e:
        print(f"  skip {rel}: {e}")
        continue
    if "trajectory" not in d:
        continue
    traj = d["trajectory"]
    if not traj:
        continue
    cfg = d.get("config", {})
    # max-gen cooperation
    max_gen_coop = max(t["cooperation_rate_mean"] for t in traj)
    final_gen_coop = traj[-1]["cooperation_rate_mean"]
    max_gen_idx = max(range(len(traj)), key=lambda i: traj[i]["cooperation_rate_mean"])
    records.append({
        "file": rel,
        "obs": cfg.get("observability", "?"),
        "p": cfg.get("observability_p"),
        "seed": cfg.get("seed"),
        "llm": cfg.get("llm_model"),
        "N": cfg.get("population_size"),
        "G": cfg.get("num_eliminate", "?"),
        "n_rounds": cfg.get("num_rounds_per_gen"),
        "max_coop": max_gen_coop,
        "max_coop_gen": traj[max_gen_idx]["generation"],
        "final_coop": final_gen_coop,
        "n_gens": len(traj),
    })

# Sort by max_coop desc
records.sort(key=lambda r: r["max_coop"], reverse=True)
print(f"\nTop 15 by max-cooperation-rate (any generation):")
print(f"{'file':60s} {'obs':15s} {'seed':4s} {'N':3s} {'max_coop':9s} {'@gen':4s} {'final':6s}")
for r in records[:15]:
    print(f"{r['file'][:60]:60s} {r['obs']:15s} {str(r['seed']):4s} {str(r['N']):3s} {r['max_coop']:.4f}    {r['max_coop_gen']:3d} {r['final_coop']:.3f}")

# Also: by FINAL cooperation
records_final = sorted(records, key=lambda r: r["final_coop"], reverse=True)
print(f"\nTop 15 by final-cooperation-rate:")
for r in records_final[:15]:
    print(f"{r['file'][:60]:60s} {r['obs']:15s} {str(r['seed']):4s} {str(r['N']):3s} {r['final_coop']:.4f}")
