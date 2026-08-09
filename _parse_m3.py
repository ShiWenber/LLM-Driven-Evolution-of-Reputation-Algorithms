import json
from pathlib import Path
p = Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline\LLM_evolution_seed0\evolutionary.json")
r = json.loads(p.read_text(encoding="utf-8"))
cfg = r.get("config", {})
traj = r.get("trajectory", [])
print("config:", cfg)
print("elapsed_sec:", r.get("elapsed_sec"))
print("total gens:", len(traj))
print("schema_version:", cfg.get("schema_version"))
print("agent_type:", cfg.get("agent_type"))
print("---")
print("per gen coop / fitness:")
for i, t in enumerate(traj):
    co = t.get("cooperation_rate_mean", 0.0)
    fi = t.get("fitness_mean", 0.0)
    ua = t.get("unique_actions", "?")
    print(f"  gen {i:2d}: coop={co:.4f}  fitness={fi:.3f}  unique_actions={ua}")
print("---")
final = r.get("final_population", [])
print("final_population size:", len(final))
print("final agent_ids sample:", sorted([a.get("agent_id") for a in final]))
# Count fallback vs non-fallback in final
fallback_n = sum(1 for a in final if a.get("code", "").startswith("# FALLBACK"))
non_fb_n = len(final) - fallback_n
print(f"final fallback agents: {fallback_n}, non-fallback: {non_fb_n}")
# Count unique non-fallback codes
codes = set()
for a in final:
    code = a.get("code", "")
    if not code.startswith("# FALLBACK"):
        codes.add(code[:200])
print(f"unique non-fallback code prefixes in final: {len(codes)}")
