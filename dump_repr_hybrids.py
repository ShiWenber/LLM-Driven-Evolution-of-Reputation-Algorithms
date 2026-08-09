"""Dump 1 representative code per Hybrid subtype found in final pop."""
import json, os, sys
# Force UTF-8 stdout for Windows PowerShell compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"
target = os.path.join(base, "exp1_method_n10", "full_seed0")
files = [f for f in os.listdir(target) if f.startswith("evo_") and f.endswith(".json")]
with open(os.path.join(target, files[0]), encoding="utf-8") as fp:
    d = json.load(fp)
final = d["final_population"]

# Pick representative strategies
picks = {
    "1_EMA_alpha": 7,       # alpha + EMA + fixed threshold
    "2_decay_simple": 8,    # simple decay + fixed threshold
    "3_mood_threshold": 4,  # decay_ema + mood threshold
    "4_asymmetric_alpha": 0, # alpha + asymmetric + self-ref
    "5_simple_threshold": 2, # alpha + self-ref threshold
    "6_round_special": 13,  # alpha + EMA + early-round
}

for label, idx in picks.items():
    a = final[idx]
    print("=" * 80)
    print(f"## {label}  (agent {a['agent_id']}, strategy {a['strategy_id'][:8]}, "
          f"coop={a.get('cooperation_rate', 0):.2f}, fit={a.get('fitness', 0):.1f}, "
          f"gen={a['generation']})")
    print("=" * 80)
    print(a["code"])
    print()
