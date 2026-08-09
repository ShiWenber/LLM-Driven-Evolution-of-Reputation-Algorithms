"""Extract the highest-cooperation final-population strategies."""
import json, os

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"

targets = [
    # FINAL coop 1.000 — full obs, N=10
    ("exp8_intern_ceiling_v18/C_reputation_noise/full_seed1", "Intern-S2-Preview, full obs, C probe (rep noise), seed 1, N=10"),
    # FINAL coop 1.000 — partial_0.7 obs, N=10
    ("exp9_bc_scan/partial_0.7_b3c1/seed0", "DeepSeek-V4-Flash, partial_0.7 obs, b/c=3, seed 0, N=10"),
    # FINAL coop 0.996 — partial_0.7 obs, N=15
    ("exp7_algorithmic_ceiling/B_recent_window/partial_0.7_seed1", "DeepSeek-V4-Flash, partial_0.7, B probe (recent_window), seed 1, N=15"),
    # FINAL coop 0.989 — partial_0.7 obs, N=15
    ("exp1_method_n10/partial_0.7_seed0", "DeepSeek-V4-Flash, partial_0.7, main plan, seed 0, N=15"),
]

for sub, desc in targets:
    path = os.path.join(base, sub)
    files = [f for f in os.listdir(path) if f.startswith("evo_") and f.endswith(".json")]
    if not files:
        print(f"!!! no evo file in {sub}")
        continue
    f = os.path.join(path, files[0])
    with open(f, encoding="utf-8") as fp:
        d = json.load(fp)
    cfg = d["config"]
    traj = d["trajectory"]
    final = d.get("final_population", [])
    print("=" * 80)
    print(f"FILE: {sub}")
    print(f"DESC: {desc}")
    print(f"CONFIG:  N={cfg['population_size']}  T={cfg['num_rounds_per_gen']}  obs={cfg['observability']}  p={cfg.get('observability_p')}  LLM={cfg['llm_model']}  seed={cfg['seed']}")
    print(f"TRAJECTORY:")
    for t in traj:
        print(f"   gen {t['generation']:2d}  coop {t['cooperation_rate_mean']:.3f}  fit_mean {t['fitness_mean']:.1f}  fit_max {t['fitness_max']}")
    print(f"FINAL POPULATION ({len(final)} strategies):")
    for a in final:
        print(f"\n  --- agent {a['agent_id']}  strategy {a['strategy_id'][:8]}  gen {a['generation']}  fit {a['fitness']:.1f}  coop {a.get('cooperation_rate', '?')} ---")
        print("  " + a["code"].replace("\n", "\n  "))
    print()
