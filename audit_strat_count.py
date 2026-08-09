"""Audit final-population strategy counts across all experiment folders."""
import json, glob, os
from collections import defaultdict

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"
subdirs = [
    "exp1_method", "exp1_method_n10",
    "exp2_threshold",
    "exp3_static", "exp3_static_g10", "exp3_static_g10_n10",
    "exp4_random_mut",
    "exp5_robustness",
    "exp6_batch_test", "exp6_leading_eight",
    "exp6_sweep_AB", "exp6_sweep_AB_n5", "exp6_sweep_CD_n5",
    "exp7_algorithmic_ceiling",
    "exp8_intern_ceiling", "exp8_intern_ceiling_v18", "exp8_intern_ceiling_v19_A", "exp8_intern_quick",
    "exp9_bc_scan",
    "exp10_reasoning_trace",
    "exp11_thinking_compare",
    "dryrun_complexity_ceiling", "dryrun_complexity_ceiling_v2",
]

totals = {}
for d in subdirs:
    p = os.path.join(base, d)
    if not os.path.exists(p):
        continue
    n_strats = 0
    n_trials = 0
    for root, dirs, files in os.walk(p):
        for f in files:
            if f.endswith(".json") and ("evo_" in f or "evolutionary" in f):
                try:
                    with open(os.path.join(root, f), encoding="utf-8") as fh:
                        data = json.load(fh)
                    pop = data.get("final_population", [])
                    if pop:
                        n_strats += len(pop)
                        n_trials += 1
                except Exception:
                    pass
    if n_trials > 0:
        totals[d] = (n_trials, n_strats)

print(f"{'subdir':<40} {'trials':>8} {'strategies':>10}")
print("-" * 60)
grand_trials = 0
grand_strats = 0
for d, (t, s) in sorted(totals.items()):
    print(f"{d:<40} {t:>8} {s:>10}")
    grand_trials += t
    grand_strats += s
print("-" * 60)
print(f"{'TOTAL':<40} {grand_trials:>8} {grand_strats:>10}")
