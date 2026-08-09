"""Inspect actual evaluate() functions to design better regex."""
import json, glob, os, re, random

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"

target_dirs = [
    "exp1_method_n10", "exp5_robustness", "exp6_sweep_AB_n5", "exp6_sweep_CD_n5",
    "exp7_algorithmic_ceiling", "exp8_intern_ceiling_v18", "exp8_intern_ceiling_v19_A",
    "exp9_bc_scan",
]

# Collect all final-pop codes
all_strats = []
for d in target_dirs:
    pattern = os.path.join(base, d, "**", "evo_*.json")
    for f in glob.glob(pattern, recursive=True):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception: continue
        if "final_population" not in data: continue
        for a in data["final_population"]:
            all_strats.append(a.get("code", ""))

random.seed(0)
sample = random.sample(all_strats, 8)
for i, code in enumerate(sample):
    print(f"\n{'='*60}\n[Sample {i}]\n{'='*60}")
    # Extract just the evaluate() function
    m = re.search(r"def evaluate\(.*?(?=\ndef |\Z)", code, re.DOTALL)
    if m:
        print(m.group(0))
