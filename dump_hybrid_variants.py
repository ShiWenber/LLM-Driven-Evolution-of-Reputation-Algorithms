"""Find a trial with high strategy diversity (multiple Hybrid subtypes) at gen 9/10."""
import json, os
from collections import Counter

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"

# Look at exp1_method_n10 full_seed0 (the one whose dump we already saw)
# It has high diversity: 15 different strategy_ids across Hybrid, EMA, etc.
target = os.path.join(base, "exp1_method_n10", "full_seed0")
files = [f for f in os.listdir(target) if f.startswith("evo_") and f.endswith(".json")]
f = os.path.join(target, files[0])
with open(f, encoding="utf-8") as fp:
    d = json.load(fp)

final = d["final_population"]
print(f"FILE: {target}")
print(f"CONFIG: N={d['config']['population_size']}  T={d['config']['num_rounds_per_gen']}  obs={d['config']['observability']}  LLM={d['config']['llm_model']}")
print(f"FINAL POP: {len(final)} strategies, {len(set(a['strategy_id'] for a in final))} unique IDs\n")

# Cluster by similarity: count "key features" per code
def features(code):
    if code is None: return {}
    f = {}
    # Update rule
    if "current_reputation + 1" in code or "current_reputation + 1.0" in code:
        f["simple_addition"] = True
    if "0.9 * current_reputation" in code or "0.85 * current_reputation" in code or "0.8 * current_reputation" in code or "0.7 * current_reputation" in code or "0.6 * current_reputation" in code:
        f["decay_ema"] = True
    if "alpha" in code.lower():
        f["alpha_param"] = True
    if "(1 - alpha) * current_reputation" in code or "(1-alpha)*current_reputation" in code:
        f["ema_formula"] = True
    if "0.5 * current_reputation" in code:
        f["half_ema"] = True
    if "current_reputation" not in code:
        f["no_reputation"] = True
    # Asymmetric
    if re.search(r"delta\s*=\s*0\.\d+.*else.*-0\.\d+", code, re.DOTALL) or re.search(r"\+ 0\.\d+\n.*else.*- 0\.\d+", code, re.DOTALL):
        f["asymmetric_delta"] = True
    # Self-referential threshold
    if "my_history" in code and ("if " in code and "my_history" in code) and "threshold" in code:
        f["self_ref_threshold"] = True
    # Global counter
    if re.search(r"^\w+\s*=\s*\[0\]", code, re.MULTILINE) or re.search(r"^\w+\s*=\s*0\s*$", code, re.MULTILINE):
        f["global_counter"] = True
    # Decide rule
    if re.search(r"return\s+recipient_reputation\s*>=\s*-?0\.\d+", code):
        f["fixed_threshold"] = True
    if re.search(r"threshold\s*=\s*-?0\.\d+\s*-\s*0\.\d+\s*\*", code):
        f["mood_threshold"] = True
    if re.search(r"round_num\s*<=\s*\d+", code):
        f["early_round_special"] = True
    if "if round_num <=" in code and "return True" in code:
        f["honeymoon_period"] = True
    return f

import re
for a in final:
    fs = features(a["code"])
    feat_str = ", ".join(k for k, v in fs.items() if v) or "(none)"
    print(f"  agent {a['agent_id']:2d}  str {a['strategy_id'][:8]}  coop {a.get('cooperation_rate', 0):.2f}  fit {a.get('fitness', 0):5.1f}  gens-old {d['trajectory'][-1]['generation'] - a['generation']}")
    print(f"     features: {feat_str}")
    print()
