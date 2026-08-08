"""Interpret the strategies that emerged in the Fermi Z-like 3-seed run.

For each seed, look at the 15 final agents' code and classify them
into rough strategy families. Show counts and a sample description.
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("results/quantitative_baseline")
SEEDS = [0, 1, 2]

# Strategy family classification — simple keyword + structure heuristics.
def classify(code: str, coop_rate: float) -> str:
    """Return a short family label."""
    if "return True" in code and "return False" not in code:
        return "ALLC-like (always cooperate)"
    if "return False" in code and "return True" not in code:
        return "ALLD-like (always defect)"
    # Has 'reputation' dict-like state
    has_reputation = bool(re.search(r"self\.reputation\s*=", code))
    has_history = bool(re.search(r"self\.history\s*=", code))
    has_last_action = bool(re.search(r"self\.last_action\s*=", code))
    has_opponent_state = bool(re.search(r"self\.opponent[a-z_]*\s*=", code))
    has_random = "random.random" in code
    has_threshold = bool(re.search(r"threshold\s*=|self\.\w*_threshold\s*=", code, re.IGNORECASE))
    # has 'if' branch that depends on reputation/history
    has_reputation_branch = bool(re.search(r"if\s+.*reputation", code))
    has_history_branch = bool(re.search(r"if\s+.*history", code))
    has_last_action_branch = bool(re.search(r"if\s+.*last_action", code))

    features = []
    if has_reputation: features.append("rep")
    if has_history: features.append("hist")
    if has_last_action: features.append("lastA")
    if has_opponent_state: features.append("oppState")
    if has_random: features.append("random")
    if has_threshold: features.append("thresh")
    if has_reputation_branch or has_history_branch or has_last_action_branch: features.append("cond")

    # cooperation rate as a feature
    if coop_rate >= 0.98:
        return f"high-coop ({','.join(features) if features else 'no-state'})"
    if coop_rate <= 0.05:
        return f"defect-or-no-state ({','.join(features) if features else 'no-state'})"
    return f"mixed ({','.join(features) if features else 'no-state'})"


for seed in SEEDS:
    p = BASE / f"LLM_v3_fermi_z_g100_1000inter_seed{seed}/evolutionary.json"
    d = json.load(open(p))
    pop = d["final_population"]
    families = Counter()
    family_samples = defaultdict(list)
    per_agent_data = []
    for a in pop:
        f = classify(a["code"], a["cooperation_rate"])
        families[f] += 1
        if len(family_samples[f]) < 1:
            family_samples[f].append({
                "agent_id": a["agent_id"],
                "coop": a["cooperation_rate"],
                "fitness": a["fitness"],
                "code_first_300": a["code"][:300],
            })
        per_agent_data.append({
            "agent_id": a["agent_id"],
            "family": f,
            "coop": a["cooperation_rate"],
            "fitness": a["fitness"],
        })
    # Sort by coop desc
    per_agent_data.sort(key=lambda x: -x["coop"])

    print(f"\n=== seed{seed} ===")
    print(f"  families ({len(families)} unique):")
    for fam, cnt in families.most_common():
        print(f"    [{cnt:2d}] {fam}")
    print(f"  per-agent coop rates (sorted desc):")
    for a in per_agent_data:
        print(f"    agent{a['agent_id']:2d}: coop={a['coop']:.3f}, fitness={a['fitness']:5.1f}, family={a['family']}")
    print(f"  sample codes (one per family):")
    for fam, samples in family_samples.items():
        s = samples[0]
        print(f"    --- {fam} (agent {s['agent_id']}, coop={s['coop']:.3f}) ---")
        # extract first 5 lines of code
        first_lines = s["code_first_300"].split("\n")[:5]
        for line in first_lines:
            print(f"      {line[:120]}")
