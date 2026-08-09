"""Dump the final generation (gen 29) best agent strategy for all 3 LLM seeds + a brief signature."""
import json
import re

def signature(code):
    """Extract a compact behavioral signature from the strategy code."""
    # decide threshold
    decide = re.search(r"return\s+opponent_reputation\s*([><=!]+)\s*\(?(-?[0-9.]+)?", code)
    thresh = f"{decide.group(1)}{decide.group(2)}" if decide else "?"
    # evaluate: my_reputation contribution
    my_rep = re.search(r"(\(?-?[0-9.]+\)?\s*\*\s*my_reputation)", code)
    my_rep_w = my_rep.group(1) if my_rep else "none"
    # evaluate: recipient_reputation contribution
    rec_rep = re.search(r"(\(?-?[0-9.]+\)?\s*\*\s*recipient_reputation)", code)
    rec_rep_w = rec_rep.group(1) if rec_rep else "none"
    # step size
    step = re.findall(r"step\s*=\s*([0-9.]+)", code)
    return f"decide:{thresh} | my_w:{my_rep_w} | rec_w:{rec_rep_w} | steps:{step}"


for seed in [0, 1, 2]:
    path = f"results/quantitative_baseline/LLM_evolution_seed{seed}/evolutionary.json"
    d = json.load(open(path, encoding="utf-8"))
    last = d["trajectory"][-1]
    coop = last["cooperation_rate_mean"]
    pop = last["population"]
    best = max(pop, key=lambda a: a.get("fitness", -1e9))
    code = best["code"]
    print(f"=== LLM seed {seed}  gen 29  coop(pop)={coop:.3f}  best(fit={best['fitness']:.2f}, own_coop={best['cooperation_rate']:.2f}) ===")
    print(f"Signature: {signature(code)}")
    print(f"--- code ---")
    print(code)
    print()
