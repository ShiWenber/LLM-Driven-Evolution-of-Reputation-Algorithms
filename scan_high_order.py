"""Scan all LLM strategies for any high-order / non-IS patterns."""
import json
import re
from collections import Counter

all_strategies = []
for seed in [0, 1, 2]:
    d = json.load(open(f"results/quantitative_baseline/LLM_evolution_seed{seed}/evolutionary.json", encoding="utf-8"))
    for e in d["trajectory"]:
        for a in e.get("population", []):
            code = a.get("code", "")
            all_strategies.append((seed, e["generation"], a.get("fitness", 0), code))

print(f"Total LLM strategies scanned: {len(all_strategies)} (3 seeds x 30 gens x 15 agents)")

# Detect patterns
patterns = {
    "uses_my_reputation": lambda c: "my_reputation" in c,
    "uses_recipient_reputation": lambda c: "recipient_reputation" in c,
    "has_AND_in_decide": lambda c: bool(re.search(r"def decide\(.*?\):(.*?)(\n\ndef |\Z)", c, re.DOTALL))
        and " and " in re.search(r"def decide\(.*?\):(.*?)(\n\ndef |\Z)", c, re.DOTALL).group(1),
    "has_OR_in_decide": lambda c: bool(re.search(r"def decide\(.*?\):(.*?)(\n\ndef |\Z)", c, re.DOTALL))
        and " or " in re.search(r"def decide\(.*?\):(.*?)(\n\ndef |\Z)", c, re.DOTALL).group(1),
    "uses_abs()": lambda c: "abs(" in c,
    "uses_NOT_in_decide": lambda c: bool(re.search(r"return\s+not\b", c)),
    "uses_arithmetic_in_decide": lambda c: bool(re.search(r"return\s+opponent_reputation\s*[+\-*/]", c))
        or bool(re.search(r"return\s+my_reputation\s*[+\-*/]", c)),
    "uses_dict_or_list_storage": lambda c: "[" in c.split("def ")[0] or "{" in c.split("def ")[0] or "dict(" in c or "list(" in c,
    "uses_while_or_for_loop": lambda c: "\nwhile " in c or "\nfor " in c,
    "calls_random": lambda c: "random" in c,
    "has_extra_function_defs": lambda c: c.count("def ") > 2,
    "calls_my_reputation_in_decide": lambda c: "my_reputation" in (re.search(r"def decide\(.*?\):(.*?)(\n\ndef |\Z)", c, re.DOTALL).group(1) if re.search(r"def decide\(.*?\):(.*?)(\n\ndef |\Z)", c, re.DOTALL) else ""),
    "uses_reputation_in_recipient": lambda c: "my_reputation" in c.split("def evaluate")[1].split("def decide")[0] if "def evaluate" in c and "def decide" in c else False,
}

counts = {}
for k, fn in patterns.items():
    counts[k] = sum(1 for _, _, _, c in all_strategies if fn(c))

for k, n in counts.items():
    print(f"  {k:38s}: {n:4d}/{len(all_strategies)} ({100*n/len(all_strategies):.1f}%)")

# Find the most "exotic" strategies
print()
print("=== Most exotic strategies (by code length) ===")
all_strategies.sort(key=lambda x: -len(x[3]))
for s, g, f, code in all_strategies[:3]:
    print(f"  seed={s} gen={g} fit={f} len={len(code)} chars")
    print(f"  --- code ---")
    print(code[:500] + "..." if len(code) > 500 else code)
    print()
