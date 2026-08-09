"""Analyze v3 LLM final-generation strategies: extract code, classify, compare.
Output: human-readable summary + per-seed final strategies + key features.
"""
import json
import re
from pathlib import Path

ROOT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline")

def extract_code_features(code: str) -> dict:
    """Extract key features from a v2 strategy code."""
    feats = {}

    # Determine decide() body
    m = re.search(r"def\s+decide\s*\([^)]*\)\s*:\s*return\s+([^\n]+)", code)
    if m:
        body = m.group(1).strip()
        feats["decide"] = body
        # Categorize
        if "my_reputation" in body and "opponent_reputation" in body:
            # Extract threshold if exists
            tm = re.search(r"opponent_reputation\s*([><=!]+)\s*(-?\d*\.?\d+)", body)
            if tm:
                op = tm.group(1)
                th = float(tm.group(2))
                if op == ">=":
                    feats["decide_class"] = f"IS_thr{th:+.2f}"
                elif op == ">":
                    feats["decide_class"] = f"IS_strict_thr{th:+.2f}"
                elif op == "<=":
                    feats["decide_class"] = f"defect_under_thr{th:+.2f}"
                else:
                    feats["decide_class"] = f"op{op}{th}"
            else:
                feats["decide_class"] = "complex(mixed)"
        elif "True" in body:
            feats["decide_class"] = "ALLC"
        elif "False" in body:
            feats["decide_class"] = "ALLD"
        else:
            feats["decide_class"] = f"complex({body[:40]}...)"
    else:
        feats["decide"] = "(multi-line)"
        feats["decide_class"] = "complex"

    # Check evaluate() body
    m = re.search(r"def\s+evaluate\s*\([^)]*\)\s*:\s*([\s\S]+?)(?=\ndef\s|\Z)", code)
    if m:
        ebody = m.group(1)
    else:
        ebody = code
    # Look for 4-quadrant IS-like structure
    has_donor_action = "donor_action" in ebody
    has_recipient_rep = "recipient_reputation" in ebody
    has_donor_rep = "donor_reputation" in ebody
    has_my_rep = "my_reputation" in ebody
    feats["eval_uses_donor_action"] = has_donor_action
    feats["eval_uses_recipient_rep"] = has_recipient_rep
    feats["eval_uses_donor_rep"] = has_donor_rep
    feats["eval_uses_my_rep"] = has_my_rep
    # Canonical IS 4-quadrant
    is_canonical = has_donor_action and has_recipient_rep and ("cooperate" in ebody or "'cooperate'" in ebody or '"cooperate"' in ebody)
    feats["is_canonical_4quadrant"] = is_canonical
    return feats


print("=" * 80)
print("V3 LLM Final-Generation Strategies (3 seeds)")
print("=" * 80)

for seed in [0, 1, 2]:
    f = ROOT / f"LLM_evolution_seed{seed}" / "evolutionary.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    traj = j["trajectory"]
    final_pop = j["final_population"]
    final_coop = traj[-1]["cooperation_rate_mean"]
    final_mean = sum(t["cooperation_rate_mean"] for t in traj) / len(traj)

    print(f"\n{'='*80}")
    print(f"SEED {seed}: final coop = {final_coop:.3f}, trajectory mean = {final_mean:.3f}")
    print(f"{'='*80}")

    # Sort final population by fitness
    sorted_pop = sorted(final_pop, key=lambda a: a["fitness"], reverse=True)

    # Show top 3 + bottom 1
    for label, agents in [("TOP 3 (highest fitness)", sorted_pop[:3]),
                          ("BOTTOM 1 (lowest fitness)", sorted_pop[-1:])]:
        print(f"\n--- {label} ---")
        for i, a in enumerate(agents):
            print(f"\n  [agent {a['agent_id']}, fit={a['fitness']:.0f}, coop={a['cooperation_rate']:.2f}, self_rep={a['self_reputation']:.3f}]")
            feats = extract_code_features(a["code"])
            print(f"    decide: {feats.get('decide', '?')}")
            print(f"    class: {feats['decide_class']}")
            print(f"    eval uses: donor_action={feats['eval_uses_donor_action']}, "
                  f"recipient_rep={feats['eval_uses_recipient_rep']}, "
                  f"donor_rep={feats['eval_uses_donor_rep']}, "
                  f"my_rep={feats['eval_uses_my_rep']}")
            print(f"    canonical 4-quadrant IS? {feats['is_canonical_4quadrant']}")

# Cross-seed summary
print(f"\n{'='*80}")
print("CROSS-SEED SUMMARY")
print("="*80)

# Check if all final seeds converged to IS family
all_decide_classes = []
for seed in [0, 1, 2]:
    f = ROOT / f"LLM_evolution_seed{seed}" / "evolutionary.json"
    j = json.loads(f.read_text(encoding="utf-8"))
    for a in j["final_population"]:
        feats = extract_code_features(a["code"])
        all_decide_classes.append(feats["decide_class"])

from collections import Counter
class_counts = Counter(all_decide_classes)
print(f"\ndecide() class distribution across all 3 seeds × 15 agents = 45 final strategies:")
for cls, n in class_counts.most_common():
    print(f"  {n:3d} × {cls}")
