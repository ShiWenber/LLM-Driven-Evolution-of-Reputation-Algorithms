"""
Quantify asymmetric delta across all 1440 final-population strategies.

Definition of "asymmetric delta": in evaluate(), the positive update magnitude
differs from the negative update magnitude by some meaningful amount.

Detection: find the two target/delta values used for "cooperate" vs "defect"
and check if |positive - negative| > threshold AND the signs are correct.
"""
import json, glob, os, re
from collections import Counter, defaultdict

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"

# All trial dirs
target_dirs = [
    "exp1_method_n10", "exp5_robustness", "exp6_sweep_AB_n5", "exp6_sweep_CD_n5",
    "exp7_algorithmic_ceiling", "exp8_intern_ceiling_v18", "exp8_intern_ceiling_v19_A",
    "exp9_bc_scan",
]

# Collect (llm, obs, code) for every final-pop strategy
all_strats = []
for d in target_dirs:
    pattern = os.path.join(base, d, "**", "evo_*.json")
    for f in glob.glob(pattern, recursive=True):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue
        if "final_population" not in data: continue
        cfg = data.get("config", {})
        for a in data["final_population"]:
            all_strats.append({
                "code": a.get("code", ""),
                "llm": cfg.get("llm_model", "?"),
                "obs": cfg.get("observability", "?"),
                "file": os.path.relpath(f, base),
                "coop": a.get("cooperation_rate", 0),
                "fit": a.get("fitness", 0),
            })

print(f"Total strategies scanned: {len(all_strats)}")


def extract_deltas(code: str):
    """
    Try to extract (delta_pos, delta_neg) from the evaluate() function.
    Returns (pos, neg) or None if not extractable.
    Handles several common LLM-written patterns:
      - if action == "cooperate": target = 0.6
      - if action == "cooperate": delta = 0.5 + ...
      - if action == "cooperate": new_rep = current + 0.3
      - if action == "cooperate": return current + 1.0
    """
    if code is None: return None
    # find action=='cooperate' block
    # Pattern 1: target = X / update = X / delta = X
    m_pos = re.search(r"action\s*[=!]=\s*['\"]cooperate['\"]\s*.*?(?:target|update|delta|base)\s*=\s*(-?\d+\.?\d*)", code, re.DOTALL)
    m_neg = re.search(r"action\s*[=!]=\s*['\"]defect['\"]\s*.*?(?:target|update|delta|base)\s*=\s*(-?\d+\.?\d*)", code, re.DOTALL)
    if m_pos and m_neg:
        return float(m_pos.group(1)), float(m_neg.group(1))
    # Pattern 2: if action == "cooperate": return current_reputation + X
    m_pos2 = re.search(r"action\s*[=!]=\s*['\"]cooperate['\"]\s*.*?return\s+current_reputation\s*\+\s*(-?\d+\.?\d*)", code, re.DOTALL)
    m_neg2 = re.search(r"action\s*[=!]=\s*['\"]defect['\"]\s*.*?return\s+current_reputation\s*-\s*(-?\d+\.?\d*)", code, re.DOTALL)
    if m_pos2 and m_neg2:
        return float(m_pos2.group(1)), float(m_neg2.group(1))
    # Pattern 3: if action == "cooperate": new_rep = current + X / elif ... current - Y
    m_pos3 = re.search(r"action\s*[=!]=\s*['\"]cooperate['\"].*?current_reputation\s*\+\s*(-?\d+\.?\d*)", code, re.DOTALL)
    m_neg3 = re.search(r"action\s*[=!]=\s*['\"]defect['\"].*?current_reputation\s*-\s*(-?\d+\.?\d*)", code, re.DOTALL)
    if m_pos3 and m_neg3:
        return float(m_pos3.group(1)), float(m_neg3.group(1))
    return None

def classify_asymmetry(pos, neg, tol=0.05):
    """
    Classify by sign of both deltas + magnitude.
    pos = positive delta when cooperate; neg = negative delta when defect.
    Conventions: reward positive, punishment negative.
    Returns ('symmetric' | 'asymmetric_pos_bigger' | 'asymmetric_neg_bigger' | 'reverse'
             | 'no_punishment' | 'no_reward' | 'unknown')
    """
    if pos is None or neg is None: return "unknown"
    if pos < 0: return "reverse"     # cooperate reduces rep — weird
    if neg > 0: return "no_punishment"  # defect doesn't reduce rep — always-cooperator
    # Both have expected signs: pos > 0, neg < 0
    if abs(pos + neg) <= tol: return "symmetric"
    if pos > -neg: return "asymmetric_pos_bigger"  # reward > punishment
    if pos < -neg: return "asymmetric_neg_bigger"  # punishment > reward
    return "symmetric"


# Run analysis
buckets = Counter()
per_llm = defaultdict(Counter)
per_obs = defaultdict(Counter)
extracted = 0
unknown = 0
asym_examples = []
for s in all_strats:
    deltas = extract_deltas(s["code"])
    if deltas is None:
        buckets["unknown"] += 1
        per_llm[s["llm"]]["unknown"] += 1
        per_obs[s["obs"]]["unknown"] += 1
        unknown += 1
        continue
    extracted += 1
    pos, neg = deltas
    cls = classify_asymmetry(pos, neg)
    buckets[cls] += 1
    per_llm[s["llm"]][cls] += 1
    per_obs[s["obs"]][cls] += 1
    if cls.startswith("asymmetric") and len(asym_examples) < 4:
        asym_examples.append((s["llm"], s["obs"], pos, neg, cls, s["coop"], s["fit"], s["file"]))

print(f"\nExtracted (pos, neg) from {extracted} / {len(all_strats)} strategies ({100*extracted/len(all_strats):.1f}%)")
print(f"Unknown / no clean pattern: {unknown}")

print(f"\n=== Overall asymmetry distribution (over {extracted} extractable) ===")
total_extracted = extracted
for cls, count in buckets.most_common():
    print(f"  {cls:30s}  {count:4d}  {100*count/total_extracted:5.1f}%")

# Asymmetric merged
asym = buckets["asymmetric_pos_bigger"] + buckets["asymmetric_neg_bigger"]
print(f"\n  Asymmetric (any direction):  {asym}  {100*asym/total_extracted:5.1f}%")
print(f"    with reward > punishment:  {buckets['asymmetric_pos_bigger']:4d}  {100*buckets['asymmetric_pos_bigger']/total_extracted:5.1f}%")
print(f"    with punishment > reward:  {buckets['asymmetric_neg_bigger']:4d}  {100*buckets['asymmetric_neg_bigger']/total_extracted:5.1f}%")

print(f"\n=== Per LLM ===")
llms = sorted(per_llm.keys())
print(f"{'LLM':25s}  {'sym':>4s}  {'asym_pos>':>10s}  {'asym_neg>':>10s}  {'reverse':>8s}  {'no_pun':>6s}  {'unk':>4s}  {'total':>5s}")
for llm in llms:
    c = per_llm[llm]
    total = sum(c.values())
    sym = c["symmetric"]
    apos = c["asymmetric_pos_bigger"]
    aneg = c["asymmetric_neg_bigger"]
    rev = c["reverse"]
    np_ = c["no_punishment"]
    unk = c["unknown"]
    print(f"{llm[:25]:25s}  {sym:4d}  {apos:10d}  {aneg:10d}  {rev:8d}  {np_:6d}  {unk:4d}  {total:5d}")
    if total > 0 and (sym + apos + aneg) > 0:
        asym_share = 100 * (apos + aneg) / (sym + apos + aneg + 0.001)
        print(f"{'':25s}    asym share (excl. unknown/reverse/no_pun): {asym_share:.1f}%")

print(f"\n=== Per observability ===")
print(f"{'obs':15s}  {'sym':>4s}  {'asym_pos>':>10s}  {'asym_neg>':>10s}  {'reverse':>8s}  {'no_pun':>6s}  {'unk':>4s}  {'total':>5s}")
for obs in sorted(per_obs.keys()):
    c = per_obs[obs]
    total = sum(c.values())
    print(f"{obs[:15]:15s}  {c['symmetric']:4d}  {c['asymmetric_pos_bigger']:10d}  {c['asymmetric_neg_bigger']:10d}  {c['reverse']:8d}  {c['no_punishment']:6d}  {c['unknown']:4d}  {total:5d}")

print(f"\n=== Asymmetric examples (first 4) ===")
for llm, obs, pos, neg, cls, coop, fit, fn in asym_examples:
    print(f"  {llm[:25]:25s} obs={obs:12s} pos=+{pos:.2f}  neg={neg:.2f}  -> {cls}")
    print(f"     coop={coop:.2f} fit={fit:.1f}  file={fn}")
