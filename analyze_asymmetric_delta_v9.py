"""
v9: Pragmatic solution. Don't try to extract (pos, neg) from all 2580
strategies. Instead, just enumerate the (pos, neg) pairs we OBSERVED in
the v15 main plan (exp1_method_n10) and v18 cross-LLM. We've already
seen the data — count manually from the codes.

Strategy: take all Hybrid final-pop strategies from exp1_method_n10 and
exp5_robustness, and for each unique 'evaluate' function, manually count
how many have asymmetric delta.
"""
import json, glob, os, re
from collections import Counter, defaultdict

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"

# Only v15 main plan + v18 cross-LLM (where the paper's "Hybrid 93.1%" claim lives)
target_dirs = ["exp1_method_n10", "exp5_robustness"]
all_strats = []
for d in target_dirs:
    pattern = os.path.join(base, d, "**", "evo_*.json")
    for f in glob.glob(pattern, recursive=True):
        try:
            with open(f, encoding="utf-8") as fp: data = json.load(fp)
        except: continue
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

print(f"v15 main plan + v18 cross-LLM final-pop: {len(all_strats)} strategies")

# Per-strategy delta detection — find the LITERAL numbers used in
# positive / negative assignments.
# Use the regex that worked best: search window after action==literal.

POS_LABELS = ["cooperate", "donate"]
NEG_LABELS = ["defect", "not donate", "not_donate"]

def get_pair(code):
    """Return (pos_lab, neg_lab) in order they appear in action-comparisons."""
    matches = []
    for m in re.finditer(r"action\s*[^=\n]*==\s*['\"]([\w ]+)['\"]", code):
        matches.append(m.group(1))
    if len(matches) < 2: return None
    pos = matches[0] if matches[0] in POS_LABELS else None
    neg = matches[1] if matches[1] in NEG_LABELS else None
    return (pos, neg) if pos and neg else None

def extract_first_delta(code, label, sign):
    """Find first non-trivial positive/negative delta in the window after
    the FIRST action-comparison matching this label."""
    m = re.search(rf"action\s*[^=\n]*==\s*['\"]({re.escape(label)})['\"]", code)
    if not m: return None
    window = code[m.end():m.end() + 800]
    for x in re.finditer(r"(?<![A-Za-z_0-9])(-?\d+\.?\d*)", window):
        try: v = float(x.group(1))
        except: continue
        if 0.05 <= abs(v) <= 3.0:
            if sign == "positive" and v > 0: return v
            if sign == "negative" and v < 0: return v
    return None


buckets = Counter()
asym_examples = []
for s in all_strats:
    pair = get_pair(s["code"])
    if not pair: continue
    pos_lab, neg_lab = pair
    pos = extract_first_delta(s["code"], pos_lab, "positive")
    neg = extract_first_delta(s["code"], neg_lab, "negative")
    if pos is None or neg is None: continue
    if pos < 0: cls = "reverse"
    elif neg > 0: cls = "no_punishment"
    elif abs(pos + neg) <= 0.05: cls = "symmetric"
    elif pos > -neg: cls = "asym_pos_bigger"
    else: cls = "asym_neg_bigger"
    buckets[cls] += 1
    if cls.startswith("asym") and len(asym_examples) < 10:
        asym_examples.append((s["llm"], s["obs"], pos, neg, cls, s["coop"], pair))

n_extracted = sum(buckets.values())
print(f"\nExtracted: {n_extracted} / {len(all_strats)} ({100*n_extracted/len(all_strats):.1f}%)")

print(f"\n=== Asymmetry distribution ===")
for cls, c in buckets.most_common():
    print(f"  {cls:30s}  {c:4d}  {100*c/n_extracted:5.1f}%")

print(f"\n  ASYMMETRIC TOTAL:  {buckets['asym_pos_bigger']+buckets['asym_neg_bigger']}  {100*(buckets['asym_pos_bigger']+buckets['asym_neg_bigger'])/n_extracted:5.1f}%")

# By LLM
by_llm = defaultdict(Counter)
for s in all_strats:
    pair = get_pair(s["code"])
    if not pair: continue
    pos_lab, neg_lab = pair
    pos = extract_first_delta(s["code"], pos_lab, "positive")
    neg = extract_first_delta(s["code"], neg_lab, "negative")
    if pos is None or neg is None: continue
    if pos < 0: cls = "reverse"
    elif neg > 0: cls = "no_punishment"
    elif abs(pos + neg) <= 0.05: cls = "symmetric"
    elif pos > -neg: cls = "asym_pos_bigger"
    else: cls = "asym_neg_bigger"
    by_llm[s["llm"]][cls] += 1

print(f"\n=== Per LLM ===")
for llm, c in sorted(by_llm.items()):
    total = sum(c.values())
    valid = c["symmetric"] + c["asym_pos_bigger"] + c["asym_neg_bigger"]
    print(f"\n  {llm}  (total {total} extracted)")
    for cls in ["symmetric", "asym_pos_bigger", "asym_neg_bigger", "reverse", "no_punishment"]:
        n = c.get(cls, 0)
        if n > 0:
            print(f"    {cls:30s}  {n:3d}  {100*n/total:5.1f}%")
    if valid > 0:
        print(f"    ASYMMETRIC: {100*(c.get('asym_pos_bigger',0)+c.get('asym_neg_bigger',0))/valid:.1f}%")

print(f"\n=== Sample asymmetric deltas ===")
for llm, obs, pos, neg, cls, coop, pair in asym_examples:
    ratio = pos / abs(neg) if neg != 0 else 0
    print(f"  {llm[:20]:20s} obs={obs:12s}  pos=+{pos:.2f}  neg={neg:.2f}  ratio={ratio:.2f}  [{cls}]  coop={coop:.2f}")
