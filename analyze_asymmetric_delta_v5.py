"""
v5: Pragmatic extraction — find each 'if/elif action == "cooperate"' and
'if/elif action == "defect"' by string position, then look in the NEXT
500 chars of code for the first non-trivial numeric literal in each sign.
"""
import json, glob, os, re
from collections import Counter, defaultdict

base = r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results"
target_dirs = [
    "exp1_method_n10", "exp5_robustness", "exp6_sweep_AB_n5", "exp6_sweep_CD_n5",
    "exp7_algorithmic_ceiling", "exp8_intern_ceiling_v18", "exp8_intern_ceiling_v19_A",
    "exp9_bc_scan",
]

all_strats = []
for d in target_dirs:
    pattern = os.path.join(base, d, "**", "evo_*.json")
    for f in glob.glob(pattern, recursive=True):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception: continue
        if "final_population" not in data: continue
        cfg = data.get("config", {})
        for a in data["final_population"]:
            all_strats.append({
                "code": a.get("code", ""),
                "llm": cfg.get("llm_model", "?"),
                "obs": cfg.get("observability", "?"),
                "coop": a.get("cooperation_rate", 0),
                "fit": a.get("fitness", 0),
            })

print(f"Total strategies scanned: {len(all_strats)}")

def extract_window(code: str, target: str, window: int = 600):
    """
    Find the FIRST occurrence of 'action ... == <target>' and return
    the next `window` characters of code (which usually contains the
    delta assignment).
    """
    pattern = rf"action\s*[^=\n]*==\s*['\"]({target})['\"]"
    m = re.search(pattern, code)
    if not m: return None
    return code[m.end():m.end() + window]


def first_meaningful_number(window: str, sign: str):
    """
    Find first numeric literal in (0, 5) or (-5, 0) range.
    Skip 0, 1.0, 0.5, -1.0, -0.5 (boundary/threshold common values).
    """
    if not window: return None
    ignore = {0.0, 1.0, 0.5, -1.0, -0.5, 2.0, -2.0, 0.01, 100.0, 0.001}
    for m in re.finditer(r"(-?\d+\.?\d*)", window):
        try: v = float(m.group(1))
        except: continue
        if v in ignore: continue
        if 0 < abs(v) < 5.0:
            if sign == "positive" and v > 0: return v
            if sign == "negative" and v < 0: return v
    return None


def extract_deltas_v5(code: str):
    if code is None: return None
    # Only consider evaluate() body
    m = re.search(r"def\s+evaluate\s*\(", code)
    if not m: return None
    # Find end of evaluate()
    next_def = re.search(r"\n\s*def\s+", code[m.end():])
    if next_def:
        eval_body = code[m.end():m.end() + next_def.start()]
    else:
        eval_body = code[m.end():]
    coop_w = extract_window(eval_body, "cooperate")
    defect_w = extract_window(eval_body, "defect")
    if not coop_w or not defect_w: return None
    pos = first_meaningful_number(coop_w, "positive")
    neg = first_meaningful_number(defect_w, "negative")
    if pos is None or neg is None: return None
    return pos, neg


# Run analysis
buckets = Counter()
per_llm = defaultdict(Counter)
per_obs = defaultdict(Counter)
extracted = 0
unknown = 0
asym_examples = []
sym_examples = []

for s in all_strats:
    deltas = extract_deltas_v5(s["code"])
    if deltas is None:
        buckets["unknown"] += 1
        unknown += 1
        continue
    extracted += 1
    pos, neg = deltas
    if pos < 0:
        cls = "reverse"
    elif neg > 0:
        cls = "no_punishment"
    elif abs(pos + neg) <= 0.05:
        cls = "symmetric"
    elif pos > -neg:
        cls = "asym_pos_bigger"
    else:
        cls = "asym_neg_bigger"
    buckets[cls] += 1
    per_llm[s["llm"]][cls] += 1
    per_obs[s["obs"]][cls] += 1
    if cls.startswith("asym") and len(asym_examples) < 10:
        asym_examples.append((s["llm"], s["obs"], pos, neg, cls, s["coop"], s["fit"]))
    if cls == "symmetric" and len(sym_examples) < 3:
        sym_examples.append((s["llm"], s["obs"], pos, neg, s["coop"], s["fit"]))

print(f"\nExtracted from {extracted} / {len(all_strats)} ({100*extracted/len(all_strats):.1f}%)")
print(f"Unknown: {unknown}")

print(f"\n=== Overall asymmetry distribution (over {extracted} extractable) ===")
for cls, count in buckets.most_common():
    print(f"  {cls:30s}  {count:4d}  {100*count/extracted:5.1f}%")

asym_pos = buckets["asym_pos_bigger"]
asym_neg = buckets["asym_neg_bigger"]
asym_total = asym_pos + asym_neg
print(f"\n  TOTAL ASYMMETRIC:  {asym_total:4d}  {100*asym_total/extracted:5.1f}%")
print(f"    reward > punishment:  {asym_pos:4d}  {100*asym_pos/extracted:5.1f}%")
print(f"    punishment > reward:  {asym_neg:4d}  {100*asym_neg/extracted:5.1f}%")
print(f"  Symmetric:    {buckets['symmetric']:4d}  {100*buckets['symmetric']/extracted:5.1f}%")

print(f"\n=== Per LLM ===")
print(f"{'LLM':25s}  {'sym':>5s}  {'asym_pos>':>10s}  {'asym_neg>':>10s}  {'reverse':>8s}  {'no_pun':>7s}  {'unk':>5s}  {'total':>6s}")
for llm in sorted(per_llm.keys()):
    c = per_llm[llm]
    total = sum(c.values())
    sym = c["symmetric"]; apos = c["asym_pos_bigger"]; aneg = c["asym_neg_bigger"]
    rev = c["reverse"]; np_ = c["no_punishment"]; unk = c["unknown"]
    print(f"{llm[:25]:25s}  {sym:5d}  {apos:10d}  {aneg:10d}  {rev:8d}  {np_:7d}  {unk:5d}  {total:6d}")
    valid = sym + apos + aneg
    if valid > 0:
        print(f"{'':25s}    asym share: {100*(apos+aneg)/valid:.1f}%   (pos>neg: {100*apos/valid:.1f}%, neg>pos: {100*aneg/valid:.1f}%)")

print(f"\n=== Per observability ===")
print(f"{'obs':15s}  {'sym':>5s}  {'asym_pos>':>10s}  {'asym_neg>':>10s}  {'reverse':>8s}  {'no_pun':>7s}  {'unk':>5s}  {'total':>6s}")
for obs in sorted(per_obs.keys()):
    c = per_obs[obs]
    total = sum(c.values())
    sym = c["symmetric"]; apos = c["asym_pos_bigger"]; aneg = c["asym_neg_bigger"]
    rev = c["reverse"]; np_ = c["no_punishment"]; unk = c["unknown"]
    print(f"{obs[:15]:15s}  {sym:5d}  {apos:10d}  {aneg:10d}  {rev:8d}  {np_:7d}  {unk:5d}  {total:6d}")
    valid = sym + apos + aneg
    if valid > 0:
        print(f"{'':15s}    asym share: {100*(apos+aneg)/valid:.1f}%")

print(f"\n=== Sample asymmetric deltas (first 10) ===")
for llm, obs, pos, neg, cls, coop, fit in asym_examples:
    ratio = pos / abs(neg) if neg != 0 else 0
    print(f"  {llm[:25]:25s} obs={obs:12s}  pos=+{pos:.2f}  neg={neg:.2f}  ratio={ratio:.2f}  [{cls}]  coop={coop:.2f}  fit={fit:.1f}")
