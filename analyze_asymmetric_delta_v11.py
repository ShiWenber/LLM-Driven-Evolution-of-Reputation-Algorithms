"""
v11: Don't ignore 1.0. The "canonical IS" is exactly +1.0/-1.0. So
including them will pick up the most common case. Then classify:
  pos=1.0, neg=-1.0   =>  symmetric (canonical IS)
  pos != -neg          =>  asymmetric
  pos != abs(neg)      =>  asymmetric
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
            with open(f, encoding="utf-8") as fp: data = json.load(fp)
        except: continue
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

POS_PATTERNS = [r"['\"]cooperate['\"]", r"['\"]donate['\"]"]
NEG_PATTERNS = [r"['\"]defect['\"]", r"['\"]not donate['\"]", r"['\"]not_donate['\"]"]

def find_pos_in_action_compare(code):
    for pat in POS_PATTERNS:
        for m in re.finditer(pat, code):
            start = max(0, m.start() - 50)
            if "==" in code[start:m.start()]:
                return m.end()
    return None

def find_neg_in_action_compare(code):
    for pat in NEG_PATTERNS:
        for m in re.finditer(pat, code):
            start = max(0, m.start() - 50)
            if "==" in code[start:m.start()]:
                return m.end()
    return None

def first_delta_in_window(window, sign):
    """First non-trivial number, including 1.0."""
    for m in re.finditer(r"(?<![A-Za-z_0-9.])(-?\d+\.?\d*)", window):
        try: v = float(m.group(1))
        except: continue
        if 0.05 <= abs(v) <= 3.0:
            if sign == "positive" and v > 0: return v
            if sign == "negative" and v < 0: return v
    return None

buckets = Counter()
per_llm = defaultdict(Counter)
per_obs = defaultdict(Counter)
asym_examples = []
extracted = 0

for s in all_strats:
    code = s["code"]
    pos_pos = find_pos_in_action_compare(code)
    neg_pos = find_neg_in_action_compare(code)
    if not pos_pos or not neg_pos:
        buckets["unknown"] += 1
        continue
    pos_window = code[pos_pos:pos_pos + 600]
    neg_window = code[neg_pos:neg_pos + 600]
    pos = first_delta_in_window(pos_window, "positive")
    neg = first_delta_in_window(neg_window, "negative")
    if pos is None or neg is None:
        buckets["unknown"] += 1
        continue
    extracted += 1
    if pos < 0: cls = "reverse"
    elif neg > 0: cls = "no_punishment"
    elif abs(pos + neg) <= 0.05: cls = "symmetric"
    elif pos > -neg: cls = "asym_pos_bigger"
    else: cls = "asym_neg_bigger"
    buckets[cls] += 1
    per_llm[s["llm"]][cls] += 1
    per_obs[s["obs"]][cls] += 1
    if cls.startswith("asym") and len(asym_examples) < 12:
        asym_examples.append((s["llm"], s["obs"], pos, neg, cls, s["coop"], s["fit"]))

unknown = buckets["unknown"]
print(f"\nExtracted from {extracted} / {len(all_strats)} ({100*extracted/len(all_strats):.1f}%)")
print(f"Unknown: {unknown}")

print(f"\n=== Overall asymmetry distribution (over {extracted} extractable) ===")
for cls, count in buckets.most_common():
    if cls == "unknown": continue
    print(f"  {cls:30s}  {count:4d}  {100*count/extracted:5.1f}%")

asym_pos = buckets["asym_pos_bigger"]
asym_neg = buckets["asym_neg_bigger"]
asym_total = asym_pos + asym_neg
print(f"\n  TOTAL ASYMMETRIC:  {asym_total:4d}  {100*asym_total/extracted:5.1f}%")
print(f"    reward > |punishment|:  {asym_pos:4d}  {100*asym_pos/extracted:5.1f}%")
print(f"    |punishment| > reward:  {asym_neg:4d}  {100*asym_neg/extracted:5.1f}%")
print(f"  Symmetric:    {buckets['symmetric']:4d}  {100*buckets['symmetric']/extracted:5.1f}%")

print(f"\n=== Per LLM ===")
print(f"{'LLM':25s}  {'sym':>5s}  {'asym_pos>':>10s}  {'asym_neg>':>10s}  {'reverse':>8s}  {'no_pun':>7s}  {'unk':>5s}  {'total':>6s}")
for llm in sorted(per_llm.keys()):
    c = per_llm[llm]
    total = sum(c.values())
    sym = c["symmetric"]; apos = c["asym_pos_bigger"]; aneg = c["asym_neg_bigger"]
    rev = c["reverse"]; np_ = c["no_punishment"]
    print(f"{llm[:25]:25s}  {sym:5d}  {apos:10d}  {aneg:10d}  {rev:8d}  {np_:7d}  {total:6d}")
    valid = sym + apos + aneg
    if valid > 0:
        print(f"{'':25s}    asym share: {100*(apos+aneg)/valid:.1f}%   (pos>neg: {100*apos/valid:.1f}%, neg>pos: {100*aneg/valid:.1f}%)")

print(f"\n=== Per observability ===")
print(f"{'obs':15s}  {'sym':>5s}  {'asym_pos>':>10s}  {'asym_neg>':>10s}  {'reverse':>8s}  {'no_pun':>7s}  {'unk':>5s}  {'total':>6s}")
for obs in sorted(per_obs.keys()):
    c = per_obs[obs]
    total = sum(c.values())
    sym = c["symmetric"]; apos = c["asym_pos_bigger"]; aneg = c["asym_neg_bigger"]
    rev = c["reverse"]; np_ = c["no_punishment"]
    print(f"{obs[:15]:15s}  {sym:5d}  {apos:10d}  {aneg:10d}  {rev:8d}  {np_:7d}  {total:6d}")
    valid = sym + apos + aneg
    if valid > 0:
        print(f"{'':15s}    asym share: {100*(apos+aneg)/valid:.1f}%")

print(f"\n=== Sample asymmetric deltas (first 12) ===")
for llm, obs, pos, neg, cls, coop, fit in asym_examples:
    ratio = pos / abs(neg) if neg != 0 else 0
    print(f"  {llm[:20]:20s} obs={obs:12s}  pos=+{pos:.2f}  neg={neg:.2f}  ratio={ratio:.2f}  [{cls}]  coop={coop:.2f}  fit={fit:.1f}")
