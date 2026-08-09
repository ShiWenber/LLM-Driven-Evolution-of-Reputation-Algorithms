"""
v4: Robust per-branch number extraction. Walk through the source, find
each `if/elif action == "X"` branch, extract the FIRST number in (0, 5)
or (-5, 0) range, ignoring "round_num" / "1.0" boundary literals.
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

# Find the body of the evaluate() function
def split_evaluate(code: str):
    m = re.search(r"def\s+evaluate\s*\([^)]*\)\s*:", code)
    if not m: return None
    start = m.end()
    # find matching end: next `def ` at column 0, OR balanced paren-end heuristic
    # Simpler: scan to the next "def " at start of line, or end of string
    m_end = re.search(r"^def\s+\w+\s*\(", code[start:], re.MULTILINE)
    if m_end:
        body = code[start:start + m_end.start()]
    else:
        body = code[start:]
    return body


def find_action_branches(body: str):
    """
    Return (coop_branch, defect_branch) — the substring of body between
    the if/elif action==X test and the next elif/else/return.
    """
    # Find if/elif ... action == "cooperate" or "defect"
    branches = {}
    # iterate over each if/elif
    for m in re.finditer(r"^\s*(if|elif)\s+[^:]*action\s*[^=]*==\s*['\"](cooperate|defect)['\"]", body, re.MULTILINE):
        kw = m.group(1)
        action = m.group(2)
        # Skip if this is the second/elif for the same action (rare)
        if action in branches and kw == "elif": continue
        if action in branches and kw == "if": continue  # only first 'if'
        # find end of this branch: next elif/else at same indent, or return
        start = m.end()
        # look for next keyword at same indent
        m_end = re.search(rf"^\s{{,{len(m.group(0)) - len(m.group(0).lstrip())}}}(elif|else|return)\b", body[start:], re.MULTILINE)
        if m_end:
            branches[action] = body[start:start + m_end.start()]
        else:
            branches[action] = body[start:]
    return branches.get("cooperate"), branches.get("defect")


def first_meaningful_number(block: str, want_sign: str):
    """
    Scan block for numeric literals. want_sign = 'positive' or 'negative'.
    Skip "round_num" literals (0, 1, 2, 3, 5, 10), skip 1.0 (clamp boundary),
    skip 0.0 (no-op). Skip comparison values that are clearly thresholds.
    Heuristic: take the first number in (0, 5) or (-5, 0) that's NOT a
    small threshold candidate.
    """
    if not block: return None
    # Reject numbers that look like thresholds or boundary
    ignore = {0.0, 1.0, 0.5, -1.0, -0.5}
    candidates = []
    for m in re.finditer(r"(-?\d+\.?\d*)", block):
        try: v = float(m.group(1))
        except: continue
        if v in ignore: continue
        if 0 < abs(v) <= 5.0:
            candidates.append(v)
    if not candidates: return None
    if want_sign == "positive":
        for v in candidates:
            if v > 0: return v
    elif want_sign == "negative":
        for v in candidates:
            if v < 0: return v
    return None


def extract_deltas_v4(code: str):
    if code is None: return None
    body = split_evaluate(code)
    if body is None: return None
    coop_b, defect_b = find_action_branches(body)
    if not coop_b or not defect_b: return None
    pos = first_meaningful_number(coop_b, "positive")
    neg = first_meaningful_number(defect_b, "negative")
    if pos is None or neg is None: return None
    return pos, neg


# Run analysis
buckets = Counter()
per_llm = defaultdict(Counter)
per_obs = defaultdict(Counter)
extracted = 0
unknown = 0
asym_examples = []

for s in all_strats:
    deltas = extract_deltas_v4(s["code"])
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
        cls = "asym_pos_bigger"  # reward > |punishment|
    else:
        cls = "asym_neg_bigger"  # |punishment| > reward
    buckets[cls] += 1
    per_llm[s["llm"]][cls] += 1
    per_obs[s["obs"]][cls] += 1
    if cls.startswith("asym") and len(asym_examples) < 8:
        asym_examples.append((s["llm"], s["obs"], pos, neg, cls, s["coop"], s["fit"]))

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
print(f"  Reverse / no_punishment:  {buckets['reverse']+buckets['no_punishment']:4d}  {100*(buckets['reverse']+buckets['no_punishment'])/extracted:5.1f}%")

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

print(f"\n=== Sample asymmetric deltas ===")
for llm, obs, pos, neg, cls, coop, fit in asym_examples:
    ratio = pos / abs(neg) if neg != 0 else 0
    print(f"  {llm[:25]:25s} obs={obs:12s}  pos=+{pos:.2f}  neg={neg:.2f}  ratio={ratio:.2f}  [{cls}]  coop={coop:.2f}  fit={fit:.1f}")
