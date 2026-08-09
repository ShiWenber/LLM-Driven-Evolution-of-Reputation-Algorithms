"""
Robust delta extraction — regex over the whole code.

For each strategy code:
  1. Split the source into "cooperate block" and "defect block" by
     finding the position of "cooperate" and "defect" action literals
     in the action-comparison `if ... == 'cooperate' / 'defect'`.
  2. Within each block, look for any assignment involving numbers
     in the range (0, 5).
  3. Take the FIRST non-trivial number from each block.

This is intentionally permissive. The point is to spot *patterns* of
asymmetric-delta style choices, not to be perfectly precise.
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
        except Exception:
            continue
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

def find_block_text(code: str, target_action: str) -> str:
    """
    Find the substring between 'action ... == "<target>"' and the next
    'elif action ... ==' OR the next 'else' OR 500 chars, whichever first.
    """
    # Looser pattern: 'cooperate' or 'defect' inside an if/elif test
    # We find the FIRST occurrence of action == 'cooperate' (or defect)
    # and the FIRST occurrence of the OTHER one or an else after it.
    target_re = re.escape(target_action)
    other = "defect" if target_action == "cooperate" else "cooperate"
    other_re = re.escape(other)

    # Find start: 'action ... == "<target>"'
    m_start = re.search(rf"action\s*[^=]*==\s*['\"]{target_re}['\"]", code)
    if not m_start: return ""
    start = m_start.end()
    # Find end: either action==other OR else (without 'if' = elif/else) OR
    # the next 'if' or 'return' at the same indent, or 800 chars
    candidates = []
    m_end = re.search(rf"action\s*[^=]*==\s*['\"]{other_re}['\"]", code[start:])
    if m_end: candidates.append(start + m_end.start())
    m_else = re.search(r"^\s*else\s*:", code[start:], re.MULTILINE)
    if m_else: candidates.append(start + m_else.start())
    m_return = re.search(r"^\s*return\s+", code[start:], re.MULTILINE)
    if m_return: candidates.append(start + m_return.start())
    # Cut at 1000 chars
    candidates.append(start + 1000)
    end = min(candidates) if candidates else start + 1000
    return code[start:end]


def extract_deltas_regex(code: str):
    if code is None: return None
    coop_block = find_block_text(code, "cooperate")
    defect_block = find_block_text(code, "defect")
    if not coop_block or not defect_block: return None

    def first_positive_num(block):
        """Find first number in (0, 5) range that looks like a delta."""
        for m in re.finditer(r"(-?\d+\.?\d*)", block):
            v = float(m.group(1))
            if 0 < v < 5.0:
                return v
        return None

    def first_negative_num(block):
        """Find first negative number in (-5, 0) range."""
        for m in re.finditer(r"(-?\d+\.?\d*)", block):
            v = float(m.group(1))
            if -5.0 < v < 0:
                return v
        # fallback: any negative
        for m in re.finditer(r"(-?\d+\.?\d*)", block):
            v = float(m.group(1))
            if v < 0:
                return v
        return None

    pos = first_positive_num(coop_block)
    neg = first_negative_num(defect_block)
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
    deltas = extract_deltas_regex(s["code"])
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
    if cls.startswith("asym") and len(asym_examples) < 6:
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
