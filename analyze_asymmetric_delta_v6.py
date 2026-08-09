"""
v6: Most pragmatic — find EVERY numeric literal in (0.05, 3.0) within the
evaluate() body, then assign a "side" (positive/negative/none) by context:

  - If the number is preceded by ' + ' (a space-plus-space) and is in a
    cooperate block, count as positive delta.
  - If preceded by ' - ' in a defect block, count as negative delta.
  - If ' += ' or ' = -', the same.

Skip numbers that appear inside `max(-1, ...)` / `min(1, ...)` clamp.
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

# Find evaluate() body
def get_eval_body(code: str):
    m = re.search(r"def\s+evaluate\s*\(", code)
    if not m: return None
    nxt = re.search(r"\n\s*def\s+", code[m.end():])
    end = m.end() + nxt.start() if nxt else len(code)
    return code[m.end():end]


def extract_deltas_v6(code: str):
    """
    Strategy: find the "if/elif action == X" position. Then for each
    numeric literal in the next 800 chars, decide if it belongs to the
    X branch (good) or the next one (skip).

    Simpler approach: just find pairs of "if action == 'cooperate'" and
    "if action == 'defect'" positions. Between each, take numeric
    literals. For each branch, take the FIRST literal in (0.05, 3.0)
    that has the expected sign (positive for cooperate, negative for
    defect).
    """
    if code is None: return None
    body = get_eval_body(code)
    if not body: return None

    # Find all "action ... == '<X>'" positions
    matches = []
    for m in re.finditer(r"action\s*[^=\n]*==\s*['\"](cooperate|defect)['\"]", body):
        matches.append((m.start(), m.end(), m.group(1)))
    if len(matches) < 2: return None

    # We need at least one "cooperate" and one "defect" in the order
    # they appear in the code
    coop_positions = [(s, e) for s, e, x in matches if x == "cooperate"]
    defect_positions = [(s, e) for s, e, x in matches if x == "defect"]
    if not coop_positions or not defect_positions: return None

    # Take the first occurrence of each
    cs, ce = coop_positions[0]
    ds, de = defect_positions[0]

    # Get the window of code in each branch:
    # If they appear in order (cooperate first, then defect), the
    # cooperate window is from ce to ds, defect is from de to next
    # elif/else/return.
    if cs < ds:
        coop_window = body[ce:ds]
        # defect window: de to end or 600 chars
        defect_window = body[de:de + 600]
    else:
        # defect first (rare)
        defect_window = body[de:cs]
        coop_window = body[ce:ce + 600]

    def first_in_range(window, sign):
        for m in re.finditer(r"(?<![A-Za-z_])(-?\d+\.?\d*)", window):
            try: v = float(m.group(1))
            except: continue
            if 0.05 <= abs(v) <= 3.0:
                if sign == "positive" and v > 0: return v
                if sign == "negative" and v < 0: return v
        return None

    pos = first_in_range(coop_window, "positive")
    neg = first_in_range(defect_window, "negative")
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
    deltas = extract_deltas_v6(s["code"])
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

print(f"\nExtracted from {extracted} / {len(all_strats)} ({100*extracted/len(all_strats):.1f}%)")
print(f"Unknown: {unknown}")

print(f"\n=== Overall asymmetry distribution (over {extracted} extractable) ===")
for cls, count in buckets.most_common():
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
