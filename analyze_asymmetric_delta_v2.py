"""
Robust asymmetric-delta analysis using AST inspection of evaluate().

Strategy: parse each strategy's code, find the evaluate() function,
walk its body, and for each "if/elif" branch that branches on
observation['action'] == "cooperate" / "defect", extract all numeric
literals that appear in the assignment to whatever variable holds the
update (target / delta / new_reputation / update / etc.).
"""
import json, glob, os, re, ast
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
                "file": os.path.relpath(f, base),
                "coop": a.get("cooperation_rate", 0),
                "fit": a.get("fitness", 0),
            })

print(f"Total strategies scanned: {len(all_strats)}")

def extract_deltas_v2(code: str):
    """
    Better extraction:
    1. Try to parse the code as Python AST.
    2. Find the evaluate() function.
    3. Find the if/elif/else that tests observation['action'].
    4. From each branch, collect numeric values that look like
       "target", "delta", "update", or "base" assignments.
    """
    if code is None: return None
    # Strip comments & docstrings for cleaner AST
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    # Find evaluate
    eval_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            eval_fn = node
            break
    if eval_fn is None: return None

    # Walk the body looking for the if/elif action== pattern
    def find_action_branch(stmts, target_action):
        """Return the list of statements that branch on action == target_action."""
        for stmt in stmts:
            if not isinstance(stmt, ast.If): continue
            test = stmt.test
            # test should be: observation['action'] == 'cooperate' (or 'defect')
            hit = False
            if (isinstance(test, ast.Compare) and len(test.ops) == 1 and
                isinstance(test.ops[0], ast.Eq)):
                left = test.left
                right = test.comparators[0]
                # left is observation['action'] (Subscript)
                # right is Constant 'cooperate' or 'defect'
                if (isinstance(left, ast.Subscript) and
                    isinstance(right, ast.Constant) and
                    isinstance(right.value, str) and
                    right.value == target_action):
                    hit = True
            if hit:
                return stmt.body
        return None

    coop_stmts = find_action_branch(eval_fn.body, "cooperate")
    defect_stmts = find_action_branch(eval_fn.body, "defect")
    if coop_stmts is None or defect_stmts is None: return None

    # Extract numerical literals from each branch.
    # Focus on positive constants (0 < x < 5) and small negative constants.
    def collect_numbers(stmts):
        nums = []
        for stmt in stmts:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                    v = float(n.value)
                    if 0 < abs(v) < 5.0:  # ignore 0, 1.0 clamp boundary, etc
                        nums.append(v)
                elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
                    if isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, (int, float)):
                        v = -float(n.operand.value)
                        if 0 < abs(v) < 5.0:
                            nums.append(v)
        return nums

    coop_nums = collect_numbers(coop_stmts)
    defect_nums = collect_numbers(defect_stmts)
    if not coop_nums or not defect_nums: return None
    # Take the FIRST non-trivial positive number from each branch
    # as the canonical "delta" value
    pos = coop_nums[0]
    neg = defect_nums[0]
    return pos, neg


# Run analysis
buckets = Counter()
per_llm = defaultdict(Counter)
per_obs = defaultdict(Counter)
extracted = 0
unknown = 0
asym_examples = []

for s in all_strats:
    deltas = extract_deltas_v2(s["code"])
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
        cls = "asym_pos_bigger"  # reward > punishment
    else:
        cls = "asym_neg_bigger"  # punishment > reward
    buckets[cls] += 1
    per_llm[s["llm"]][cls] += 1
    per_obs[s["obs"]][cls] += 1
    if cls.startswith("asym") and len(asym_examples) < 6:
        asym_examples.append((s["llm"], s["obs"], pos, neg, cls, s["coop"], s["fit"]))

print(f"\nExtracted from {extracted} / {len(all_strats)} strategies ({100*extracted/len(all_strats):.1f}%)")
print(f"Unknown: {unknown}")

print(f"\n=== Overall asymmetry distribution (over {extracted} extractable) ===")
for cls, count in buckets.most_common():
    print(f"  {cls:30s}  {count:4d}  {100*count/extracted:5.1f}%")

# Highlight
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
    sym, apos, aneg, rev, np_, unk = c["symmetric"], c["asym_pos_bigger"], c["asym_neg_bigger"], c["reverse"], c["no_punishment"], c["unknown"]
    print(f"{llm[:25]:25s}  {sym:5d}  {apos:10d}  {aneg:10d}  {rev:8d}  {np_:7d}  {unk:5d}  {total:6d}")
    valid = sym + apos + aneg
    if valid > 0:
        print(f"{'':25s}    asym share within valid: {100*(apos+aneg)/valid:.1f}%   (pos>neg: {100*apos/valid:.1f}%, neg>pos: {100*aneg/valid:.1f}%)")

print(f"\n=== Per observability ===")
print(f"{'obs':15s}  {'sym':>5s}  {'asym_pos>':>10s}  {'asym_neg>':>10s}  {'reverse':>8s}  {'no_pun':>7s}  {'unk':>5s}  {'total':>6s}")
for obs in sorted(per_obs.keys()):
    c = per_obs[obs]
    total = sum(c.values())
    sym, apos, aneg, rev, np_, unk = c["symmetric"], c["asym_pos_bigger"], c["asym_neg_bigger"], c["reverse"], c["no_punishment"], c["unknown"]
    print(f"{obs[:15]:15s}  {sym:5d}  {apos:10d}  {aneg:10d}  {rev:8d}  {np_:7d}  {unk:5d}  {total:6d}")
    valid = sym + apos + aneg
    if valid > 0:
        print(f"{'':15s}    asym share within valid: {100*(apos+aneg)/valid:.1f}%")

print(f"\n=== Sample asymmetric deltas ===")
for llm, obs, pos, neg, cls, coop, fit in asym_examples:
    ratio = pos / abs(neg) if neg != 0 else 0
    print(f"  {llm[:25]:25s} obs={obs:12s}  delta_pos=+{pos:.2f}  delta_neg={neg:.2f}  ratio={ratio:.2f}  [{cls}]  coop={coop:.2f}  fit={fit:.1f}")
