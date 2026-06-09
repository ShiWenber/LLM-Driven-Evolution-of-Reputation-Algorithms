"""Simpler: just look at the actual Hybrid evaluate() bodies and cluster them.

Focus on identifying the *families* of reputation-update rules the LLMs
discovered, not exact counts.
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

EXPERIMENTS = {
    'deepseek-v4-flash (Standard)': Path('results/exp1_method'),
    'deepseek-coder (Robustness)': Path('results/exp5_robustness'),
}


def is_hybrid(code: str) -> bool:
    return bool(re.search(r"observation\s*\[\s*['\"]action['\"]\s*\]\s*==\s*['\"]donate['\"]", code)) \
        and bool(re.search(r"recipient_reputation\s*[><=!]+", code)) \
        and bool(re.search(r"my_history", code))


def get_evaluate_lines(code: str) -> str:
    """Get all lines belonging to the evaluate() function, stripped of comments."""
    lines = code.split('\n')
    in_eval = False
    eval_indent = None
    out = []
    for line in lines:
        m = re.match(r'^(\s*)def\s+evaluate\b', line)
        if m:
            in_eval = True
            eval_indent = len(m.group(1))
            continue
        if in_eval:
            if not line.strip():
                out.append(line)
                continue
            cur_indent = len(line) - len(line.lstrip())
            if line.lstrip().startswith('def '):
                break
            if cur_indent <= eval_indent and line.strip():
                break
            # Strip comments
            clean = re.sub(r'#.*', '', line)
            out.append(clean)
    return '\n'.join(out)


def fingerprint(eval_body: str) -> str:
    """Build a structural fingerprint of the evaluate body."""
    if not eval_body.strip():
        return 'EMPTY'
    s = eval_body
    feats = []
    # Branch on observation['action']?
    if re.search(r"observation\s*\[\s*['\"]action['\"]\s*\]\s*==\s*['\"]donate['\"]", s):
        feats.append('branch-donate')
    # Look for arithmetic operations on current_reputation
    # Cases:
    # A) Additive: r + X
    add = re.findall(r"current_reputation\s*\+\s*([\d.]+)", s)
    sub = re.findall(r"current_reputation\s*-\s*([\d.]+)", s)
    # B) EMA: r * k + X
    ema = re.findall(r"current_reputation\s*\*\s*([\d.]+)", s)
    # C) Reset (r = X) (no current_reputation in RHS)
    if not add and not sub and not ema and re.search(r"return\s+[\d.]+|return\s+current_reputation\b", s):
        feats.append('constant-or-passthrough')
    if add:
        feats.append(f"add({','.join(sorted(set(add)))})")
    if sub:
        feats.append(f"sub({','.join(sorted(set(sub)))})")
    if ema:
        feats.append(f"ema({','.join(sorted(set(ema)))})")
    # Round modulation?
    if 'round_num' in s and re.search(r"round_num\s*[/+\-*]", s):
        feats.append('round-mod')
    # Clamp to [-1, 1]?
    if 'max(-1' in s and 'min(1' in s:
        feats.append('clamp[-1,1]')
    # Uses my_history in evaluate?
    if 'my_history' in s:
        feats.append('self-hist')
    # Counter / mutable list pattern?
    if re.search(r"\b\w+\s*\[0\]\s*[+\-]=\s*1", s):
        feats.append('counter')
    return '|'.join(feats) if feats else 'OTHER'


# Collect fingerprints
for label, root in EXPERIMENTS.items():
    if not root.exists():
        continue
    print(f"\n=== {label} ===\n")
    fps = []
    for trial_dir in sorted(root.iterdir()):
        if not trial_dir.is_dir():
            continue
        for f in trial_dir.glob('evo_*.json'):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            for a in d.get('final_population', []):
                code = a.get('code', '')
                if not is_hybrid(code):
                    continue
                body = get_evaluate_lines(code)
                fp = fingerprint(body)
                fps.append((fp, body, trial_dir.name, a.get('cooperation_rate'), a.get('fitness')))
    counter = Counter(fp for fp, *_ in fps)
    total = len(fps)
    print(f"Total Hybrid: {total}")
    print(f"Distinct fingerprints: {len(counter)}\n")
    for fp, n in counter.most_common():
        pct = 100 * n / total
        # Find an example with the highest cooperation rate
        matches = [r for r in fps if r[0] == fp]
        matches.sort(key=lambda r: r[3] or 0, reverse=True)
        ex = matches[0]
        print(f"  {n:3d} ({pct:5.1f}%)  [{fp}]")
        print(f"        example {ex[2]} coop={ex[3]:.3f} fit={ex[4]:.1f}:")
        # Show just the body (truncated)
        for line in ex[1].split('\n')[:10]:
            if line.strip():
                print(f"          {line.rstrip()}")
        print()
