"""Look at the actual update rules more carefully.

Read all Hybrid strategies, group by structurally distinct update rules,
report the canonical formula and frequency.
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import ast

EXPERIMENTS = {
    'deepseek-v4-flash (Standard)': Path('results/exp1_method'),
    'deepseek-coder (Robustness)': Path('results/exp5_robustness'),
}


def is_hybrid(code: str) -> bool:
    return bool(re.search(r"observation\s*\[\s*['\"]action['\"]\s*\]\s*==\s*['\"]donate['\"]", code)) \
        and bool(re.search(r"recipient_reputation\s*[><=!]+", code)) \
        and bool(re.search(r"my_history", code))


def extract_evaluate_body(code: str) -> str:
    """Get the body of the evaluate() function as text."""
    m = re.search(r"def\s+evaluate\s*\([^)]*\)\s*:", code)
    if not m:
        return ''
    start = m.end()
    # Find matching indent level
    lines = code[start:].split('\n')
    body_lines = []
    base_indent = None
    for line in lines:
        if not line.strip():
            continue
        if base_indent is None:
            stripped = line.lstrip()
            if not (line.startswith(' ') or line.startswith('\t')):
                return '\n'.join(body_lines)
            base_indent = len(line) - len(stripped)
            if stripped.startswith(('return', '#', '"""', "'''", 'pass')) and 'return' in stripped:
                body_lines.append(line)
                continue
        if not line.strip():
            body_lines.append(line)
            continue
        cur_indent = len(line) - len(line.lstrip())
        if cur_indent < base_indent:
            break
        body_lines.append(line)
    return '\n'.join(body_lines)


def normalise_eval(code: str) -> str:
    """Extract a structural signature of the evaluate() function.

    We try to capture the mathematical form rather than exact code.
    """
    body = extract_evaluate_body(code)
    if not body:
        return 'no-body'
    # Strip comments and string literals to focus on structure
    s = re.sub(r'#.*', '', body)
    s = re.sub(r'\"\"\".*?\"\"\"', '', s, flags=re.DOTALL)
    s = re.sub(r"'''.*?'''", '', s, flags=re.DOTALL)
    # Detect: is there a 'return current_reputation + X' pattern?
    is_simple_addition = bool(re.search(r"return\s+current_reputation\s*[+\-]\s*[\d.]+", s))
    # is there an EMA: current_reputation * X + Y or current_reputation * X - Y?
    is_ema = bool(re.search(r"return\s+max\(.*?min\(.*?current_reputation\s*\*\s*[\d.]+\s*[+\-]", s))
    # is there a clamp to [-1, 1]?
    is_clamp = 'max(-1' in s and 'min(1' in s
    # Is there a 'donate' / 'else' branch?
    has_branch = "donate" in s and 'else' in s
    # Is there a counter / list mutation?
    has_counter = bool(re.search(r"^\s*[\w_]+\s*\[0\]\s*[+\-]=\s*1", s, re.MULTILINE)) \
        or bool(re.search(r"\b\w+\s*\[0\]\s*=\s*\w+\s*\[0\]\s*\+\s*1", s))
    # Is there a round-modulation factor?
    has_round_mod = bool(re.search(r"round_num", s)) and bool(re.search(r"[/+\-*]\s*[\d.]+", s))
    # Is there my_history usage in evaluate?
    has_history = 'my_history' in s

    sig = []
    if is_ema: sig.append('ema')
    if is_simple_addition: sig.append('additive')
    if has_branch: sig.append('if-donate')
    if has_counter: sig.append('counter')
    if has_round_mod: sig.append('round-mod')
    if has_history: sig.append('self-hist')
    if is_clamp: sig.append('clamp-1')
    return '+'.join(sig) if sig else 'unknown'


# Now collect all distinct signatures
for label, root in EXPERIMENTS.items():
    if not root.exists():
        continue
    print(f"=== {label} ===\n")
    sigs = []
    full_codes_by_sig = defaultdict(list)
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
                sig = normalise_eval(code)
                sigs.append(sig)
                if len(full_codes_by_sig[sig]) < 1:
                    full_codes_by_sig[sig].append({
                        'code': code,
                        'trial': trial_dir.name,
                        'coop': a.get('cooperation_rate'),
                        'fit': a.get('fitness'),
                    })
    counter = Counter(sigs)
    total = len(sigs)
    print(f"  Total Hybrid strategies: {total}")
    print(f"  Distinct structural signatures: {len(counter)}")
    print()
    print(f"  Top 8 structural signatures (out of {total} total):")
    for sig, n in counter.most_common(8):
        pct = 100 * n / total
        ex = full_codes_by_sig[sig][0]
        print(f"    {n:3d} ({pct:5.1f}%)  {sig}")
        print(f"          example: {ex['trial']} coop={ex['coop']:.3f} fit={ex['fit']:.1f}")
    print()
