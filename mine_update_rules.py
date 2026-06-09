"""Mine the actual reputation-update rules the LLMs discovered across all 36+6 trials.

For each Hybrid strategy, extract:
- delta on donate (positive update amount)
- delta on defect (negative update amount)
- clamp range
- use of my_history in evaluate (e.g., self-modulation, generosity_boost)
- use of round_num modulation
- presence of EMA / decay factor

Output a markdown report comparing the two LLMs' discovered update rules.
"""
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

EXPERIMENTS = {
    'deepseek-v4-flash (Standard)': Path('results/exp1_method'),
    'deepseek-coder (Robustness)': Path('results/exp5_robustness'),
}


def is_hybrid(code: str) -> bool:
    return bool(re.search(r"observation\s*\[\s*['\"]action['\"]\s*\]\s*==\s*['\"]donate['\"]", code)) \
        and bool(re.search(r"recipient_reputation\s*[><=!]+", code)) \
        and bool(re.search(r"my_history", code))


def extract_update_rule(code: str) -> dict:
    """Try to extract a simple structured description of the evaluate() update rule."""
    info = {
        'donate_delta': None,
        'defect_delta': None,
        'uses_ema': False,
        'uses_round_factor': False,
        'uses_self_history': False,
        'uses_global_counter': False,
        'clamp_range': '[-1, 1]' if 'max(-1.0' in code and 'min(1.0' in code else None,
        'asymmetric': False,
    }
    # donate delta (look for patterns like "if ... == 'donate': return ... + X" or "... + X")
    m = re.search(r"action\s*==\s*['\"]donate['\"][^a-zA-Z][^)]*?([+]\s*([\d.]+))", code)
    if m:
        info['donate_delta'] = float(m.group(2))
    # defect delta
    m = re.search(r"else\s*:[^a-zA-Z]*?([-]\s*([\d.]+))", code)
    if m:
        info['defect_delta'] = float(m.group(2))
    # Also look for "if donate: ... = current_reputation * 0.9 + X"
    m = re.search(r"donate['\"][^)]*?\*\s*([\d.]+)\s*\+\s*([\d.]+)", code)
    if m:
        info['uses_ema'] = True
        info['ema_decay'] = 1 - float(m.group(1))
        # Recompute donate_delta as effective delta at r=0
        if info['donate_delta'] is None:
            info['donate_delta'] = float(m.group(2))
    # Asymmetric?
    if info['donate_delta'] is not None and info['defect_delta'] is not None:
        if abs(info['donate_delta']) != abs(info['defect_delta']):
            info['asymmetric'] = True
    # round factor
    if 'round_num' in code and re.search(r"round_num[^a-zA-Z]*?[/+\-*]", code):
        info['uses_round_factor'] = True
    # self-history modulation in evaluate
    if re.search(r"def\s+evaluate\([^)]*my_history", code) and re.search(r"my_history", code.split('def decide')[0] if 'def decide' in code else code):
        info['uses_self_history'] = True
    # global counter (list-of-int pattern)
    if re.search(r"^[\w_]+\s*=\s*\[\s*0\s*\]", code, re.MULTILINE):
        info['uses_global_counter'] = True
    return info


# Collect all Hybrid strategies from each experiment
all_data = {}
for label, root in EXPERIMENTS.items():
    if not root.exists():
        continue
    records = []
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
                rule = extract_update_rule(code)
                records.append({
                    'trial': trial_dir.name,
                    'agent_id': a.get('agent_id'),
                    'cooperation_rate': a.get('cooperation_rate'),
                    'fitness': a.get('fitness'),
                    'rule': rule,
                })
    all_data[label] = records

# Print summary
for label, recs in all_data.items():
    print(f"=== {label}: {len(recs)} Hybrid strategies ===\n")
    if not recs:
        continue
    # Aggregate statistics
    d_deltas = [r['rule']['donate_delta'] for r in recs if r['rule']['donate_delta'] is not None]
    x_deltas = [r['rule']['defect_delta'] for r in recs if r['rule']['defect_delta'] is not None]
    ema_count = sum(1 for r in recs if r['rule']['uses_ema'])
    asymm_count = sum(1 for r in recs if r['rule']['asymmetric'])
    round_count = sum(1 for r in recs if r['rule']['uses_round_factor'])
    hist_count = sum(1 for r in recs if r['rule']['uses_self_history'])
    counter_count = sum(1 for r in recs if r['rule']['uses_global_counter'])
    if d_deltas:
        print(f"  Donate delta (positive update):")
        print(f"    median={sorted(d_deltas)[len(d_deltas)//2]:.3f}  range=[{min(d_deltas):.3f}, {max(d_deltas):.3f}]")
        print(f"    distribution: {Counter([round(d, 1) for d in d_deltas]).most_common()}")
    if x_deltas:
        print(f"  Defect delta (negative update):")
        print(f"    median={sorted(x_deltas)[len(x_deltas)//2]:.3f}  range=[{min(x_deltas):.3f}, {max(x_deltas):.3f}]")
        print(f"    distribution: {Counter([round(d, 1) for d in x_deltas]).most_common()}")
    print(f"  Asymmetric delta: {asymm_count}/{len(recs)} ({100*asymm_count/len(recs):.1f}%)")
    print(f"  Uses EMA (current_reputation * k + delta): {ema_count}/{len(recs)} ({100*ema_count/len(recs):.1f}%)")
    print(f"  Uses round_num modulation in evaluate(): {round_count}/{len(recs)} ({100*round_count/len(recs):.1f}%)")
    print(f"  Uses my_history in evaluate(): {hist_count}/{len(recs)} ({100*hist_count/len(recs):.1f}%)")
    print(f"  Uses global counter (mutable list): {counter_count}/{len(recs)} ({100*counter_count/len(recs):.1f}%)")
    print()
    # Show the top 3 by cooperation rate
    print("  Top 3 by cooperation rate (showing the update formula):")
    recs.sort(key=lambda r: r['cooperation_rate'] or 0, reverse=True)
    for r in recs[:3]:
        print(f"    {r['trial']} a={r['agent_id']} coop={r['cooperation_rate']:.3f} fit={r['fitness']:.1f} -> {r['rule']}")
    print()
