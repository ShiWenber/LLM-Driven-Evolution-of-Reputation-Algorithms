"""v24: pair evo files with reasoning files by content overlap.
The 'on' trial's reasoning entries contain explicit LLM reasoning content
(reasoning != empty); the 'off' trial wouldn't have reasoning captured
but since both ran with --enable-thinking in this experiment (no, only on did),
so we can pair by which evo file's timestamp window matches the reasoning entries' timestamps.
"""
import json
import time
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp11_thinking_compare')

# Group evo files by obs - filename pattern: evo_<obs>_<model>_<ts>.json
# where obs may contain underscores (partial_0.3, partial_0.7)
evo_by_obs = {}
for evo in RES.glob('evo_*.json'):
    name = evo.name  # e.g. 'evo_partial_0.3_deepseek-v4-flash_20260713_171228.json'
    rest = name.replace('evo_', '').replace('.json', '')
    # strip trailing timestamp _YYYYMMDD_HHMMSS
    # also strip model name "deepseek-v4-flash" (always)
    parts_no_ts = rest.rsplit('_', 2)[0]  # remove last 2 underscore parts (date, time)
    obs = parts_no_ts.replace('_deepseek-v4-flash', '')
    evo_by_obs.setdefault(obs, []).append(evo)

# For each obs: check if the reasoning file is non-trivial (i.e. has actual reasoning content)
results = {}  # (obs, mode) -> evo_path
for obs, evos in evo_by_obs.items():
    if len(evos) != 2:
        print(f'WARN: obs={obs} has {len(evos)} evo files, expected 2')
    r_path = RES / f'{obs}_seed0' / f'reasoning_deepseek-v4-flash_{obs}_seed0.json'
    has_reasoning = False
    if r_path.exists():
        log = json.loads(r_path.read_text(encoding='utf-8', errors='ignore'))
        has_reasoning = any(e.get('reasoning', '') for e in log)
    # Match by mtime: the 'on' trial's evo file was created around the same time
    # as the reasoning file (within a few minutes). The 'off' trial's evo is older.
    if has_reasoning and r_path.exists():
        r_mtime = r_path.stat().st_mtime
        for evo in evos:
            e_mtime = evo.stat().st_mtime
            # The on trial's evo was written AFTER the off trial (off ran first in our plan),
            # but actually we ran on first, so on's evo is OLDER
            # Actually our plan ran [on, off] in sequence. on is at start of obs, off later.
            # So on evo mtime < off evo mtime < reasoning mtime
            # Find the on trial as the one with smaller mtime gap to reasoning
            gap = abs(e_mtime - r_mtime)
            print(f'  {obs} {evo.name}: mtime_gap_to_reasoning={gap:.0f}s')
            # The on trial is closer to reasoning file (similar timing)
            # Actually, since reasoning file is written at end of on trial,
            # and off trial is run AFTER on, off's mtime is later
            # So on is the one with smaller mtime
            # Use heuristic: smaller mtime = on
        # Sort by mtime, first is on
        sorted_evos = sorted(evos, key=lambda p: p.stat().st_mtime)
        results[(obs, 'on')] = sorted_evos[0]
        results[(obs, 'off')] = sorted_evos[1]
    else:
        for evo in evos:
            results[(obs, 'off')] = evo

# Print results
print('=== Pairs found ===')
for (obs, mode), path in sorted(results.items()):
    print(f'  {obs:>12s} thinking={mode}: {path.name}')

# Now run the analysis
def load_trial(path):
    d = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    fp = d.get('final_population') or []
    if not isinstance(fp, list): return None
    traj = d.get('trajectory', [])
    if not traj: return None
    return d, fp, traj

print('\n=== Final cooperation by trial ===')
print(f'{"obs":>12s} {"mode":>10s} {"final_coop":>10s} {"fit_mean":>10s}')
for (obs, mode), td in sorted(results.items()):
    trial = load_trial(td)
    if not trial: continue
    d, fp, traj = trial
    fg = traj[-1]
    coop = fg.get('cooperation_rate_mean', 0) or 0
    fit = fg.get('fitness_mean', 0) or 0
    print(f'{obs:>12s} {mode:>10s} {coop:>10.3f} {fit:>10.1f}')

print('\n=== Cooperation by (obs, mode) ===')
print(f'{"obs":>12s} {"thinking=on":>15s} {"thinking=off":>15s} {"delta":>8s}')
for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
    on_p = results.get((obs, 'on'))
    off_p = results.get((obs, 'off'))
    on_c = 0; off_c = 0
    if on_p:
        t = load_trial(on_p)
        if t: on_c = t[2][-1].get('cooperation_rate_mean', 0) or 0
    if off_p:
        t = load_trial(off_p)
        if t: off_c = t[2][-1].get('cooperation_rate_mean', 0) or 0
    print(f'{obs:>12s} {on_c:>15.3f} {off_c:>15.3f} {on_c - off_c:>+8.3f}')

print('\n=== Strategy class distribution ===')
def classify(code):
    if 'def evaluate' not in code: return '?'
    import re
    m = re.search(r'def evaluate\([^)]*\):(.*?)(?=\ndef |\Z)', code, re.S)
    if not m: return '?'
    body = m.group(1)
    # Mirror paper classifier: Hybrid = uses observation[action] AND threshold AND my_history
    has_action_check = bool(re.search(r"observation\[.+\baction\b", body))
    has_threshold = bool(re.search(r"recipient_reputation\s*[><=!]+", body))
    has_my_history = 'my_history' in body
    if has_action_check and has_threshold and has_my_history:
        return 'Hybrid'
    if has_action_check and has_threshold:
        return 'ImageScoring'
    return 'Other'

def uses_recipient_rep_in_evaluate(code):
    """Mirror paper's check: does evaluate() body reference recipient_reputation?"""
    if 'def evaluate' not in code: return False
    import re
    m = re.search(r'def evaluate\([^)]*\):(.*?)(?=\ndef |\Z)', code, re.S)
    if not m: return False
    return 'recipient_reputation' in m.group(1)

for mode in ['on', 'off']:
    print(f'\n  --- thinking={mode} ---')
    for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
        path = results.get((obs, mode))
        if not path: continue
        t = load_trial(path)
        if not t: continue
        d, fp, _ = t
        total = h = rec = 0
        for a in fp:
            if not isinstance(a, dict): continue
            code = a.get('code', '')
            if not code: continue
            total += 1
            if classify(code) == 'Hybrid': h += 1
            if uses_recipient_rep_in_evaluate(code): rec += 1
        print(f'    {obs:>12s}: total={total} Hybrid={h} ({100*h/max(1,total):.0f}%) uses_rec={rec} ({100*rec/max(1,total):.0f}%)')

print('\n=== Reasoning-trace count (thinking=on) ===')
total_r = 0
for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
    r_path = RES / f'{obs}_seed0' / f'reasoning_deepseek-v4-flash_{obs}_seed0.json'
    if r_path.exists():
        log = json.loads(r_path.read_text(encoding='utf-8', errors='ignore'))
        n_with = sum(1 for e in log if e.get('reasoning'))
        print(f'  {obs}: {len(log)} entries, {n_with} with non-empty reasoning')
        total_r += len(log)
print(f'  TOTAL: {total_r}')
