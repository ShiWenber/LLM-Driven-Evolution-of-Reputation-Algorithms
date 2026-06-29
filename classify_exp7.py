"""Classify all exp7 final-population strategies with extended taxonomy:
- F1-F8 (existing) from classify_cd_strategies
- New: 'window-user' (uses recent_window), 'EMA-style' (uses target + alpha)
- New: 'multi-cond' (3+ conditions in decide), 'iter' (for loop), 'numpy' (np)
- New: 'RL-like' (has q_value / td / discount / alpha*target + (1-alpha)*current)
"""
import json, os, re
from collections import Counter, defaultdict

base = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\exp7_algorithmic_ceiling'

EXPS = {
    'A_larger_budget': ('A budget (N=30,G=20)', 'exp', 'A_budget'),
    'B_recent_window': ('B recent-window=5', 'exp', 'B_window'),
    'C_reputation_noise': ('C reputation noise=0.1', 'exp', 'C_noise'),
    'D_exploration_mutation': ('D exploration-mutation', 'exp', 'D_explore'),
    'E_all_combined': ('E all combined', 'exp', 'E_kitchen'),
}

OBS_LIST = ['full', 'partial_0.7']

def classify(code):
    """Return (family_short, family_long, has_window, has_iter, has_np, has_decay)."""
    d = code
    has_window = 'recent_window' in d
    has_iter = bool(re.search(r'\bfor\s+\w+\s+in\s+', d))
    has_np = bool(re.search(r'\bnp\.', d))
    has_decay = bool(re.search(r'decay|discount|forget|tau|lambda|alpha|beta|gamma|eta|EMA|ema', d))
    has_q = bool(re.search(r'\bq_|qvalue|td_target|td_error|reward|policy', d, re.IGNORECASE))
    has_ucb = bool(re.search(r'ucb|upper.confidence|exploit|explore|sigmoid|softmax', d, re.IGNORECASE))
    has_threshold = re.search(r'recipient_reputation\s*[><=!]+\s*[-\d.]+', d) is not None
    has_my_history = 'my_history' in d
    n_if = len(re.findall(r'\bif\b', d))
    has_dynamic_thr = bool(re.search(r'\bthreshold\s*=', d))
    has_return_true = re.search(r'return\s+True\b', d) is not None
    has_return_false = re.search(r'return\s+False\b', d) is not None

    # Family classification
    if has_return_true and not has_threshold and not has_dynamic_thr and not has_my_history:
        family = 'F5a: always-True'
    elif has_return_false and not has_threshold and not has_dynamic_thr and not has_my_history:
        family = 'F5b: always-False'
    elif has_threshold and has_dynamic_thr and has_my_history:
        family = 'F4: dyn-thr + hist'
    elif has_threshold and has_dynamic_thr:
        family = 'F3: dyn-thr + round'
    elif has_threshold and has_my_history:
        family = 'F2: thr + hist'
    elif has_threshold:
        family = 'F1: simple thr'
    elif not has_threshold and has_my_history:
        family = 'F2b: hist only'
    elif n_if >= 2:
        family = 'F7: complex'
    else:
        family = 'F8: other'

    return {
        'family': family, 'has_window': has_window, 'has_iter': has_iter,
        'has_np': has_np, 'has_decay': has_decay, 'has_q': has_q,
        'has_ucb': has_ucb, 'n_if': n_if,
    }


results = {}
for subdir, (label, _, _) in EXPS.items():
    results[subdir] = {'label': label, 'per_obs': {}}
    for obs in OBS_LIST:
        per_seed = []
        for seed in range(3):
            d = os.path.join(base, subdir, f'{obs}_seed{seed}')
            files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
            if not files: continue
            files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
            with open(os.path.join(d, files[0])) as f:
                t = json.load(f)
            coop = t['trajectory'][-1]['cooperation_rate_mean'] if t['trajectory'] else None
            classes = [classify(a['code']) for a in t['final_population']]
            per_seed.append({
                'coop': coop, 'n': len(classes), 'classes': classes,
            })
        results[subdir]['per_obs'][obs] = per_seed

# Print summary
print('=== EXP7 ALGORITHMIC-COMPLEXITY-CEILING SUMMARY ===\n')
for subdir, info in results.items():
    label = info['label']
    print(f'\n--- {label} ---')
    for obs in OBS_LIST:
        per_seed = info['per_obs'][obs]
        if not per_seed:
            print(f'  {obs}: no data')
            continue
        coops = [s['coop'] for s in per_seed if s['coop'] is not None]
        all_classes = [c for s in per_seed for c in s['classes']]
        n = len(all_classes)
        fc = Counter(c['family'] for c in all_classes)
        n_window = sum(1 for c in all_classes if c['has_window'])
        n_iter = sum(1 for c in all_classes if c['has_iter'])
        n_np = sum(1 for c in all_classes if c['has_np'])
        n_decay = sum(1 for c in all_classes if c['has_decay'])
        n_q = sum(1 for c in all_classes if c['has_q'])
        n_ucb = sum(1 for c in all_classes if c['has_ucb'])
        n_ok = sum(1 for c in coops if c > 0.5)
        mean_coop = sum(coops) / len(coops)
        print(f'  {obs}: mean={mean_coop:.3f} ({n_ok}/{len(coops)} OK), '
              f'n_agents={n}, window={n_window}/{n}, iter={n_iter}/{n}, '
              f'np={n_np}/{n}, decay={n_decay}/{n}, q={n_q}/{n}, ucb={n_ucb}/{n}')
        # Top family
        top = fc.most_common(3)
        print(f'    top families: {top}')