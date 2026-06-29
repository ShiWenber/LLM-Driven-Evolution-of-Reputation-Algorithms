"""4 exp sweep — algorithmic complexity ceiling probes (v16).

Each exp tests one hypothesis about why evolved strategies stay simple.
Default: 3 trials per (config, obs) to control cost; full obs + partial 0.7.

Concurrent 5 workers per exp, run sequentially (so we can monitor each).
Total wall-clock: ~6h (LLM cost ~$2.50).
"""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUTPUT_ROOT = Path(REPO) / 'results' / 'exp7_algorithmic_ceiling'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Each entry: (exp_name, subdir, extra_args, obs_list, n_seeds)
# obs_list = ['full', 'partial_0.7']
# n_seeds = 3 per obs
EXPS = [
    (
        'A_budget',
        'A_larger_budget',
        ['--population', '30', '--generations', '20', '--rounds', '50',
         '--elitism', '4', '--tournament', '3', '--eliminate', '8'],
        ['full', 'partial_0.7'],
        3,
    ),
    (
        'B_window',
        'B_recent_window',
        ['--population', '15', '--generations', '10', '--rounds', '30',
         '--elitism', '2', '--tournament', '3', '--eliminate', '5',
         '--recent-window', '5'],
        ['full', 'partial_0.7'],
        3,
    ),
    (
        'C_noise',
        'C_reputation_noise',
        ['--population', '15', '--generations', '10', '--rounds', '30',
         '--elitism', '2', '--tournament', '3', '--eliminate', '5',
         '--reputation-noise', '0.1'],
        ['full', 'partial_0.7'],
        3,
    ),
    (
        'D_explore',
        'D_exploration_mutation',
        ['--population', '15', '--generations', '10', '--rounds', '30',
         '--elitism', '2', '--tournament', '3', '--eliminate', '5',
         '--exploration-mutation'],
        ['full', 'partial_0.7'],
        3,
    ),
]

# Optional: combined window+noise+exploration — the "kitchen sink" test
EXPS_COMBINED = [
    (
        'E_kitchen',
        'E_all_combined',
        ['--population', '15', '--generations', '10', '--rounds', '30',
         '--elitism', '2', '--tournament', '3', '--eliminate', '5',
         '--recent-window', '5', '--reputation-noise', '0.1',
         '--exploration-mutation'],
        ['full', 'partial_0.7'],
        3,
    ),
]


def run_one(exp_name, subdir, extra, obs, seed):
    outdir = OUTPUT_ROOT / subdir / f'{obs}_seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
        '--observability', obs, '--seeds', '1', '--output', str(outdir),
        '--models', 'deepseek-v4-flash',
    ] + extra
    env = {**os.environ, 'PYTHONHASHSEED': str(seed * 1000 + 1), 'PYTHONUNBUFFERED': '1'}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=2400)
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        # Try to find trajectory
        coop_final = None
        for f in os.listdir(outdir):
            if f.startswith('evo_') and f.endswith('.json'):
                with open(outdir / f) as fp:
                    t = json.load(fp)
                if t.get('trajectory'):
                    coop_final = t['trajectory'][-1]['cooperation_rate_mean']
                break
        return {
            'exp': exp_name, 'subdir': subdir, 'obs': obs, 'seed': seed,
            'ok': ok, 'elapsed_sec': round(elapsed, 1),
            'coop_final': coop_final,
        }
    except subprocess.TimeoutExpired:
        return {'exp': exp_name, 'obs': obs, 'seed': seed, 'ok': False, 'error': 'timeout'}


def run_all_exps():
    all_results = []
    overall_start = time.time()
    for exp_name, subdir, extra, obs_list, n_seeds in EXPS:
        print(f'\n{"="*60}\n  EXP {exp_name}: {subdir}\n{"="*60}')
        for obs in obs_list:
            print(f'\n  Obs: {obs}')
            for seed in range(n_seeds):
                t0 = time.time()
                r = run_one(exp_name, subdir, extra, obs, seed)
                print(f'    [{seed+1}/{n_seeds}] obs={obs} seed={seed} '
                      f'-> ok={r.get("ok")} coop={r.get("coop_final")} '
                      f'elapsed={r.get("elapsed_sec", 0):.0f}s '
                      f'(total wall: {(time.time() - overall_start)/60:.0f} min)')
                all_results.append(r)
                (OUTPUT_ROOT / '_manifest.json').write_text(
                    json.dumps(all_results, indent=2), encoding='utf-8'
                )
    return all_results


def run_kitchen_sink():
    """E exp: all 3 features combined (kitchen sink)."""
    print(f'\n{"="*60}\n  EXP E (kitchen sink: window + noise + exploration)\n{"="*60}')
    all_results = []
    overall_start = time.time()
    for exp_name, subdir, extra, obs_list, n_seeds in EXPS_COMBINED:
        for obs in obs_list:
            print(f'\n  Obs: {obs}')
            for seed in range(n_seeds):
                r = run_one(exp_name, subdir, extra, obs, seed)
                print(f'    [{seed+1}/{n_seeds}] obs={obs} seed={seed} '
                      f'-> ok={r.get("ok")} coop={r.get("coop_final")} '
                      f'elapsed={r.get("elapsed_sec", 0):.0f}s '
                      f'(total wall: {(time.time() - overall_start)/60:.0f} min)')
                all_results.append(r)
                (OUTPUT_ROOT / '_manifest.json').write_text(
                    json.dumps(all_results, indent=2), encoding='utf-8'
                )
    return all_results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--phase', default='ab', help='ab=Exp A+B, cd=Exp C+D, e=Exp E only, all=everything')
    args = p.parse_args()
    if args.phase in ('ab', 'all'):
        run_all_exps()
    if args.phase in ('cd', 'all'):
        run_all_exps()
    if args.phase in ('e', 'all'):
        run_kitchen_sink()
    print('\nAll done.')