"""Intern cross-LLM: 4 obs × 3 seeds × 4 probe = 48 trials.
Probe A: N=30, G=20, T=50
Probe B: recent_window=5
Probe C: reputation noise=0.1
Probe D: exploration mutation
Probe: baseline (N=15, G=10, no extra feature)
"""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = Path(REPO) / 'results' / 'exp8_intern_ceiling'
OUT.mkdir(parents=True, exist_ok=True)

PROBES = {
    'A_larger_budget':    ['--population', '30', '--generations', '20', '--rounds', '50',
                            '--elitism', '4', '--tournament', '3', '--eliminate', '8'],
    'B_recent_window':    ['--population', '15', '--generations', '10', '--rounds', '30',
                            '--elitism', '2', '--tournament', '3', '--eliminate', '5',
                            '--recent-window', '5'],
    'C_reputation_noise': ['--population', '15', '--generations', '10', '--rounds', '30',
                            '--elitism', '2', '--tournament', '3', '--eliminate', '5',
                            '--reputation-noise', '0.1'],
    'D_exploration':      ['--population', '15', '--generations', '10', '--rounds', '30',
                            '--elitism', '2', '--tournament', '3', '--eliminate', '5',
                            '--exploration-mutation'],
    'E_baseline':         ['--population', '15', '--generations', '10', '--rounds', '30',
                            '--elitism', '2', '--tournament', '3', '--eliminate', '5'],
}
OBS_LIST = ['full', 'partial_0.7']  # 2 obs (most informative, matches exp7)
N_SEEDS = 3

# Total trials: 5 probes × 2 obs × 3 seeds = 30 trials (not 48 — I cut to 2 obs to save time)
PLAN = []
for pname, pargs in PROBES.items():
    for obs in OBS_LIST:
        for seed in range(N_SEEDS):
            PLAN.append((pname, obs, seed, pargs))

print(f'Total trials: {len(PLAN)}')
print(f'Estimated wall-clock: ~{len(PLAN) * 12 / 60:.1f} h (sequential, 4 workers per trial)\n')

start_all = time.time()
manifest = []
for i, (pname, obs, seed, pargs) in enumerate(PLAN, 1):
    outdir = OUT / f'{pname}' / f'{obs}_seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
        '--observability', obs, '--seeds', '1', '--output', str(outdir),
        '--models', 'paratera-intern',
    ] + pargs
    env = {**os.environ, 'PYTHONHASHSEED': str(seed*1000+1), 'PYTHONUNBUFFERED': '1'}
    t0 = time.time()
    print(f'[{i}/{len(PLAN)}] {pname:18s} {obs:12s} seed{seed} | starting...', flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=2400)
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        coop = None
        for f in os.listdir(outdir):
            if f.startswith('evo_') and f.endswith('.json'):
                with open(outdir / f) as fp:
                    t = json.load(fp)
                if t.get('trajectory'):
                    coop = t['trajectory'][-1]['cooperation_rate_mean']
                break
        manifest.append({'probe': pname, 'obs': obs, 'seed': seed, 'ok': ok,
                          'elapsed_sec': round(elapsed, 1), 'coop_final': coop})
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.0f}s | coop={coop} '
              f'| total wall: {(time.time()-start_all)/3600:.2f}h')
        if not ok:
            print(f'  stderr: {proc.stderr[-300:]}')
    except subprocess.TimeoutExpired:
        manifest.append({'probe': pname, 'obs': obs, 'seed': seed, 'ok': False, 'error': 'timeout'})
        print(f'  -> TIMEOUT after {time.time()-t0:.0f}s')
    (OUT / '_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

elapsed_all = time.time() - start_all
print(f'\nALL DONE in {elapsed_all / 3600:.2f} h')
print(f'Successful: {sum(1 for m in manifest if m.get("ok"))}/{len(manifest)}')