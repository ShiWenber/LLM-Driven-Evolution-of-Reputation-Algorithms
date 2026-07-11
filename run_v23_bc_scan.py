"""v23: b/c ratio scan.
3 obs (private, partial_0.7, full) x 4 ratios (1.5, 2, 3, 4) x 3 seeds = 36 trials.
Tests whether the methodology's findings hold across benefit/cost ratios.
Each trial ~7 min with 8 workers -> 36 / 8 * 7 min ~ 30 min (parallel).
"""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = Path(REPO) / 'results' / 'exp9_bc_scan'
OUT.mkdir(parents=True, exist_ok=True)

# N=10, G=10, R=20 -- matches v18/v19 ceiling config
BASE = ['--population', '10', '--generations', '10', '--rounds', '20',
        '--elitism', '1', '--tournament', '2', '--eliminate', '2']

# 3 obs x 4 ratios x 3 seeds = 36 trials
OBS = ['private', 'partial_0.7', 'full']
RATIOS = [(1.5, 1), (2, 1), (3, 1), (4, 1)]  # (benefit, cost)
SEEDS = 3

PLAN = []
for obs in OBS:
    for benefit, cost in RATIOS:
        for seed in range(SEEDS):
            PLAN.append((obs, benefit, cost, seed))

print(f'Trials: {len(PLAN)} (3 obs x 4 ratios x 3 seeds)')

manifest_path = OUT / '_manifest.json'
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
else:
    manifest = []
done_keys = {(m['obs'], m['benefit'], m['cost'], m['seed']) for m in manifest if m.get('ok')}
PLAN = [p for p in PLAN if (p[0], p[1], p[2], p[3]) not in done_keys]
print(f'To run: {len(PLAN)}')

start = time.time()
for i, (obs, benefit, cost, seed) in enumerate(PLAN, 1):
    outdir = OUT / f'{obs}_b{benefit}c{cost}' / f'seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
        '--observability', obs, '--seeds', '1', '--output', str(outdir),
        '--models', 'deepseek-v4-flash',
        '--benefit', str(benefit), '--cost', str(cost),
    ] + BASE
    env = {**os.environ, 'PYTHONHASHSEED': str(seed*1000+1), 'PYTHONUNBUFFERED': '1',
           'LLM_MUTATION_WORKERS': '8'}
    t0 = time.time()
    print(f'[{i}/{len(PLAN)}] obs={obs:10s} b/c={benefit}/{cost} seed{seed} | starting...', flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1200)
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        coop = None
        for f in outdir.iterdir():
            if f.name.startswith('evo_') and f.name.endswith('.json'):
                with open(f) as fp:
                    t = json.load(fp)
                if t.get('trajectory'):
                    coop = t['trajectory'][-1]['cooperation_rate_mean']
                break
        new_entry = {'obs': obs, 'benefit': benefit, 'cost': cost, 'seed': seed,
                     'ok': ok, 'elapsed_sec': round(elapsed, 1), 'coop_final': coop}
        manifest.append(new_entry)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.0f}s | coop={coop} | total: {(time.time()-start)/60:.1f} min')
    except subprocess.TimeoutExpired:
        print(f'  -> TIMEOUT after {time.time()-t0:.0f}s')
        manifest.append({'obs': obs, 'benefit': benefit, 'cost': cost, 'seed': seed, 'ok': False, 'error': 'timeout'})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

print(f'\nALL DONE in {(time.time()-start)/60:.1f} min')
n_ok = sum(1 for m in manifest if m.get('ok'))
print(f'OK: {n_ok}/{len(manifest)} total')