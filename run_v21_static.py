"""v21 URGENT: re-run ALL static control trials to fix paper data integrity issue.
Static data was missing (raw JSONs gone, .gitignore excluded them).
4 obs x 10 seeds = 40 trials. ~150s/trial deterministic.
"""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = Path(REPO) / 'results' / 'exp3_static_g10_n10'
OUT.mkdir(parents=True, exist_ok=True)

def has_evo(outdir):
    if not outdir.exists():
        return False
    for f in outdir.iterdir():
        if f.name.startswith('evo_') and f.name.endswith('.json'):
            try:
                with open(f) as fp:
                    t = json.load(fp)
                if t.get('trajectory'):
                    return True
            except Exception:
                pass
    return False

PLAN = []
for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
    for seed in range(10):
        outdir = OUT / f'{obs}_seed{seed}'
        if not has_evo(outdir):
            PLAN.append((obs, seed))
print(f'Static to run: {len(PLAN)} (4 obs x 10 = 40, all missing)')

start = time.time()
for i, (obs, seed) in enumerate(PLAN, 1):
    outdir = OUT / f'{obs}_seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main', '--run', 'static',
        '--observability', obs, '--population', '15', '--generations', '10', '--rounds', '30',
        '--seeds', '1', '--output', str(outdir),
        '--models', 'deepseek-v4-flash',
        '--elitism', '2', '--tournament', '3', '--eliminate', '5',
    ]
    env = {**os.environ, 'PYTHONHASHSEED': str(seed*1000+1), 'PYTHONUNBUFFERED': '1',
           'LLM_MUTATION_WORKERS': '4'}
    t0 = time.time()
    print(f'[{i}/{len(PLAN)}] {obs:12s} seed{seed} | starting...', flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
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
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.0f}s | coop={coop} | total: {(time.time()-start)/60:.1f} min')
        if not ok:
            print(f'  STDERR: {proc.stderr[-300:]}')
    except subprocess.TimeoutExpired:
        print(f'  -> TIMEOUT after {time.time()-t0:.0f}s')

print(f'\nALL DONE in {(time.time()-start)/60:.1f} min')