"""v20: Complete the data holes.
- Static partial_0.7 (10 seeds, fills the gap in v15 static control)
- Random mutation n=5 per obs (3 obs x 5 seeds = 15 trials, replaces thin n=2)

Both are deterministic (only initial population uses 1 LLM call each).
Total wall-clock: ~10 min with 8 concurrent workers.
"""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'

# === STATIC partial_0.7, 10 seeds ===
STATIC_OUT = Path(REPO) / 'results' / 'exp3_static_g10_n10'
STATIC_OUT.mkdir(parents=True, exist_ok=True)
static_plan = [('partial_0.7', s) for s in range(10)]
# skip already-done
done_static = set()
for obs, seed in static_plan:
    if (STATIC_OUT / f'{obs}_seed{seed}').exists():
        done_static.add((obs, seed))
static_plan = [p for p in static_plan if p not in done_static]
print(f'Static partial_0.7 to run: {len(static_plan)} (skipping {len(done_static)} done)')

# === RANDOM MUTATION n=5 per obs ===
RAND_OUT = Path(REPO) / 'results' / 'exp4_random_mut'
RAND_OUT.mkdir(parents=True, exist_ok=True)
random_plan = []
for obs in ['private', 'partial_0.3', 'full']:
    for seed in range(5):
        random_plan.append((obs, seed))
done_random = set()
for obs, seed in random_plan:
    if (RAND_OUT / f'{obs}_seed{seed}').exists():
        done_random.add((obs, seed))
random_plan = [p for p in random_plan if p not in done_random]
print(f'Random mutation to run: {len(random_plan)} (skipping {len(done_random)} done)')

# Build all commands
def make_cmd(obs, seed, kind):
    outdir = (STATIC_OUT if kind == 'static' else RAND_OUT) / f'{obs}_seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main',
        '--run', kind,
        '--observability', obs,
        '--population', '15', '--generations', '10', '--rounds', '30',
        '--seeds', '1', '--output', str(outdir),
        '--models', 'deepseek-v4-flash',
        '--elitism', '2', '--tournament', '3', '--eliminate', '5',
    ]
    return cmd, outdir

def has_evo(outdir):
    """Check if outdir has a successful evo_*.json (not just empty dir)."""
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

# Re-filter using has_evo
static_plan_checked = [p for p in static_plan if not has_evo(STATIC_OUT / f'{p[0]}_seed{p[1]}')]
random_plan_checked = [p for p in random_plan if not has_evo(RAND_OUT / f'{p[0]}_seed{p[1]}')]
print(f'After re-check: static={len(static_plan_checked)} to run, random={len(random_plan_checked)} to run')
static_plan = static_plan_checked
random_plan = random_plan_checked

PLAN = []
for obs, seed in static_plan:
    PLAN.append(('static', obs, seed))
for obs, seed in random_plan:
    PLAN.append(('random-mutation', obs, seed))
print(f'\nTotal trials: {len(PLAN)}')

start = time.time()
for i, (kind, obs, seed) in enumerate(PLAN, 1):
    cmd, outdir = make_cmd(obs, seed, kind)
    env = {**os.environ, 'PYTHONHASHSEED': str(seed*1000+1), 'PYTHONUNBUFFERED': '1',
           'LLM_MUTATION_WORKERS': '4'}
    t0 = time.time()
    print(f'[{i}/{len(PLAN)}] {kind:15s} {obs:12s} seed{seed} | starting...', flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
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
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.0f}s | coop={coop} | total: {(time.time()-start)/3600:.2f}h')
        if not ok:
            print(f'  STDERR: {proc.stderr[-300:]}')
    except subprocess.TimeoutExpired:
        print(f'  -> TIMEOUT after {time.time()-t0:.0f}s')

print(f'\nALL DONE in {(time.time()-start)/60:.1f} min')