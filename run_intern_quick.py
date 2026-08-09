"""Quick batch with Intern: 2 obs (full, partial_0.7) x 2 seeds = 4 trials.
Goal: validate Intern can actually generate usable code with max_tokens=3000
before committing to the full 48-trial batch."""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = Path(REPO) / 'results' / 'exp8_intern_quick'
OUT.mkdir(parents=True, exist_ok=True)

PLAN = [
    ('full', 0), ('full', 1),
    ('partial_0.7', 0), ('partial_0.7', 1),
]

for obs, seed in PLAN:
    outdir = OUT / f'{obs}_seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
        '--observability', obs, '--population', '8', '--generations', '5',
        '--rounds', '20', '--seeds', '1', '--output', str(outdir),
        '--models', 'paratera-intern',
        '--elitism', '2', '--tournament', '3', '--eliminate', '3',
    ]
    env = {**os.environ, 'PYTHONHASHSEED': str(seed*1000+1), 'PYTHONUNBUFFERED': '1'}
    t0 = time.time()
    print(f'[{obs} seed{seed}] starting...', flush=True)
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
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.0f}s | final coop={coop}')
    except subprocess.TimeoutExpired:
        print(f'  -> TIMEOUT after {time.time()-t0:.0f}s')

print('\n=== Quick batch done ===')