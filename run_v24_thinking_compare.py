"""v24 thinking-mode comparison experiment.
4 obs x 2 modes (thinking_on vs thinking_off) x 1 seed = 8 trials.
Direct comparison of deepseek-v4-flash with and without thinking mode enabled.
Same config as v23 (N=10, G=10, R=20, b/c=2).
"""
import subprocess
import sys
import os
import time
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUT = REPO / 'results' / 'exp11_thinking_compare'
OUT.mkdir(parents=True, exist_ok=True)

# Match v15 main plan / v23 config exactly
BASE = ['--population', '10', '--generations', '10', '--rounds', '20',
        '--elitism', '1', '--tournament', '2', '--eliminate', '2']

# 4 obs x 2 modes x 1 seed = 8 trials
OBS = ['private', 'partial_0.3', 'partial_0.7', 'full']
PLAN = []
for obs in OBS:
    for mode in ['on', 'off']:
        PLAN.append((obs, mode, 0))

print(f'v24 thinking-mode comparison: {len(PLAN)} trials')
for obs, mode, seed in PLAN:
    print(f'  {obs} x thinking={mode} seed{seed}')
print()

for obs, mode, seed in PLAN:
    outdir = OUT / f'{obs}_thinking_{mode}' / f'seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
        '--observability', obs, '--seeds', '1', '--output', str(OUT),
        '--models', 'deepseek-v4-flash',
    ] + BASE
    if mode == 'on':
        cmd += ['--enable-thinking', '--reasoning-effort', 'high']
    env = {**os.environ,
           'PYTHONHASHSEED': str(seed*1000+1),
           'PYTHONUNBUFFERED': '1',
           'LLM_MUTATION_WORKERS': '4'}
    t0 = time.time()
    print(f'[{obs} thinking={mode} seed{seed}] starting...', flush=True)
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800)
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.0f}s')
        if not ok:
            print(f'  stderr: {proc.stderr[-500:]}')
    except subprocess.TimeoutExpired:
        print(f'  -> TIMEOUT after {time.time()-t0:.0f}s')

print('\nAll trials complete.')
print(f'Results in: {OUT}')
