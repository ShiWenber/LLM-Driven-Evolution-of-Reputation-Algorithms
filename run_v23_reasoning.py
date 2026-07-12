"""v23 reasoning-trace mini experiment.
6 trials: 3 obs (private, partial_0.7, full) x 2 seeds.
Captures deepseek-v4-flash reasoning_content at every LLM call (init + mutation).
Goal: understand WHY LLM never wrote leading-eight structure, by reading the
actual thinking content.

Cost: ~$1, ~2h wall time.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUT = REPO / 'results' / 'exp10_reasoning_trace'
OUT.mkdir(parents=True, exist_ok=True)

# Match v15 main plan config (N=10, G=10, R=20, b/c=2) for direct comparability
BASE = ['--population', '10', '--generations', '10', '--rounds', '20',
        '--elitism', '1', '--tournament', '2', '--eliminate', '2']

# 3 obs x 2 seeds
PLAN = [
    ('private', 0), ('private', 1),
    ('partial_0.7', 0), ('partial_0.7', 1),
    ('full', 0), ('full', 1),
]

start_msg = 'v23 reasoning-trace mini-experiment: 6 trials, thinking mode on'
print(start_msg)
print(f'Output: {OUT}')
print(f'Plan: {len(PLAN)} trials')
print()

for obs, seed in PLAN:
    outdir = OUT / f'{obs}_seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
        '--observability', obs, '--seeds', '1', '--output', str(OUT),
        '--models', 'deepseek-v4-flash',
        '--enable-thinking',
        '--reasoning-effort', 'high',
    ] + BASE
    env = {**__import__('os').environ,
           'PYTHONHASHSEED': str(seed*1000+1),
           'PYTHONUNBUFFERED': '1',
           'LLM_MUTATION_WORKERS': '4'}
    import time
    t0 = time.time()
    print(f'[{obs} seed{seed}] starting...', flush=True)
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=2400)
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.0f}s')
        if not ok:
            print(f'  stderr: {proc.stderr[-500:]}')
    except subprocess.TimeoutExpired:
        print(f'  -> TIMEOUT after {time.time()-t0:.0f}s')

print('\nAll trials complete.')
print(f'Reasoning logs in: {OUT}/*/reasoning_*.json')
