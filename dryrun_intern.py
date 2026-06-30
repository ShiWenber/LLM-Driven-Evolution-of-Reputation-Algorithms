"""Dry-run: 1 evolutionary trial with Paratera Intern-S2-Preview.
Confirms API + model name + base URL work, gives us a sanity check before
batch-running cross-LLM experiment.
"""
import subprocess, sys, os
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = os.path.join(REPO, 'results', 'dryrun_intern')
Path(OUT).mkdir(parents=True, exist_ok=True)

print('=== Dry run: Intern-S2-Preview via Paratera ===')
print('Config: N=6, G=2, R=5, full obs, 1 seed')
print('Expected: ~5-10 min wall-clock, 1 trial\n')

cmd = [
    sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '6', '--generations', '2',
    '--rounds', '5', '--seeds', '1', '--output', os.path.join(OUT, 'smoke'),
    '--models', 'paratera-intern',
    '--elitism', '1', '--tournament', '2', '--eliminate', '2',
]
env = {**os.environ, 'PYTHONHASHSEED': '1', 'PYTHONUNBUFFERED': '1'}
try:
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    print(f'\nreturncode={proc.returncode}')
    if proc.returncode == 0:
        print('SUCCESS — Intern API works')
        # Find the output file
        outdir = os.path.join(OUT, 'smoke')
        for f in os.listdir(outdir):
            if f.startswith('evo_') and f.endswith('.json'):
                print(f'  Output: {os.path.join(outdir, f)}')
        # Show last 30 lines of stdout
        print('\nLast 30 lines of stdout:')
        for line in proc.stdout.strip().split('\n')[-30:]:
            print(f'  {line}')
    else:
        print('FAILED — see stderr')
        print('STDERR (last 1500 chars):')
        print(proc.stderr[-1500:])
        print('\nSTDOUT (last 1500 chars):')
        print(proc.stdout[-1500:])
except subprocess.TimeoutExpired:
    print('TIMEOUT — Intern API might be slow or hung')