"""Verify recent_window appears in actual observation during a full trial.
Then check if any LLM-generated strategy actually references recent_window.
"""
import subprocess, sys, json, os
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = os.path.join(REPO, 'results', 'dryrun_complexity_ceiling_v2')
Path(OUT).mkdir(parents=True, exist_ok=True)

# A: --recent-window 5 with full obs (N=8, G=2, R=10) — small enough to inspect
print('=== A: --recent-window 5 (full obs) ===')
proc = subprocess.run([
    sys.executable, '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '8', '--generations', '2',
    '--rounds', '10', '--seeds', '1', '--output', os.path.join(OUT, 'recent5'),
    '--models', 'deepseek-v4-flash', '--elitism', '2', '--tournament', '2', '--eliminate', '3',
    '--recent-window', '5',
], cwd=REPO, capture_output=True, text=True, timeout=1200)
print(f'  returncode={proc.returncode}')
if proc.returncode != 0:
    print(f'  STDERR: {proc.stderr[-1000:]}')
    print(f'  STDOUT: {proc.stdout[-1000:]}')
else:
    # Did any strategy reference recent_window?
    import glob, re
    files = glob.glob(os.path.join(OUT, 'recent5', '**', 'evo_*.json'), recursive=True)
    if files:
        with open(files[0]) as f:
            t = json.load(f)
        codes = [a['code'] for a in t['final_population']]
        n_ref = sum(1 for c in codes if 'recent_window' in c)
        print(f'  final pop: {len(codes)} agents, {n_ref} reference recent_window')
        if n_ref > 0:
            for c in codes:
                if 'recent_window' in c:
                    # Show the use
                    for line in c.split('\n'):
                        if 'recent_window' in line:
                            print(f'    >> {line.strip()[:120]}')

# B: --exploration-mutation only
print('\n=== B: --exploration-mutation only (full obs) ===')
proc = subprocess.run([
    sys.executable, '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '8', '--generations', '2',
    '--rounds', '10', '--seeds', '1', '--output', os.path.join(OUT, 'exploration'),
    '--models', 'deepseek-v4-flash', '--elitism', '2', '--tournament', '2', '--eliminate', '3',
    '--exploration-mutation',
], cwd=REPO, capture_output=True, text=True, timeout=1200)
print(f'  returncode={proc.returncode}')
if proc.returncode != 0:
    print(f'  STDERR: {proc.stderr[-1000:]}')

print('\n=== DONE ===')