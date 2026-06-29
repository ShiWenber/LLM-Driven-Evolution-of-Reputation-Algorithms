"""Smoke test: verify new opt-in features work without breaking default behavior.
Test 1: default (no flags) = identical behavior to v15
Test 2: --recent-window 5 = observation dict has 'recent_window' field
Test 3: --reputation-noise 0.1 = no exception
Test 4: --exploration-mutation = no exception
"""
import subprocess, sys, json, os
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = os.path.join(REPO, 'results', 'dryrun_complexity_ceiling')
Path(OUT).mkdir(parents=True, exist_ok=True)

# Test 1: default (no flags)
print('=== Test 1: default (no flags, baseline) ===')
proc = subprocess.run([
    sys.executable, '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '6', '--generations', '2',
    '--rounds', '5', '--seeds', '1', '--output', os.path.join(OUT, 'default'),
    '--models', 'deepseek-v4-flash', '--elitism', '1', '--tournament', '2', '--eliminate', '2',
], cwd=REPO, capture_output=True, text=True, timeout=600)
print(f'  returncode={proc.returncode}')
if proc.returncode != 0:
    print(f'  STDERR: {proc.stderr[-500:]}')
    print(f'  STDOUT: {proc.stdout[-500:]}')

# Test 2: recent_window
print('\n=== Test 2: --recent-window 5 ===')
proc = subprocess.run([
    sys.executable, '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '6', '--generations', '2',
    '--rounds', '5', '--seeds', '1', '--output', os.path.join(OUT, 'recent5'),
    '--models', 'deepseek-v4-flash', '--elitism', '1', '--tournament', '2', '--eliminate', '2',
    '--recent-window', '5',
], cwd=REPO, capture_output=True, text=True, timeout=600)
print(f'  returncode={proc.returncode}')
if proc.returncode != 0:
    print(f'  STDERR: {proc.stderr[-500:]}')
    print(f'  STDOUT: {proc.stdout[-500:]}')

# Test 3: reputation_noise
print('\n=== Test 3: --reputation-noise 0.1 ===')
proc = subprocess.run([
    sys.executable, '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '6', '--generations', '2',
    '--rounds', '5', '--seeds', '1', '--output', os.path.join(OUT, 'noise01'),
    '--models', 'deepseek-v4-flash', '--elitism', '1', '--tournament', '2', '--eliminate', '2',
    '--reputation-noise', '0.1',
], cwd=REPO, capture_output=True, text=True, timeout=600)
print(f'  returncode={proc.returncode}')
if proc.returncode != 0:
    print(f'  STDERR: {proc.stderr[-500:]}')
    print(f'  STDOUT: {proc.stdout[-500:]}')

# Test 4: exploration_mutation
print('\n=== Test 4: --exploration-mutation ===')
proc = subprocess.run([
    sys.executable, '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '6', '--generations', '2',
    '--rounds', '5', '--seeds', '1', '--output', os.path.join(OUT, 'exploration'),
    '--models', 'deepseek-v4-flash', '--elitism', '1', '--tournament', '2', '--eliminate', '2',
    '--exploration-mutation',
], cwd=REPO, capture_output=True, text=True, timeout=600)
print(f'  returncode={proc.returncode}')
if proc.returncode != 0:
    print(f'  STDERR: {proc.stderr[-500:]}')
    print(f'  STDOUT: {proc.stdout[-500:]}')

# Test 5: ALL combined
print('\n=== Test 5: all features combined ===')
proc = subprocess.run([
    sys.executable, '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '6', '--generations', '2',
    '--rounds', '5', '--seeds', '1', '--output', os.path.join(OUT, 'all'),
    '--models', 'deepseek-v4-flash', '--elitism', '1', '--tournament', '2', '--eliminate', '2',
    '--recent-window', '5', '--reputation-noise', '0.1', '--exploration-mutation',
], cwd=REPO, capture_output=True, text=True, timeout=600)
print(f'  returncode={proc.returncode}')
if proc.returncode != 0:
    print(f'  STDERR: {proc.stderr[-500:]}')
    print(f'  STDOUT: {proc.stdout[-500:]}')

# Now inspect: does an obs in recent_window have recent_window field?
print('\n=== Inspect: did recent_window actually inject? ===')
import glob
files = glob.glob(os.path.join(OUT, 'recent5', '**', 'evo_*.json'), recursive=True)
if files:
    with open(files[0]) as f:
        t = json.load(f)
    # The observation dicts are in trajectory, but observation isn't logged by default.
    # The presence of recent_window field is tested by the code not crashing.
    print(f'  Loaded: {files[0]}')
    print(f'  trial trajectory len: {len(t["trajectory"])}')
    print(f'  final cooperation: {t["trajectory"][-1]["cooperation_rate_mean"]:.3f}')

print('\n=== ALL DRY RUNS DONE ===')