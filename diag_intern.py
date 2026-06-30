"""Diagnose: 1 trial with Intern, capture stderr."""
import subprocess, sys
REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
cmd = [
    sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
    '--observability', 'full', '--population', '4', '--generations', '1',
    '--rounds', '3', '--seeds', '1', '--output', REPO + r'\results\diag_intern',
    '--models', 'paratera-intern',
    '--elitism', '1', '--tournament', '1', '--eliminate', '1',
]
proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
print(f'returncode={proc.returncode}')
print('STDOUT:')
print(proc.stdout[-3000:])
print('STDERR:')
print(proc.stderr[-3000:])