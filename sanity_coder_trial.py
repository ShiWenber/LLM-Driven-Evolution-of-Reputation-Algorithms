"""Sanity trial: 1 evolutionary run with deepseek-coder, 2 gens, partial_0.3."""
import os
import sys
import subprocess
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
out = REPO / 'results' / 'robustness_sanity'
out.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, '-u', '-m', 'experiments.main',
    '--run', 'evolutionary',
    '--observability', 'partial_0.3',
    '--population', '15',
    '--generations', '2',
    '--rounds', '30',
    '--seeds', '1',
    '--output', str(out),
    '--models', 'deepseek-coder',
]
print(f"$ {' '.join(cmd[2:])}")
print()

proc = subprocess.run(
    cmd,
    cwd=str(REPO),
    env={**os.environ, 'PYTHONHASHSEED': '1', 'PYTHONUNBUFFERED': '1'},
    capture_output=True,
    text=True,
    timeout=900,
)
print(f"return code: {proc.returncode}")
print()
print("--- stdout ---")
print(proc.stdout)
print("--- stderr (last 1000) ---")
print(proc.stderr[-1000:])

fallback_count = proc.stdout.count('using random mutation fallback') + proc.stdout.count('using fallback strategies')
api_err_count = proc.stdout.count('invalid_api_key') + proc.stdout.count('invalid_request_error')
print()
print(f"=== DIAGNOSTICS ===")
print(f"  Fallback events: {fallback_count}")
print(f"  API errors: {api_err_count}")
if api_err_count == 0 and fallback_count == 0:
    print("VERDICT: LLM path is clean.")
else:
    print("VERDICT: LLM path failed.")
