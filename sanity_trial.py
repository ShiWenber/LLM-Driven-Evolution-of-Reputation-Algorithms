"""Sanity trial: run 1 evolutionary trial with --models deepseek-v4-flash
and confirm the LLM mutation path is actually exercised (not fallback)."""
import os
import sys
import subprocess
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

# Output to a clean directory so we don't pollute the existing exp1_method/private_seed0
out = REPO / 'results' / 'sanity_trial'
out.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, '-u', '-m', 'experiments.main',
    '--run', 'evolutionary',
    '--observability', 'private',
    '--population', '15',
    '--generations', '2',  # only 2 generations to keep trial short (~3 min)
    '--rounds', '30',
    '--seeds', '1',
    '--output', str(out),
    '--models', 'deepseek-v4-flash',  # the fix
]
print(f"$ {' '.join(cmd[2:])}")
print()

proc = subprocess.run(
    cmd,
    cwd=str(REPO),
    env={**os.environ, 'PYTHONHASHSEED': '1', 'PYTHONUNBUFFERED': '1'},
    capture_output=True,
    text=True,
    timeout=900,  # 15 min safety
)
print(f"return code: {proc.returncode}")
print(f"elapsed: ~{proc.stdout.split('elapsed=')[-1].split('s')[0] if 'elapsed=' in proc.stdout else '?'}s")
print()
print("--- stdout (full) ---")
print(proc.stdout)
print("--- stderr (last 1000) ---")
print(proc.stderr[-1000:])

# Check for fallback indicators
fallback_count = proc.stdout.count('using random mutation fallback') + proc.stdout.count('using fallback strategies')
llm_init_count = proc.stdout.count('Init:') + proc.stdout.count('[init]')
mutation_count = proc.stdout.count('[mutation]')
api_err_count = proc.stdout.count('invalid_api_key') + proc.stdout.count('invalid_request_error')
print()
print(f"=== DIAGNOSTICS ===")
print(f"  Fallback events: {fallback_count}")
print(f"  API errors: {api_err_count}")
print(f"  [init] events: {llm_init_count}")
print(f"  [mutation] events: {mutation_count}")
if api_err_count == 0 and fallback_count == 0:
    print("VERDICT: LLM path is clean (no API errors, no fallback).")
elif api_err_count > 0:
    print("VERDICT: API key still failing.")
else:
    print("VERDICT: LLM path failed for other reason (check output).")
