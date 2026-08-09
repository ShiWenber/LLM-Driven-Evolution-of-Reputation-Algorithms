"""Experiment 6: Leading-Eight Exploration with the augmented observation
interface (donor_reputation + recipient_reputation in evaluate's observation dict).

One trial: full observability, G=10, 1 seed, deepseek-v4-flash.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUTPUT_DIR = REPO / 'results' / 'exp6_leading_eight'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, '-u', '-m', 'experiments.main',
    '--run', 'evolutionary',
    '--observability', 'full',
    '--population', '15',
    '--generations', '10',
    '--rounds', '30',
    '--seeds', '1',
    '--output', str(OUTPUT_DIR),
    '--models', 'deepseek-v4-flash',
    '--elitism', '3',
    '--tournament', '3',
    '--eliminate', '5',
]

env = {**os.environ, 'PYTHONHASHSEED': '7001', 'PYTHONUNBUFFERED': '1'}

print(f"[exp6] cmd: {' '.join(cmd[2:])}")
t0 = time.time()
proc = subprocess.run(
    cmd,
    cwd=str(REPO),
    env=env,
    capture_output=True,
    text=True,
    timeout=2400,  # 40 min safety margin
)
elapsed = time.time() - t0
ok = proc.returncode == 0
print(f"[exp6] {'OK' if ok else 'FAIL'} in {elapsed:.1f}s")
print(f"[exp6] stdout tail: {proc.stdout[-500:]}")
if not ok:
    print(f"[exp6] stderr tail: {proc.stderr[-1000:]}")

# Save manifest
import json
manifest = {
    'experiment': 'exp6_leading_eight',
    'obs': 'full',
    'n_seeds': 1,
    'G': 10,
    'ok': ok,
    'elapsed_sec': round(elapsed, 1),
    'stdout_tail': proc.stdout[-500:],
    'stderr_tail': proc.stderr[-500:],
}
(OUTPUT_DIR / '_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f"[exp6] manifest saved to {OUTPUT_DIR / '_manifest.json'}")
