"""Robustness plan: deepseek-coder × partial_0.3/0.7 × 3 seeds = 6 trials.

Purpose: cross-LLM validation of the v7 main result. v7 used deepseek-v4-flash.
This plan uses deepseek-coder — a different model from the same family.
Partial_0.3 and partial_0.7 are the two observability levels at which v7
showed peak cooperation (0.51 and 0.39 respectively).

Output: results/exp5_robustness/{obs}_seed{N}/
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
RESULTS = REPO / 'results'
OUT_ROOT = RESULTS / 'exp5_robustness'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Build the 6-trial plan
PLAN = []
for seed in (0, 1, 2):
    for obs in ('partial_0.3', 'partial_0.7'):
        out = OUT_ROOT / f'{obs}_seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        PLAN.append((obs, seed, out))

print(f"Total trials: {len(PLAN)}")
print(f"Estimated wall-clock: ~{len(PLAN)*5/60:.1f} hours (deepseek-coder is fast)\n")

manifest_path = RESULTS / 'robustness_manifest.json'
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
else:
    manifest = []

start_all = time.time()
for i, (obs, seed, outdir) in enumerate(PLAN, 1):
    t0 = time.time()
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main',
        '--run', 'evolutionary',
        '--observability', obs,
        '--population', '15',
        '--generations', '10',
        '--rounds', '30',
        '--seeds', '1',
        '--output', str(outdir),
        '--models', 'deepseek-coder',
    ]
    env_seed = str(seed * 1000 + 1)
    print(f"[{i}/{len(PLAN)}] obs={obs} seed={seed}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env={**os.environ, 'PYTHONHASHSEED': env_seed, 'PYTHONUNBUFFERED': '1'},
            capture_output=True, text=True, timeout=1800,
        )
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        # Find the final cooperation from the aggregate file
        agg = list(outdir.glob('evolutionary_*.json'))
        final_coop = None
        if agg:
            try:
                d = json.loads(agg[0].read_text())
                ts = d.get('trials_summary', [])
                if ts:
                    final_coop = ts[0].get('final_mean_cooperation')
            except Exception:
                pass
        manifest.append({
            'i': i, 'obs': obs, 'seed': seed, 'ok': ok, 'elapsed_sec': round(elapsed, 1),
            'output': str(outdir), 'final_coop': final_coop,
            'stdout_tail': (proc.stdout or '')[-500:],
            'stderr_tail': (proc.stderr or '')[-300:],
        })
        print(f"  -> {'OK' if ok else 'FAIL'} in {elapsed:.1f}s  final_coop={final_coop}")
    except subprocess.TimeoutExpired:
        manifest.append({'i': i, 'obs': obs, 'seed': seed, 'ok': False, 'error': 'timeout'})
        print("  -> TIMEOUT")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

elapsed_all = time.time() - start_all
print(f"\nALL DONE in {elapsed_all/60:.1f} min")
print(f"Successful: {sum(1 for m in manifest if m.get('ok'))}/{len(manifest)}")
