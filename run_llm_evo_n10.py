"""Re-run Exp 1 (LLM-driven evolution) with n=10 (10 seeds per
observability level) for a tighter cross-seed estimate.

40 trials: 4 observability levels (private, partial_0.3, partial_0.7,
full) x 10 seeds. Same G=10.

Output: results/exp1_method_n10/ (separate from n=3 data which is
preserved as a reference).
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
OUT_ROOT = RESULTS / 'exp1_method_n10'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PLAN = []
for seed in range(10):
    for obs in ('private', 'partial_0.3', 'partial_0.7', 'full'):
        out = OUT_ROOT / f'{obs}_seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        PLAN.append((obs, seed, out))

print(f"Total trials: {len(PLAN)}")
print(f"Estimated wall-clock: ~80 min (LLM-evo = 600-1300s per trial)\n")

manifest_path = OUT_ROOT / 'manifest.json'
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
        '--models', 'deepseek-v4-flash',
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
        agg = list(outdir.glob('evolutionary_*.json'))
        final_coop = None
        if agg:
            try:
                d = json.loads(agg[0].read_text())
                ts = d.get('trials_summary', [{}])[0]
                final_coop = ts.get('final_mean_cooperation')
            except Exception:
                pass
        manifest.append({
            'i': i, 'obs': obs, 'seed': seed, 'ok': ok, 'elapsed_sec': round(elapsed, 1),
            'output': str(outdir), 'final_coop': final_coop,
        })
        print(f"  -> {'OK' if ok else 'FAIL'} in {elapsed:.1f}s  final_coop={final_coop}")
    except subprocess.TimeoutExpired:
        manifest.append({'i': i, 'obs': obs, 'seed': seed, 'ok': False, 'error': 'timeout'})
        print("  -> TIMEOUT")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

elapsed_all = time.time() - start_all
print(f"\nALL DONE in {elapsed_all/60:.1f} min")
print(f"Successful: {sum(1 for m in manifest if m.get('ok'))}/{len(manifest)}")
