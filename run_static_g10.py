"""Re-run Exp 3 (static) with G=10 instead of G=5, to match the LLM-evo
generation count and make the Fig 4 trajectory comparison fair.

6 trials: 3 observability levels (private, partial_0.3, full) x 2 seeds.
Output: results/exp3_static_g10/ (separate directory so we don't overwrite
the G=5 data, which is still useful for the Static vs LLM comparison
in the control table).
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
OUT_ROOT = RESULTS / 'exp3_static_g10'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PLAN = []
for seed in (0, 1):
    for obs in ('private', 'partial_0.3', 'full'):
        out = OUT_ROOT / f'{obs}_seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        PLAN.append((obs, seed, out))

print(f"Total trials: {len(PLAN)}")
print(f"Estimated wall-clock: ~3-5 min\n")

manifest_path = OUT_ROOT / 'manifest.json'
manifest = []

start_all = time.time()
for i, (obs, seed, outdir) in enumerate(PLAN, 1):
    t0 = time.time()
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main',
        '--run', 'static',
        '--observability', obs,
        '--population', '15',
        '--generations', '10',  # <-- the key change
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
            capture_output=True, text=True, timeout=1200,
        )
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        # Parse final_coop from the aggregate file
        agg = list(outdir.glob('static_control_*.json'))
        final_coop = None
        if agg:
            try:
                d = json.loads(agg[0].read_text())
                final_coop = d.get('final_mean_cooperation')
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
