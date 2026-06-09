"""Re-run Exp 3 (static) with G=10 and n=10 (10 seeds per observability
level) for a tighter cross-seed estimate.

30 trials: 3 observability levels (private, partial_0.3, full) x 10 seeds.
Output: results/exp3_static_g10_n10/ (separate from the G=10 n=2 data
which we keep as a sanity-check reference).
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
OUT_ROOT = RESULTS / 'exp3_static_g10_n10'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PLAN = []
for seed in range(10):
    for obs in ('private', 'partial_0.3', 'full'):
        out = OUT_ROOT / f'{obs}_seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        PLAN.append((obs, seed, out))

print(f"Total trials: {len(PLAN)}")
print(f"Estimated wall-clock: ~50 min (static = 100-150s per trial)\n")

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
            capture_output=True, text=True, timeout=900,
        )
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        agg = list(outdir.glob('static_control_*.json'))
        final_coop = None
        if agg:
            try:
                d = json.loads(agg[0].read_text())
                ts_list = d.get('trials_summary', [])
                if ts_list:
                    final_coop = ts_list[0].get('final_mean_cooperation')
                else:
                    final_coop = d.get('final_mean_cooperation')
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
    # Save manifest after each trial
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

elapsed_all = time.time() - start_all
print(f"\nALL DONE in {elapsed_all/60:.1f} min")
print(f"Successful: {sum(1 for m in manifest if m.get('ok'))}/{len(manifest)}")
