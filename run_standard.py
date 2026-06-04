"""Run the Standard experimental plan in the background.

Standard plan:
  Experiment 1 (methodology validation, 4 obs x 3 seeds, N=15, G=10, T=30):
    private (p=0), partial_0.3, partial_0.7, full (p=1)
    12 trials  * ~13 min = ~2.6 h
  Experiment 2 (phase-transition shape, 6 p x 2 seeds, N=15, G=5, T=30):
    p in {0, 0.10, 0.30, 0.50, 0.70, 1.0}
    12 trials  * ~6.5 min = ~1.3 h
  Experiment 3 (static control, 3 obs x 2 seeds, N=15, G=5, T=30):
    p in {0, 0.3, 1.0}
    6 trials  * ~6.5 min = ~0.65 h
  Experiment 4 (random-mutation control, 3 obs x 2 seeds, N=15, G=10, T=30):
    p in {0, 0.3, 1.0}
    6 trials  * ~13 min = ~1.3 h
  Total: 36 trials, ~5.8 h, $18-25

Usage:
    python run_standard.py
    python run_standard.py --skip 1     # skip first trial (already done)
"""
import os
import subprocess
import sys
import time
import json
import argparse
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

# Build the full plan of (run, obs, seeds, gens, output_label) tuples.
PLAN = []

# Experiment 1: 4 obs * 3 seeds, G=10
for seed in [0, 1, 2]:
    for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
        PLAN.append((
            'evolutionary',
            obs, seed, 10,
            f'results/exp1_method/{obs}_seed{seed}',
        ))

# Experiment 2: 6 p * 2 seeds, G=5 (threshold mode caps G at 5 anyway)
for seed in [0, 1]:
    for p in [0.0, 0.10, 0.30, 0.50, 0.70, 1.0]:
        if p == 0.0:
            obs = 'private'
        elif p >= 1.0:
            obs = 'full'
        else:
            obs = f'partial_{p}'
        PLAN.append((
            'threshold',
            obs, seed, 5,
            f'results/exp2_threshold/{obs}_seed{seed}',
        ))

# Experiment 3: 3 obs * 2 seeds, G=5, no selection
for seed in [0, 1]:
    for obs in ['private', 'partial_0.3', 'full']:
        PLAN.append((
            'static',
            obs, seed, 5,
            f'results/exp3_static/{obs}_seed{seed}',
        ))

# Experiment 4: 3 obs * 2 seeds, G=10, random mutation
for seed in [0, 1]:
    for obs in ['private', 'partial_0.3', 'full']:
        PLAN.append((
            'random-mutation',
            obs, seed, 10,
            f'results/exp4_random_mut/{obs}_seed{seed}',
        ))

# Parse --skip to allow resuming from a later trial index.
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--skip', type=int, default=0,
                help='Skip the first N trials in the plan (e.g. if they were already done)')
_args, _unknown = _p.parse_known_args()
if _args.skip:
    print(f"--skip={_args.skip}: skipping first {_args.skip} trial(s) that were already completed")
    PLAN = PLAN[_args.skip:]
    # Re-anchor manifest with the skipped entries
    _old_manifest_path = REPO / 'results' / '_manifest.json'
    if _old_manifest_path.exists():
        try:
            _old = json.loads(_old_manifest_path.read_text(encoding='utf-8'))
            _already = _old[:_args.skip]
            # Re-index to start at the post-skip number
            manifest = _already
        except Exception:
            manifest = []
    else:
        manifest = []
else:
    manifest = []

print(f"Total trials: {len(PLAN)}")
print(f"Estimated wall-clock: ~{len(PLAN)*8/60:.1f} hours\n")

# Execute
start_all = time.time()
for i, (run, obs, seed, gens, outdir) in enumerate(PLAN, 1):
    t0 = time.time()
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main',
        '--run', run,
        '--observability', obs,
        '--population', '15',
        '--generations', str(gens),
        '--rounds', '30',
        '--seeds', '1',
        '--output', outdir,
    ]
    if run == 'threshold':
        # Reuse the partial_{p} format; main expects --p-values for threshold mode
        if obs == 'private':
            cmd += ['--p-values', '0.0']
        elif obs == 'full':
            cmd += ['--p-values', '1.0']
        elif obs.startswith('partial_'):
            p = obs.split('_')[1]
            cmd += ['--p-values', p]
    if run == 'random-mutation':
        # The main.py passes --output to population.results_dir; it will write
        # trajectories normally. The mutation operator is replaced via main.py flag.
        pass
    # Use --seed-offset trick: rerun the trial with a base offset equal to the seed.
    # But main.py does not currently expose --seed-offset. Instead, we set the
    # environment variable PYTHONHASHSEED for each trial, which affects the random
    # state inside evolutionary.mutation and population.
    env_seed = str(seed * 1000 + 1)
    trial_idx = i + _args.skip  # global trial index
    print(f"[{trial_idx}/{len(PLAN)+_args.skip}] {run:18s} obs={obs:12s} seed={seed} gens={gens} | {' '.join(cmd[2:])}", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env={**os.environ, 'PYTHONHASHSEED': env_seed, 'PYTHONUNBUFFERED': '1'},
            capture_output=True,
            text=True,
            timeout=2400,  # 40 min per trial safety margin
        )
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        manifest.append({
            'i': i, 'run': run, 'obs': obs, 'seed': seed, 'gens': gens,
            'output': outdir, 'ok': ok, 'elapsed_sec': round(elapsed, 1),
            'stdout_tail': (proc.stdout or '')[-500:],
            'stderr_tail': (proc.stderr or '')[-500:],
        })
        print(f"  -> {'OK' if ok else 'FAIL'} in {elapsed:.1f}s")
        if not ok:
            print(f"  STDERR-tail: {proc.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        manifest.append({
            'i': i, 'run': run, 'obs': obs, 'seed': seed, 'gens': gens,
            'output': outdir, 'ok': False, 'elapsed_sec': 2400,
            'error': 'timeout',
        })
        print(f"  -> TIMEOUT (>40min)")
    # Save manifest after each trial so we can recover if interrupted
    Path(REPO / 'results' / '_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )

elapsed_all = time.time() - start_all
print(f"\nALL DONE in {elapsed_all/3600:.2f} h")
print(f"Successful: {sum(1 for m in manifest if m.get('ok'))}/{len(manifest)}")
