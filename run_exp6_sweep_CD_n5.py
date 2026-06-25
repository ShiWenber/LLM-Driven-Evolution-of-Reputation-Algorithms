"""Exp 6 sweep v3: 5 seeds x 4 obs with 'cooperate'/'defect' labels.
Same spec as exp6_sweep_AB_n5 so the comparison is fair."""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUTPUT_ROOT = REPO / 'results' / 'exp6_sweep_CD_n5'
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

PLAN = []
for seed in range(5):
    for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
        out = OUTPUT_ROOT / f'{obs}_seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        PLAN.append((obs, seed, str(out)))

print(f'Total trials: {len(PLAN)}')
print(f'Estimated wall-clock: ~{len(PLAN) * 7.5 / 60:.1f} h\n')

start_all = time.time()
manifest = []
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
        '--output', outdir,
        '--models', 'deepseek-v4-flash',
        '--elitism', '3',
        '--tournament', '3',
        '--eliminate', '5',
    ]
    env = {**os.environ, 'PYTHONHASHSEED': str(seed * 1000 + 1), 'PYTHONUNBUFFERED': '1'}
    print(f'[{i}/{len(PLAN)}] obs={obs:12s} seed={seed} | starting...', flush=True)
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=2400,
        )
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        manifest.append({'obs': obs, 'seed': seed, 'outdir': outdir, 'ok': ok,
                         'elapsed_sec': round(elapsed, 1)})
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.1f}s')
        if not ok:
            print(f'  stderr: {proc.stderr[-300:]}')
    except subprocess.TimeoutExpired:
        manifest.append({'obs': obs, 'seed': seed, 'outdir': outdir, 'ok': False, 'error': 'timeout'})
        print(f'  -> TIMEOUT')
    (OUTPUT_ROOT / '_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

elapsed_all = time.time() - start_all
print(f'\nALL DONE in {elapsed_all / 3600:.2f} h')
print(f'Successful: {sum(1 for m in manifest if m.get("ok"))}/{len(manifest)}')