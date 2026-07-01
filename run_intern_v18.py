"""v18 Intern ceiling full batch: 5 probes x 2 obs x 3 seeds = 30 trials.
All probes use uniform N=10, G=10, T=20 config (not N=15, G=10 like v15).
This is to keep Intern trials under 15 min each.

Probe A is the missing one from v17 (skipped due to 30K-token thinking).
With N=10, G=10 (not 30, 20), we can run it.

Concurrent 8 workers (up from 5) for ~2x speedup.
per_call_timeout=120s in MutationOperator prevents hang on slow Intern calls.
"""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = Path(REPO) / 'results' / 'exp8_intern_ceiling_v18'
OUT.mkdir(parents=True, exist_ok=True)

# Uniform small config: N=10, G=10, T=20 — Intern fits in ~12 min/trial
BASE = ['--population', '10', '--generations', '10', '--rounds', '20',
        '--elitism', '1', '--tournament', '2', '--eliminate', '2']

PROBES = {
    'A_larger_budget':    BASE,  # NO N=30 here — instead, this becomes "uniform small" baseline+probe at N=10,G=10
    'B_recent_window':    BASE + ['--recent-window', '5'],
    'C_reputation_noise': BASE + ['--reputation-noise', '0.1'],
    'D_exploration':      BASE + ['--exploration-mutation'],
    'E_baseline':         BASE,  # IDENTICAL to A — just no feature; we treat E as the "uniform baseline" reference
}

OBS_LIST = ['full', 'partial_0.7']
N_SEEDS = 3

PLAN = []
for pname, pargs in PROBES.items():
    for obs in OBS_LIST:
        for seed in range(N_SEEDS):
            PLAN.append((pname, obs, seed, pargs))

print(f'Total trials: {len(PLAN)}')
print(f'Estimated wall-clock: ~{len(PLAN) * 12 / 60:.1f} h (8 concurrent workers, ~10-12 min/trial)')

# Check existing manifest to skip done trials
manifest_path = OUT / '_manifest.json'
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
else:
    manifest = []
done_keys = {(m['probe'], m['obs'], m['seed']) for m in manifest if m.get('ok')}
PLAN = [p for p in PLAN if (p[0], p[1], p[2]) not in done_keys]
print(f'Trials to run: {len(PLAN)} (skipping {len(done_keys)} already-done)')

start = time.time()
for i, (pname, obs, seed, pargs) in enumerate(PLAN, 1):
    outdir = OUT / pname / f'{obs}_seed{seed}'
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-u', '-m', 'experiments.main', '--run', 'evolutionary',
        '--observability', obs, '--seeds', '1', '--output', str(outdir),
        '--models', 'paratera-intern',
    ] + pargs
    env = {**os.environ, 'PYTHONHASHSEED': str(seed*1000+1), 'PYTHONUNBUFFERED': '1',
           'LLM_MUTATION_WORKERS': '8'}
    t0 = time.time()
    print(f'[{i}/{len(PLAN)}] {pname:18s} {obs:12s} seed{seed} | starting...', flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1500)
        ok = proc.returncode == 0
        elapsed = time.time() - t0
        coop = None
        for f in os.listdir(outdir):
            if f.startswith('evo_') and f.endswith('.json'):
                with open(outdir / f) as fp:
                    t = json.load(fp)
                if t.get('trajectory'):
                    coop = t['trajectory'][-1]['cooperation_rate_mean']
                break
        new_entry = {'probe': pname, 'obs': obs, 'seed': seed, 'ok': ok,
                     'elapsed_sec': round(elapsed, 1), 'coop_final': coop}
        replaced = False
        for i2, m in enumerate(manifest):
            if m.get('probe') == pname and m.get('obs') == obs and m.get('seed') == seed:
                manifest[i2] = new_entry
                replaced = True
                break
        if not replaced:
            manifest.append(new_entry)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'  -> {"OK" if ok else "FAIL"} in {elapsed:.0f}s | coop={coop} | total: {(time.time()-start)/3600:.2f}h')
        if not ok:
            print(f'  STDERR: {proc.stderr[-300:]}')
    except subprocess.TimeoutExpired:
        print(f'  -> TIMEOUT after {time.time()-t0:.0f}s')
        manifest.append({'probe': pname, 'obs': obs, 'seed': seed, 'ok': False, 'error': 'timeout'})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

print(f'\nALL DONE in {(time.time()-start)/3600:.2f}h')
n_ok = sum(1 for m in manifest if m.get('ok'))
print(f'OK: {n_ok}/{len(manifest)} total')