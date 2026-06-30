"""Re-run D_exploration + E_baseline Intern trials one at a time.
Slower per-trial config (N=10, G=5, R=15) so each trial finishes in ~10-12 min."""
import os, sys, time, subprocess, json
from pathlib import Path

REPO = r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation'
OUT = Path(REPO) / 'results' / 'exp8_intern_ceiling'
OUT.mkdir(parents=True, exist_ok=True)

manifest_path = OUT / '_manifest.json'
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
else:
    manifest = []
done_keys = {(m['probe'], m['obs'], m['seed']) for m in manifest if m.get('ok')}

# D: exploration mutation (small config)
D_ARGS = ['--population', '10', '--generations', '5', '--rounds', '15',
          '--elitism', '1', '--tournament', '2', '--eliminate', '2',
          '--exploration-mutation']
# E: baseline (small config)
E_ARGS = ['--population', '10', '--generations', '5', '--rounds', '15',
          '--elitism', '1', '--tournament', '2', '--eliminate', '2']

PLAN = []
for obs in ['full', 'partial_0.7']:
    for seed in range(3):
        PLAN.append(('D_exploration', obs, seed, D_ARGS))
        PLAN.append(('E_baseline', obs, seed, E_ARGS))
PLAN = [p for p in PLAN if (p[0], p[1], p[2]) not in done_keys]
print(f'Trials to run: {len(PLAN)}')

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
           'LLM_MUTATION_WORKERS': '3'}
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