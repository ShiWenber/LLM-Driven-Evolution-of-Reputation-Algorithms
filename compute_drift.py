"""Compute actual static vs LLM-evo drift from real data."""
import json
from pathlib import Path
import statistics

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

def collect_static_drift(obs):
    """Get gen-0 and gen-9 cooperation from static trials."""
    base = REPO / 'results' / 'exp3_static_g10_n10'
    drifts = []
    for seed in range(10):
        d = base / f'{obs}_seed{seed}'
        if not d.exists(): continue
        jsons = [f for f in d.iterdir() if f.name.startswith('static_control_') and f.name.endswith('.json')]
        if not jsons: continue
        jsons.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        with open(jsons[0]) as f:
            t = json.load(f)
        if 'trials_summary' in t and t['trials_summary']:
            traj = t['trials_summary'][0].get('trajectory', [])
            if len(traj) >= 2:
                drifts.append(traj[-1]['cooperation_rate_mean'] - traj[0]['cooperation_rate_mean'])
    return drifts

def collect_llm_drift(obs):
    """Get gen-0 and gen-9 cooperation from LLM-evo main plan.
    Pick evo_*.json (per-trial full trajectory), not evolutionary_*.json (aggregate)."""
    base = REPO / 'results' / 'exp1_method_n10'
    drifts = []
    for seed in range(10):
        d = base / f'{obs}_seed{seed}'
        if not d.exists(): continue
        # evo_<obs>_<model>_<timestamp>.json has per-trial trajectory
        evos = [f for f in d.iterdir() if f.name.startswith('evo_') and f.name.endswith('.json')]
        if not evos: continue
        evos.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        with open(evos[0]) as f:
            t = json.load(f)
        traj = t.get('trajectory', [])
        if len(traj) >= 2:
            drifts.append(traj[-1]['cooperation_rate_mean'] - traj[0]['cooperation_rate_mean'])
    return drifts

print(f'{"obs":12s} {"static drift (mean)":>20s} {"LLM-evo drift (mean)":>22s} {"gap":>10s}')
print('-' * 70)
for obs in ['private', 'partial_0.3', 'partial_0.7', 'full']:
    s = collect_static_drift(obs)
    l = collect_llm_drift(obs)
    s_mean = statistics.mean(s) if s else None
    l_mean = statistics.mean(l) if l else None
    print(f'{obs:12s} {f"{s_mean:+.3f} (n={len(s)})":>20s} {f"{l_mean:+.3f} (n={len(l)})":>22s} '
          f'{f"{(l_mean - s_mean):+.3f}" if s_mean and l_mean else "n/a":>10s}')