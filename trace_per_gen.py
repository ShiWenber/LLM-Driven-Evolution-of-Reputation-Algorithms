"""For each trial, count how many strategies used recipient_reputation per generation.
This tells us WHEN (which gen) the LLM-generated rec_rep variants died out."""
import json, re
from pathlib import Path
RES = Path(r'C:\Users\shiwenbo\.mavis\workspace\llm-reputation-paper\llm-reputation\results')

def has_recipient_rep(code):
    if 'def evaluate' not in code: return False
    m = re.search(r'def evaluate\([^)]*\):(.*?)(?=\ndef |\Z)', code, re.S)
    if not m: return False
    return 'recipient_reputation' in m.group(1)

# Pick 3 trials: one with success, one with mostly fail
trials_to_check = [
    'exp1_method_n10/partial_0.7_seed3/evo_partial_0.7_deepseek-v4-flash_20260610_080750.json',
    'exp5_robustness/partial_0.7_seed0/evo_partial_0.7_deepseek-coder_20260608_225137.json',
    'exp6_sweep_AB_n5/full_seed4/evo_full_deepseek-v4-flash_20260618_133423.json',
    'exp6_sweep_AB_n5/partial_0.7_seed3/evo_partial_0.7_deepseek-v4-flash_20260618_125638.json',
    'exp6_sweep_AB_n5/private_seed4/evo_private_deepseek-v4-flash_20260618_131145.json',
]
RES = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results')

for trial_path in trials_to_check:
    td = RES / trial_path
    if not td.exists():
        print(f'MISSING: {trial_path}'); continue
    d = json.loads(td.read_text(encoding='utf-8', errors='ignore'))
    traj = d.get('trajectory', [])
    print(f'\n{"="*70}')
    print(f'Path: {trial_path}')
    print(f'{"gen":>4s} {"n_pop":>5s} {"n_rec":>5s} {"n_rec>0coop":>12s} {"max_coop":>8s} {"mean_coop":>10s}')
    for g in traj:
        pop = g.get('population', [])
        n_rec = sum(1 for a in pop if has_recipient_rep(a.get('code','')))
        rec_coops = [a.get('cooperation_rate', 0) or 0 for a in pop if has_recipient_rep(a.get('code',''))]
        coops = [a.get('cooperation_rate', 0) or 0 for a in pop]
        n_rec_high = sum(1 for c in rec_coops if c > 0.05)
        max_c = max(coops) if coops else 0
        mean_c = sum(coops)/len(coops) if coops else 0
        print(f'{g.get("generation","?"):>4d} {len(pop):>5d} {n_rec:>5d} {n_rec_high:>12d} {max_c:>8.3f} {mean_c:>10.3f}')
