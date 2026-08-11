"""Invasibility sweep: invader type × 8 leading-eight ESS × n=1..7.

Sweep runs invasion experiments for one invader type (e.g. 'ALLC', 'ALLD',
'LLM_winner') against each of 8 leading-eight ESS residents, at invader
counts n=1..7 (i.e. 1/15 to 7/15).

Usage:
  python _run_invasion_sweep.py ALLD                    # default: all 8 ESS × n=1..7
  python _run_invasion_sweep.py LLM_winner 2 6          # n=2..6 only (to fill in gaps)
  python _run_invasion_sweep.py ALLD 1 7 IS SS SH       # subset of ESS

Each (ess, n) combination is a separate run; output saved to
results/quantitative_baseline/invasion/<invader>_vs_<ESS>_n<n>_seed<seed>/invasion.json
"""
import json
import sys
import time
import random
import statistics
from pathlib import Path
from collections import Counter

ROOT = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
sys.path.insert(0, str(ROOT))
from experiments.v2_quantitative.baselines import get_baseline, BASELINES

OUT_BASE = ROOT / 'results' / 'quantitative_baseline' / 'invasion'

# The LLM winner in v2 (type-1) form — semantically faithful to seed2 agent9.
LLM_WINNER_V2 = '''
def evaluate(target_reputation, target_action, my_reputation):
    if target_reputation < 0.3:
        return -0.2
    return 0.5

def decide(my_reputation, opponent_reputation):
    if opponent_reputation < 0.25:
        return False
    return True
'''

INVADERS = {
    'ALLC':       None,  # use baseline ALLC
    'ALLD':       None,  # use baseline ALLD
    'LLM_winner': LLM_WINNER_V2,
}

LEADING_EIGHT = ['IS', 'SS', 'SJ', 'SC', 'SH', 'IS+', 'SS+', 'SJ+']

NUM_GENS = 50
N = 15
SEED = 42


def get_invader_code(invader_name: str) -> str:
    if invader_name in BASELINES:
        return get_baseline(invader_name)
    if invader_name in INVADERS and INVADERS[invader_name] is not None:
        return INVADERS[invader_name]
    raise KeyError(f'Unknown invader: {invader_name}')


def run_invasion(resident_baseline, resident_count, invader_code,
                 invader_count, num_gens, seed):
    """Run a single invasion experiment. Pure Python, no LLM."""
    from experiments.v2_quantitative.population import QuantitativeAgent

    resident_code = get_baseline(resident_baseline)
    rng = random.Random(seed)
    agents = []
    for i in range(resident_count):
        agents.append(QuantitativeAgent(i, resident_code))
    for i in range(invader_count):
        agents.append(QuantitativeAgent(resident_count + i, invader_code))

    trajectory = []
    for gen in range(num_gens + 1):
        if gen > 0:
            agents = _run_one_gen(agents, rng, resident_code, invader_code)
        n_inv = sum(1 for a in agents if a.code.strip() == invader_code.strip())
        invader_freq = n_inv / len(agents) if agents else 0
        coops = []
        for a in agents:
            c = 0
            for _ in range(10):
                opp = rng.choice([x for x in agents if x.agent_id != a.agent_id])
                rep_a = a.reputations.get(opp.agent_id, 0.5) if hasattr(a, 'reputations') else 0.5
                rep_opp = opp.reputations.get(a.agent_id, 0.5) if hasattr(opp, 'reputations') else 0.5
                try:
                    if a.decide(rep_a, rep_opp):
                        c += 1
                except Exception:
                    pass
            coops.append(c / 10)
        mean_coop = statistics.mean(coops) if coops else 0
        trajectory.append({
            'generation': gen,
            'invader_freq': invader_freq,
            'mean_coop': mean_coop,
            'n_residents': resident_count,
            'n_invaders': invader_count,
        })
    return trajectory


def _run_one_gen(agents, rng, resident_code, invader_code):
    from experiments.v2_quantitative.population import QuantitativeAgent
    N = len(agents)
    for a in agents:
        if not hasattr(a, 'reputations') or a.reputations is None:
            a.reputations = {}
    order = list(range(N))
    rng.shuffle(order)
    pairs = []
    for i in range(0, N - 1, 2):
        pairs.append((order[i], order[i + 1]))
    fitness = {a.agent_id: 0.0 for a in agents}
    for (i, j) in pairs:
        ai, aj = agents[i], agents[j]
        rep_i_j = ai.reputations.get(aj.agent_id, 0.5)
        rep_j_i = aj.reputations.get(ai.agent_id, 0.5)
        try:
            c_i = ai.decide(rep_i_j, rep_j_i)
        except Exception:
            c_i = True
        try:
            c_j = aj.decide(rep_j_i, rep_i_j)
        except Exception:
            c_j = True
        if c_i and c_j:
            f_i, f_j = 1.0, 1.0
        elif c_i and not c_j:
            f_i, f_j = -1.0, 2.0
        elif not c_i and c_j:
            f_i, f_j = 2.0, -1.0
        else:
            f_i, f_j = 0.0, 0.0
        ai.reputations[aj.agent_id] = max(0.0, min(1.0,
            ai.reputations.get(aj.agent_id, 0.5) + (0.1 if c_j else -0.1)))
        aj.reputations[ai.agent_id] = max(0.0, min(1.0,
            aj.reputations.get(ai.agent_id, 0.5) + (0.1 if c_i else -0.1)))
        fitness[ai.agent_id] += f_i
        fitness[aj.agent_id] += f_j
    survivors = sorted(agents, key=lambda a: fitness[a.agent_id], reverse=True)
    new_agents = []
    for elite in survivors[:2]:
        new_agents.append(QuantitativeAgent(len(new_agents), elite.code))
    while len(new_agents) < N:
        ts = rng.sample(agents, 3)
        winner = max(ts, key=lambda a: fitness[a.agent_id])
        new_agents.append(QuantitativeAgent(len(new_agents), winner.code))
    return new_agents


def safe_ess_name(ess: str) -> str:
    return ess  # Already uses _PLUS convention


def main():
    args = sys.argv[1:]
    if not args:
        print('Usage: _run_invasion_sweep.py <invader> [n_min] [n_max] [ess1 ess2 ...]')
        print('  invader: ALLC, ALLD, LLM_winner')
        print('  default: all 8 ESS × n=1..7')
        sys.exit(1)

    invader_name = args[0]
    n_min = int(args[1]) if len(args) > 1 else 1
    n_max = int(args[2]) if len(args) > 2 else 7
    ess_list = args[3:] if len(args) > 3 else LEADING_EIGHT

    if invader_name not in BASELINES and invader_name not in INVADERS:
        print(f'Unknown invader: {invader_name}. Available: {list(BASELINES) + list(INVADERS)}')
        sys.exit(1)

    invader_code = get_invader_code(invader_name)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f'=== Invasion sweep: {invader_name} vs {len(ess_list)} ESS × n={n_min}..{n_max} ===', flush=True)
    print(f'Total runs: {len(ess_list) * (n_max - n_min + 1)}', flush=True)
    t0_total = time.time()
    summary = []

    for ess in ess_list:
        for n in range(n_min, n_max + 1):
            out_dir = OUT_BASE / f'{invader_name}_vs_{safe_ess_name(ess)}_n{n}_seed{SEED}'
            out_path = out_dir / 'invasion.json'
            if out_path.exists():
                # Skip if already done
                d = json.loads(out_path.read_text(encoding='utf-8'))
                g_end = d['trajectory'][-1]['invader_freq']
                print(f'  [SKIP] {ess} n={n} (exists, g_end_freq={g_end:.3f})', flush=True)
                summary.append((ess, n, g_end, 'cached'))
                continue

            t0 = time.time()
            traj = run_invasion(ess, resident_count=N - n,
                                invader_code=invader_code,
                                invader_count=n, num_gens=NUM_GENS, seed=SEED)
            elapsed = time.time() - t0

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump({
                    'target': ess,
                    'invader': invader_name,
                    'invader_count': n,
                    'num_gens': NUM_GENS,
                    'seed': SEED,
                    'trajectory': traj,
                    'elapsed_sec': elapsed,
                }, f, indent=2)

            g0 = traj[0]['invader_freq']
            g_end = traj[-1]['invader_freq']
            print(f'  [END] {ess} n={n}: g0={g0:.3f} -> g_end={g_end:.3f} ({elapsed:.1f}s)', flush=True)
            summary.append((ess, n, g_end, 'new'))

    elapsed_total = time.time() - t0_total
    print(f'\n=== Sweep done in {elapsed_total:.1f}s ===', flush=True)
    print('Final invader frequencies:', flush=True)
    for ess, n, g_end, status in summary:
        verdict = 'FIXATE' if g_end >= 0.9 else ('DRIFT' if g_end >= 0.3 else 'DIES')
        print(f'  {ess:10s} n={n}: g_end={g_end:.3f} -> {verdict} ({status})', flush=True)


if __name__ == '__main__':
    main()
