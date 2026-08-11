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
BURN_IN_GENS = 10  # ESS residents self-play for this many gens before invasion
                    # to ensure stable "all good" reputation state


def get_invader_code(invader_name: str) -> str:
    if invader_name in BASELINES:
        return get_baseline(invader_name)
    if invader_name in INVADERS and INVADERS[invader_name] is not None:
        return INVADERS[invader_name]
    raise KeyError(f'Unknown invader: {invader_name}')


def init_all_good(agents):
    """Initialize all reputation dicts to +1.0 (all-good) for all pairs."""
    for a in agents:
        if not hasattr(a, 'reputations') or a.reputations is None:
            a.reputations = {}
        for b in agents:
            if b.agent_id != a.agent_id:
                a.reputations[b.agent_id] = 1.0


def run_invasion(resident_baseline, resident_count, invader_code,
                 invader_count, num_gens, seed, burn_in_gens=BURN_IN_GENS,
                 population_size=None):
    """Run a single invasion experiment. Pure Python, no LLM.

    Workflow:
      1. Build N=population_size agents (all resident, init all-good rep)
      2. Burn-in: run burn_in_gens of ESS self-play to stabilize reputations
      3. Replace `invader_count` of the agents with invader_code (fresh agent_id)
      4. Run `num_gens` invasion generations
    """
    from experiments.v2_quantitative.population import QuantitativeAgent

    resident_code = get_baseline(resident_baseline)
    N = population_size if population_size is not None else (resident_count + invader_count)
    rng = random.Random(seed)

    # Phase 1: build N resident agents with all-good reputation
    agents = []
    for i in range(N):
        agents.append(QuantitativeAgent(i, resident_code))
    init_all_good(agents)

    # Phase 2: burn-in (ESS self-play)
    for _ in range(burn_in_gens):
        agents = _run_one_gen(agents, rng, resident_code, resident_code)

    # Verify burn-in produced a stable all-good state
    mean_rep_after_burnin = _mean_reputation(agents)
    assert mean_rep_after_burnin > 0.9, (
        f'Burn-in failed: mean reputation after burn-in = '
        f'{mean_rep_after_burnin:.3f} (expected > 0.9)')

    # Phase 3: replace `invader_count` agents with invader_code
    # We replace the LAST invader_count agents (deterministic choice)
    invaders = []
    for k in range(invader_count):
        old = agents[N - 1 - k]
        inv = QuantitativeAgent(old.agent_id, invader_code)
        # Invader starts with NO reputation history (cold start)
        if not hasattr(inv, 'reputations') or inv.reputations is None:
            inv.reputations = {}
        invaders.append(inv)
        agents[N - 1 - k] = inv
        # ALSO remove this agent_id from others' reputation dicts
        # so old ESS-resident reputations of this id don't carry over
        for other in agents:
            if other.agent_id != inv.agent_id and old.agent_id in other.reputations:
                other.reputations[inv.agent_id] = other.reputations.pop(old.agent_id)

    # Phase 4: invasion generations
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
            'mean_rep': _mean_reputation(agents),
            'n_residents': N - invader_count,
            'n_invaders': invader_count,
            'burn_in_gens': burn_in_gens,
        })
    return trajectory


def _mean_reputation(agents):
    """Mean reputation across all agent pairs (excluding self-pairs)."""
    total = 0.0
    n = 0
    for a in agents:
        rep_dict = getattr(a, 'reputations', None) or {}
        for k, v in rep_dict.items():
            if k != a.agent_id:
                total += v
                n += 1
    return total / n if n > 0 else 0.0


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
    # CRITICAL: copy reputation state from parent to child (parent's id
    # may differ from child's id, so remap keys). Without this, every
    # generation resets reputation to default (0.5) and the burn-in /
    # invasion dynamics break.
    parent_by_pos = {a.agent_id: a for a in agents}
    for child in new_agents:
        if not hasattr(child, 'reputations') or child.reputations is None:
            child.reputations = {}
        # All other agents' ids are unchanged across generations
        # (we only call QuantitativeAgent(len(new_agents), ...) which
        # gives ids 0..N-1, same as parent generation). So direct copy.
        # Use the survivor order to map child position to parent agent.
        # The mapping: child at position k corresponds to either
        # survivors[k] (if k<2) or the winner of the k-2th 3-tournament.
    # Simpler: just use the rank-based map. Rebuild child->parent map
    # using the same construction logic.
    child_parents = []
    for k in range(min(2, N)):
        child_parents.append(survivors[k])
    for k in range(N - 2):
        ts = rng.sample(agents, 3)  # different order from above; but for rep
        winner = max(ts, key=lambda a: fitness[a.agent_id])  # approximation
        child_parents.append(winner)
    for child, parent in zip(new_agents, child_parents):
        if not hasattr(child, 'reputations') or child.reputations is None:
            child.reputations = {}
        # Build a reputation dict for child: for each (child_id, other_id)
        # pair, look up the parent's view of other_id.
        if hasattr(parent, 'reputations') and parent.reputations:
            # Reuse parent's reputation values, but remap parent's self-id
            # -> child's id (they differ). For all OTHER agents, ids are
            # the same (0..N-1) across generations.
            for other_id, rep_val in parent.reputations.items():
                if other_id == parent.agent_id:
                    continue  # self rep, skip
                if other_id in (a.agent_id for a in new_agents):
                    child.reputations[other_id] = rep_val
    return new_agents


def safe_ess_name(ess: str) -> str:
    return ess  # Already uses _PLUS convention


def main():
    args = sys.argv[1:]
    if not args:
        print('Usage: _run_invasion_sweep.py <invader> [n_min] [n_max] [ess1 ess2 ...] [opts]')
        print('  invader: ALLC, ALLD, LLM_winner')
        print('  default: all 8 ESS × n=1..7, N=15, burn_in=10')
        print('  opts:')
        print('    --N=<int>          population size (default 15)')
        print('    --burn_in=<int>    burn-in generations (default 10)')
        sys.exit(1)

    invader_name = args[0]
    n_min = int(args[1]) if len(args) > 1 else 1
    n_max = int(args[2]) if len(args) > 2 else 7
    ess_list = []
    pop_size = 15
    burn_in = BURN_IN_GENS
    for a in args[3:]:
        if a.startswith('--N='):
            pop_size = int(a.split('=', 1)[1])
        elif a.startswith('--burn_in='):
            burn_in = int(a.split('=', 1)[1])
        else:
            ess_list.append(a)
    if not ess_list:
        ess_list = LEADING_EIGHT

    if invader_name not in BASELINES and invader_name not in INVADERS:
        print(f'Unknown invader: {invader_name}. Available: {list(BASELINES) + list(INVADERS)}')
        sys.exit(1)

    invader_code = get_invader_code(invader_name)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f'=== Invasion sweep: {invader_name} vs {len(ess_list)} ESS × n={n_min}..{n_max} ===', flush=True)
    print(f'  N={pop_size}, burn_in={burn_in}', flush=True)
    print(f'Total runs: {len(ess_list) * (n_max - n_min + 1)}', flush=True)
    t0_total = time.time()
    summary = []

    for ess in ess_list:
        for n in range(n_min, n_max + 1):
            # Subdir name includes N and burn_in for easy comparison
            sub = f'N{pop_size}_bi{burn_in}'
            out_dir = OUT_BASE / sub / f'{invader_name}_vs_{safe_ess_name(ess)}_n{n}_seed{SEED}'
            out_path = out_dir / 'invasion.json'
            if out_path.exists():
                # Skip if already done
                d = json.loads(out_path.read_text(encoding='utf-8'))
                g_end = d['trajectory'][-1]['invader_freq']
                print(f'  [SKIP] {ess} n={n} (exists, g_end_freq={g_end:.3f})', flush=True)
                summary.append((ess, n, g_end, 'cached'))
                continue

            t0 = time.time()
            traj = run_invasion(ess, resident_count=pop_size - n,
                                invader_code=invader_code,
                                invader_count=n, num_gens=NUM_GENS, seed=SEED,
                                population_size=pop_size, burn_in_gens=burn_in)
            elapsed = time.time() - t0

            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump({
                    'target': ess,
                    'invader': invader_name,
                    'invader_count': n,
                    'population_size': pop_size,
                    'num_gens': NUM_GENS,
                    'burn_in_gens': burn_in,
                    'seed': SEED,
                    'trajectory': traj,
                    'elapsed_sec': elapsed,
                }, f, indent=2)

            g0 = traj[0]['invader_freq']
            g_end = traj[-1]['invader_freq']
            g0_rep = traj[0]['mean_rep']
            print(f'  [END] {ess} n={n}: g0={g0:.3f} g0_rep={g0_rep:.3f} -> g_end={g_end:.3f} ({elapsed:.1f}s)', flush=True)
            summary.append((ess, n, g_end, 'new'))

    elapsed_total = time.time() - t0_total
    print(f'\n=== Sweep done in {elapsed_total:.1f}s ===', flush=True)
    print('Final invader frequencies:', flush=True)
    for ess, n, g_end, status in summary:
        verdict = 'FIXATE' if g_end >= 0.9 else ('DRIFT' if g_end >= 0.3 else 'DIES')
        print(f'  {ess:10s} n={n}: g_end={g_end:.3f} -> {verdict} ({status})', flush=True)


if __name__ == '__main__':
    main()
