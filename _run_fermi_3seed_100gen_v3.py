"""Production run: 100 gen x target_interactions_per_gen=1000 x 1 seed.

M10: v4 prompts (truly minimal). Both init and mutation prompts cleaned.
- INIT_PROMPT_V3 stripped of: 'designing a strategy', 'Avoid random',
  'deterministic when possible', DS example list, 'for your strategy' meta.
- SMALL_MUTATION_PROMPT_V3 stripped of: 'recognizably related', 'within
  parent's strategy family', 5 mutation-type bullets, 'small new mechanism',
  simulation summary, 'hand-rolled reputation update'.

Both now keep only: API constraints + parent code (mutation) + framework
description (init).

Launch one seed per detached process. Run 3 in parallel.
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.config.load_env import get_api_key, get_base_url
from experiments.evolution_log import (
    evolution_json_path, run_dir, write_evolution_json,
)
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

OUT_BASE = ROOT / 'results' / 'quantitative_baseline'

api_key = get_api_key('deepseek')
base_url = get_base_url('deepseek')

# Agent type override via argv. Two calling conventions:
#   python _run_fermi_3seed_100gen_v3.py            -> agent_type=v3, seeds=[0,1,2]
#   python _run_fermi_3seed_100gen_v3.py 0          -> agent_type=v3, seeds=[0]
#   python _run_fermi_3seed_100gen_v3.py v2         -> agent_type=v2, seeds=[0,1,2]
#   python _run_fermi_3seed_100gen_v3.py 0 v2      -> agent_type=v2, seeds=[0]
AGENT_TYPE = 'v3'
if len(sys.argv) > 1 and sys.argv[1].lower() in ('v2', 'v3'):
    AGENT_TYPE = sys.argv[1].lower()
elif len(sys.argv) > 2:
    t = sys.argv[2].lower()
    if t in ('v2', 'v3'):
        AGENT_TYPE = t
    else:
        raise SystemExit(f"unknown agent_type {sys.argv[2]!r}; use 'v2' or 'v3'")

LABEL = f"LLM_{AGENT_TYPE}_fermi_z_v3_g100_1000inter_N16_genreset"

# Accept seed override via argv: pass a single int to run only that seed.
if len(sys.argv) > 1 and sys.argv[1].lower() not in ('v2', 'v3'):
    seeds = [int(sys.argv[1])]
else:
    seeds = [0, 1, 2]

NUM_GENS = 100

print(f"=== {LABEL} seeds={seeds} (sequential) ==", flush=True)
print(f"  api_key: {api_key[:8]}...{api_key[-4:]}", flush=True)
print(f"  num_gens: {NUM_GENS}, target_interactions: 1000, agent_type={AGENT_TYPE}", flush=True)
print(f"  Z-like: mu=0.1, beta=5.0, updates_per_gen=15", flush=True)
print(f"  prompts: v4 (truly minimal init + mutation)", flush=True)
print(f"  estimated: ~5-7h/seed x {len(seeds)} seed(s)", flush=True)
print(flush=True)

summary = []
overall_t0 = time.time()

for seed in seeds:
    seed_dir = run_dir(OUT_BASE, LABEL, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_path = evolution_json_path(OUT_BASE, LABEL, seed)
    if out_path.exists():
        out_path.unlink()
        print(f"  [seed {seed}] removed existing {out_path}", flush=True)

    print(f"=== seed {seed} start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    t0 = time.time()
    seed_summary = {'seed': seed, 'completed': False, 'error': None}

    try:
        pop = V2EvolutionaryPopulation(
            population_size=16,
            target_interactions_per_gen=1000,
            benefit=2.0,
            cost=1.0,
            observability='full',
            observability_p=1.0,
            elite_count=2,
            num_eliminate=5,
            tournament_size=3,
            llm_model='deepseek-v4-flash',
            api_key=api_key,
            api_base_url=base_url,
            mutation_temperature=0.8,
            seed=seed,
            results_dir=str(OUT_BASE),
            use_baseline=None,
            agent_type=AGENT_TYPE,
            llm_thinking=False,
            use_fermi=True,
            fermi_beta=5.0,
            mutation_rate_on_adoption=0.1,
            updates_per_gen=15,
            forbid_self_pairing=True,
        )
        result = pop.run_evolution(num_generations=NUM_GENS)
        elapsed = time.time() - t0

        write_evolution_json(out_path, result)

        last = result['trajectory'][-1]
        seed_summary.update({
            'completed': True,
            'elapsed_sec': elapsed,
            'elapsed_min': elapsed / 60,
            'final_coop': last['cooperation_rate_mean'],
            'final_fitness': last['fitness_mean'],
            'fallback_init': result['config']['fallback_init_count'],
            'fallback_mutation': result['config']['fallback_mutation_count'],
            'gen0_coop': result['trajectory'][0]['cooperation_rate_mean'],
        })
        print(f"=== seed {seed} done in {elapsed/60:.1f} min ===", flush=True)
        print(f"  gen 0 coop: {seed_summary['gen0_coop']:.3f}", flush=True)
        print(f"  final coop: {seed_summary['final_coop']:.3f}", flush=True)
        print(f"  final fitness: {seed_summary['final_fitness']:.1f}", flush=True)
        print(f"  FALLBACK: init={seed_summary['fallback_init']}/16, "
              f"mutation={seed_summary['fallback_mutation']}/{(NUM_GENS-1)*15}", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        seed_summary.update({
            'completed': False,
            'elapsed_sec': elapsed,
            'error': f"{type(e).__name__}: {e}",
            'traceback': traceback.format_exc(),
        })
        print(f"=== seed {seed} FAILED after {elapsed/60:.1f} min ===", flush=True)
        print(f"  {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    summary.append(seed_summary)
    summary_path = OUT_BASE / f"{LABEL}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'label': LABEL,
            'agent_type': AGENT_TYPE,
            'num_gens': NUM_GENS,
            'target_interactions_per_gen': 1000,
            'scheme': 'fermi_z_like (mu=0.1, beta=5.0, updates_per_gen=15)',
            'prompts': 'v4 (minimal init + minimal mutation)',
            'seeds': summary,
            'overall_elapsed_sec': time.time() - overall_t0,
        }, f, indent=2)

total_elapsed = time.time() - overall_t0
n_done = sum(1 for s in summary if s['completed'])
print(f"\n=== ALL SEEDS DONE in {total_elapsed/3600:.2f} h ({total_elapsed/60:.0f} min) ===", flush=True)
print(f"  {n_done}/{len(seeds)} seeds completed successfully", flush=True)
for s in summary:
    if s['completed']:
        print(f"  seed {s['seed']}: final_coop={s['final_coop']:.3f}, "
              f"final_fitness={s['final_fitness']:.1f}, "
              f"min={s['elapsed_min']:.1f}, "
              f"FALLBACK init/mut={s['fallback_init']}/{s['fallback_mutation']}")
    else:
        print(f"  seed {s['seed']}: FAILED ({s.get('error', '?')})")
