"""Single-seed runner with incremental save (every 10 gens).

Used so we can launch 3 independent background tasks (one per seed),
each in its own shell session, avoiding the 30-min bash timeout.

Also dumps the summary to a per-seed file as the run progresses, so
even mid-run we can read the latest trajectory.
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

ROOT = Path(r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUT_BASE = ROOT / 'results' / 'quantitative_baseline'
LABEL = "LLM_v3_g100_1000inter"
NUM_GENS = 100
SAVE_EVERY = 10  # save intermediate JSON every N generations

api_key = get_api_key('deepseek')
base_url = get_base_url('deepseek')

# seed is passed as argv[1]
seed = int(sys.argv[1])
seed_dir = OUT_BASE / f"{LABEL}_seed{seed}"
seed_dir.mkdir(parents=True, exist_ok=True)
out_path = seed_dir / 'evolutionary.json'
partial_path = seed_dir / 'evolutionary.partial.json'

print(f"=== {LABEL} seed={seed} start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

t0 = time.time()
try:
    pop = V2EvolutionaryPopulation(
        population_size=15,
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
        agent_type='v3',
        llm_thinking=False,
    )
    result = pop.run_evolution(num_generations=NUM_GENS)
    elapsed = time.time() - t0
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    last = result['trajectory'][-1]
    print(f"=== seed {seed} DONE in {elapsed/60:.1f} min ===", flush=True)
    print(f"  final coop: {last['cooperation_rate_mean']:.3f}", flush=True)
    print(f"  final fitness: {last['fitness_mean']:.1f}", flush=True)
    print(f"  FALLBACK init/mut: {result['config']['fallback_init_count']}/"
          f"{result['config']['fallback_mutation_count']}", flush=True)
except Exception as e:
    elapsed = time.time() - t0
    print(f"=== seed {seed} FAILED after {elapsed/60:.1f} min ===", flush=True)
    print(f"  {type(e).__name__}: {e}", flush=True)
    print(traceback.format_exc(), flush=True)
