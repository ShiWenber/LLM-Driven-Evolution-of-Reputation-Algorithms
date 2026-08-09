"""Ablation A single-seed runner with adversarial (game-theoretic) prompt.

Monkey-patches INIT_PROMPT_V3 / MUTATION_PROMPT_V3 BEFORE importing
V2EvolutionaryPopulation, so the LLM is given a strict-PD framing
instead of the no-hints neutral framing.

Same production config as the main 3-seed 1000-inter run:
  - N=15, target=1000 inter/gen => 143 rounds/gen
  - benefit=2, cost=1 (PD)
  - observability=full
  - thinking=off, max_tokens=4000
  - 100 generations

Saves to LLM_v3_g100_1000inter_ADVERSARIAL_seed{seed}/.
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

# CRITICAL: monkey-patch BEFORE importing population
import experiments.v2_quantitative.prompts as prompts_mod
from _abl_prompts import ABL_INIT_PROMPT_V3, ABL_MUTATION_PROMPT_V3
prompts_mod.INIT_PROMPT_V3 = ABL_INIT_PROMPT_V3
prompts_mod.MUTATION_PROMPT_V3 = ABL_MUTATION_PROMPT_V3

from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

ROOT = Path(r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
OUT_BASE = ROOT / 'results' / 'quantitative_baseline'
LABEL = "LLM_v3_g100_1000inter_ADVERSARIAL"
NUM_GENS = 100

api_key = get_api_key('deepseek')
base_url = get_base_url('deepseek')

seed = int(sys.argv[1])
seed_dir = OUT_BASE / f"{LABEL}_seed{seed}"
seed_dir.mkdir(parents=True, exist_ok=True)
out_path = seed_dir / 'evolutionary.json'

print(f"=== {LABEL} seed={seed} start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
print(f"  prompt: ADVERSARIAL (PD with public reputation, max-individual-payoff)", flush=True)

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
