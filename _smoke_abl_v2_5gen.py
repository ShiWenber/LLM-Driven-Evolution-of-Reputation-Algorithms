"""5-gen smoke for Ablation A v2: verify prompt patch works WITHOUT canonical hints."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r'C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

import experiments.v2_quantitative.prompts as prompts_mod
from _abl_prompts_v2 import ABL2_INIT_PROMPT_V3, ABL2_MUTATION_PROMPT_V3
prompts_mod.INIT_PROMPT_V3 = ABL2_INIT_PROMPT_V3
prompts_mod.MUTATION_PROMPT_V3 = ABL2_MUTATION_PROMPT_V3

from experiments.config.load_env import get_api_key, get_base_url
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

api_key = get_api_key('deepseek')
base_url = get_base_url('deepseek')

print("=== Ablation A v2 5-gen smoke @", time.strftime('%H:%M:%S'), "===", flush=True)
print("  prompt: ADVERSARIAL v2 (no canonical-answer hints)", flush=True)

t0 = time.time()
pop = V2EvolutionaryPopulation(
    population_size=15,
    num_rounds_per_gen=30,
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
    seed=999,
    results_dir='results/quantitative_baseline',
    use_baseline=None,
    agent_type='v3',
    llm_thinking=False,
)
result = pop.run_evolution(num_generations=5)
elapsed = time.time() - t0
print(f"=== 5-gen smoke DONE in {elapsed/60:.1f} min ===", flush=True)
for g in result['trajectory']:
    print(f"  gen {g['generation']}: coop={g['cooperation_rate_mean']:.3f} fit={g['fitness_mean']:.1f}", flush=True)
print(f"  FALLBACK: init={result['config']['fallback_init_count']}/15, mut={result['config']['fallback_mutation_count']}/25", flush=True)

agent0 = result['trajectory'][0]['population'][0]
print(f"\n--- agent 0 init strategy (first 1500 chars) ---", flush=True)
print(agent0['code'][:1500], flush=True)
