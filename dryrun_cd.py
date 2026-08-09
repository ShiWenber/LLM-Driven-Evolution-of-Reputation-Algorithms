"""Verify cooperate/defect label: 1) LLM can write code with the new labels,
2) one full trial runs end-to-end."""
import os, sys, time
sys.path.insert(0, r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\.env')

from experiments.evolution.population import EvolutionaryPopulation
from pathlib import Path
REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')

# Generate 4 strategies via LLM
pop = EvolutionaryPopulation(
    population_size=4, num_rounds_per_gen=30, benefit=2.0, cost=1.0,
    observability='private', observability_p=0.0,
    elite_count=0, num_eliminate=0, tournament_size=0,
    llm_provider='openai', llm_model='deepseek-v4-flash',
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    api_base_url='https://api.deepseek.com/v1',
    seed=0,
    results_dir=str(REPO / 'results' / 'dryrun_cd')
)

print("Asking LLM to generate 4 strategies with the 'cooperate'/'defect' interface...")
agents = pop.initialize_population()
print(f"Generated {len(agents)} strategies\n")

for i, a in enumerate(agents):
    code = a.code
    has_cooperate = "'cooperate'" in code or '"cooperate"' in code
    has_defect = "'defect'" in code or '"defect"' in code
    has_return_true = 'return True' in code
    has_return_false = 'return False' in code
    print(f"--- Strategy {i} (len={len(code)}) ---")
    print(f"  uses 'cooperate': {has_cooperate}")
    print(f"  uses 'defect':    {has_defect}")
    print(f"  return True: {has_return_true}")
    print(f"  return False: {has_return_false}")
    print(f"  code:")
    print(code[:600] + ('...' if len(code) > 600 else ''))
    print()