"""Dry run #2: ask the LLM to generate 4 strategies with the new interface,
print what it generated, and classify whether each strategy uses the
new observation fields.
"""
import os
import sys
import json
import re
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / '.env')

from experiments.evolution.population import EvolutionaryPopulation
from experiments.sandbox.validator import clean_code, validate_strategy_code, CodeValidationError

pop = EvolutionaryPopulation(
    population_size=4,
    num_rounds_per_gen=30,
    benefit=2.0,
    cost=1.0,
    observability='private',
    observability_p=0.0,
    elite_count=0,
    num_eliminate=0,
    tournament_size=0,
    llm_provider='openai',
    llm_model='deepseek-v4-flash',
    api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
    api_base_url=os.environ.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1'),
    seed=0,
    results_dir=str(REPO / 'results' / 'gen0_with_reputation_fields')
)

print("Asking LLM to generate 4 strategies with the augmented interface...")
agents = pop.initialize_population()
print(f"Generated {len(agents)} strategies\n")

# Save the strategies for inspection
out_dir = REPO / 'results' / 'gen0_with_reputation_fields'
out_dir.mkdir(parents=True, exist_ok=True)
out = {
    'n_agents': len(agents),
    'model': 'deepseek-v4-flash',
    'interface': 'with donor_reputation + recipient_reputation in observation',
    'strategies': [{'agent_id': a.agent_id, 'code': a.code} for a in agents]
}
import json
(out_dir / 'gen0_population.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(f'Saved to {out_dir / "gen0_population.json"}\n')

for i, a in enumerate(agents):
    code = a.code
    uses_donor_rep = 'donor_reputation' in code
    uses_recipient_rep = 'recipient_reputation' in code and 'recipient_reputation' in code.split('def decide')[0]  # only in evaluate
    uses_both = uses_donor_rep and uses_recipient_rep
    print(f"--- Strategy {i} (len={len(code)}, validation=ok) ---")
    print(f"  uses donor_reputation: {uses_donor_rep}")
    print(f"  uses recipient_reputation (in evaluate): {uses_recipient_rep}")
    print(f"  -> {'LEADING-8 CAPABLE' if uses_recipient_rep else 'IS-style only'}")
    print(code[:500])
    print()
