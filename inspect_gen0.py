"""Inspect the gen-0 initial population composition.

Runs a fresh init-prompt call to the LLM and saves the 15 generated strategies
so we can classify them. Cheap (~$0.01), ~30s.
"""
import os
import sys
import json
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
sys.path.insert(0, str(REPO))

# Load .env
from dotenv import load_dotenv
load_dotenv(REPO / '.env')

from experiments.evolution.population import EvolutionaryPopulation
from experiments.agents.prompts import build_init_prompt
from experiments.sandbox.validator import clean_code, validate_strategy_code, CodeValidationError

# Build a population but only call initialize_population()
pop = EvolutionaryPopulation(
    population_size=15,
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
    results_dir=str(REPO / 'results' / 'gen0_inspection')
)

agents = pop.initialize_population()

# Save gen-0 strategies
out = {
    'n_agents': len(agents),
    'model': 'deepseek-v4-flash',
    'observability_p': 0.0,
    'strategies': [
        {
            'agent_id': a.agent_id,
            'code': a.code,
        }
        for a in agents
    ]
}

out_dir = REPO / 'results' / 'gen0_inspection'
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'gen0_population.json'
out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(f'Saved {len(agents)} gen-0 strategies to {out_path}')

# Quick classification summary
import re
OBS_KEY_DOUBLE = 'observation["action"]'
OBS_KEY_SINGLE = "observation['action']"

def classify(code):
    if not code: return 'Other'
    has_observe_action = (OBS_KEY_DOUBLE in code) or (OBS_KEY_SINGLE in code)
    has_threshold = ('recipient_reputation' in code) and (re.search(r'>=|>|==|<|<=', code) is not None)
    has_my_history = 'my_history' in code
    has_return_true = re.search(r'return\s+True\b', code) and not has_threshold and ('random' not in code)
    has_return_false = re.search(r'return\s+False\b', code) and not has_threshold and ('random' not in code)
    has_random = 'random' in code
    has_round = 'round_num' in code
    if has_return_true and not has_threshold and not has_random: return 'ALLC'
    if has_return_false and not has_threshold and not has_random: return 'ALLD'
    if has_observe_action and has_threshold and not has_my_history: return 'ImageScoring'
    if has_observe_action and has_threshold and has_my_history: return 'Hybrid'
    if has_random and not has_threshold: return 'RandomStrategy'
    if has_threshold and not has_observe_action: return 'ThresholdOnly'
    if has_my_history and not has_threshold: return 'DirectExperience'
    if has_round and not has_threshold: return 'RoundDependent'
    return 'Other'

from collections import Counter
dist = Counter(classify(a.code) for a in agents)
print('\n=== GEN-0 strategy distribution (n=15) ===')
for cls, c in dist.most_common():
    print(f'  {cls:20s} {c:3d}  ({100*c/len(agents):.1f}%)')

# Print each strategy's classification
print('\n=== Per-agent ===')
for a in agents:
    cls = classify(a.code)
    # Get short fingerprint
    has_obs = 'obs' in a.code and 'action' in a.code
    has_hist = 'my_history' in a.code
    has_round = 'round_num' in a.code
    has_rand = 'random' in a.code
    print(f'  Agent {a.agent_id:2d}: {cls:20s}  obs={has_obs} hist={has_hist} round={has_round} rand={has_rand}  code_len={len(a.code)}')
