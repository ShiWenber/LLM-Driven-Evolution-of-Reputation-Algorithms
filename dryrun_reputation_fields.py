"""Dry run: verify the augmented observation dict has donor_reputation and
recipient_reputation fields, without calling the LLM.

We:
  1. Patch the LLM client to return a hard-coded simple strategy.
  2. Run a static control for G=1, R=2 with the patched strategy.
  3. Inspect what the evaluate() function actually receives.
"""
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / '.env')

# Monkey-patch the LLM client BEFORE importing main
SIMPLE_STRATEGY = '''
def evaluate(current_reputation, observation, my_history, round_num):
    # Inspect observation keys
    print(f"[eval] observation keys = {sorted(observation.keys())}")
    print(f"[eval]   donor={observation.get('donor')} recipient={observation.get('recipient')}")
    print(f"[eval]   donor_rep={observation.get('donor_reputation')} recipient_rep={observation.get('recipient_reputation')}")
    print(f"[eval]   action={observation.get('action')}")
    # Standard IS update
    if observation.get("action") == "donate":
        return min(1.0, current_reputation + 0.2)
    else:
        return max(-1.0, current_reputation - 0.2)


def decide(recipient_reputation, round_num, my_history):
    return recipient_reputation >= 0.0
'''

import experiments.evolution.population as pop_mod

original_init = pop_mod.EvolutionaryPopulation.initialize_population

def patched_init(self):
    print("[init] Patched initialize_population: skipping LLM, using hand-coded strategy")
    from experiments.agents.code_agent import CodeAgent
    from experiments.agents.prompts import build_init_prompt
    agents = []
    for i in range(self.population_size):
        a = CodeAgent(agent_id=i, code=SIMPLE_STRATEGY)
        a.generation = 0
        agents.append(a)
    self.agents = agents
    self.generation = 0
    return agents

pop_mod.EvolutionaryPopulation.initialize_population = patched_init

# Now run static control with very small params
import argparse
ns = argparse.Namespace(
    population=4,
    generations=1,
    rounds=2,
    seeds=1,
    output=str(REPO / 'results' / 'dryrun_reputation_fields'),
    models='deepseek-v4-flash',
    observability='full',  # full obs so we can see observations being distributed
    elitism=2,
    tournament=2,
    benefit=2.0,
    cost=1.0,
    eliminations=1,
    llm_provider='openai',
    api_key='',
    api_base='',
    run='static',
    p_values='1.0',
)

# Build the models list manually (use openai provider since deepseek is openai-compatible)
from experiments.config.load_env import get_api_key, get_base_url
models = [{
    'name': 'deepseek-v4-flash',
    'provider': 'openai',
    'api_key': get_api_key('deepseek'),
    'api_base_url': get_base_url('deepseek'),
}]

print("="*60)
print("DRY RUN: static control with patched init, 4 agents, 1 gen, 2 rounds, full obs")
print("="*60)
from experiments.main import run_static_control
result = run_static_control(models, ['full'], ns)
print()
print("="*60)
print("DRY RUN COMPLETE")
print("="*60)
print(f"Final cooperation rate: {result['trials'][0].get('cooperation_rate_mean', 'N/A')}")
