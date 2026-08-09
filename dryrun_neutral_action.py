"""Dry run #1: verify the action label is now 'option_0' / 'option_1' and
the chain donor -> interaction dict -> my_history -> observation works.
Hand-coded strategy: simple IS-like + always cooperate.
"""
import os
import sys
import json
import argparse
from pathlib import Path

REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv(REPO / '.env')

SIMPLE_STRATEGY = '''
def evaluate(current_reputation, observation, my_history, round_num):
    # Inspect action label
    print(f"[eval] action label = {observation.get('action')!r}")
    if observation.get("action") == "A":
        return min(1.0, current_reputation + 0.2)
    elif observation.get("action") == "B":
        return max(-1.0, current_reputation - 0.2)
    else:
        print(f"  !!! UNKNOWN action label: {observation.get('action')!r}")
        return current_reputation


def decide(recipient_reputation, round_num, my_history):
    # Print my_history actions to verify they're also A/B
    if my_history and len(my_history) <= 2:
        print(f"[decide] my_history actions so far: {[h.get('action') for h in my_history]}")
    return recipient_reputation >= 0.0
'''

import experiments.evolution.population as pop_mod
def patched_init(self):
    from experiments.agents.code_agent import CodeAgent
    agents = [CodeAgent(agent_id=i, code=SIMPLE_STRATEGY) for i in range(self.population_size)]
    for a in agents:
        a.generation = 0
    self.agents = agents
    self.generation = 0
    return agents

pop_mod.EvolutionaryPopulation.initialize_population = patched_init

from experiments.config.load_env import get_api_key, get_base_url
from experiments.main import run_static_control

models = [{
    'name': 'deepseek-v4-flash',
    'provider': 'openai',
    'api_key': get_api_key('deepseek'),
    'api_base_url': get_base_url('deepseek'),
}]

ns = argparse.Namespace(
    population=4, generations=1, rounds=2, seeds=1,
    output=str(REPO / 'results' / 'dryrun_neutral_action'),
    observability='full',
    elitism=2, tournament=2, benefit=2.0, cost=1.0,
    eliminations=1, models='deepseek-v4-flash', run='static', p_values='1.0',
)

print("="*60)
print("DRY RUN #1: 4 agents, 1 gen, 2 rounds, full obs")
print("="*60)
result = run_static_control(models, ['full'], ns)
print()
print("="*60)
print("DRY RUN #1 COMPLETE")
print("="*60)
print(f"Final mean cooperation rate: {result['trials'][0].get('cooperation_rate_mean', 'N/A')}")
