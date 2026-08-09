"""Test one full trial with the new batch mutation mode."""
import time, sys, argparse
import json
sys.path.insert(0, r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\.env')

from pathlib import Path
REPO = Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from experiments.config.load_env import get_api_key, get_base_url
from experiments.main import run_evolutionary

models = [{
    'name': 'deepseek-v4-flash',
    'provider': 'openai',
    'api_key': get_api_key('deepseek'),
    'api_base_url': get_base_url('deepseek'),
}]

ns = argparse.Namespace(
    population=15, generations=10, rounds=30, seeds=1,
    output=str(REPO / 'results' / 'exp6_batch_test' / 'partial_0.3_seed0'),
    observability='partial_0.3', elitism=3, tournament=3, benefit=2.0, cost=1.0,
    eliminations=5, models='deepseek-v4-flash', run='evolutionary', p_values='0.3',
    eliminate=5, mutation_temperature=0.9,
)

Path(ns.output).mkdir(parents=True, exist_ok=True)
t0 = time.time()
result = run_evolutionary(models, ['partial_0.3'], ns)
elapsed = time.time() - t0
print(f'\nTotal: {elapsed:.0f}s = {elapsed/60:.1f} min')
trial = result['trials_summary'][0]
final_coop = trial['trajectory'][-1]['cooperation_rate_mean']
print(f'Final coop: {final_coop:.3f}')
