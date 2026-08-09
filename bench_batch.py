import time, sys, os
sys.path.insert(0, r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\.env')

from experiments.evolution.mutation import MutationOperator

P1 = '''def evaluate(current_reputation, observation, my_history, round_num):
    if observation["action"] == "A":
        return min(1.0, current_reputation + 0.2)
    else:
        return max(-1.0, current_reputation - 0.2)

def decide(recipient_reputation, round_num, my_history):
    return recipient_reputation >= 0.0
'''

P2 = '''def evaluate(current_reputation, observation, my_history, round_num):
    if observation["action"] == "A":
        return min(1.0, current_reputation + 0.1)
    else:
        return max(-1.0, current_reputation - 0.4)

def decide(recipient_reputation, round_num, my_history):
    return recipient_reputation > 0.3
'''

PARENTS = [(P1, 25.0), (P2, 30.0)]

mut = MutationOperator(
    llm_provider='openai',
    model='deepseek-v4-flash',
    temperature=0.8,
    max_tokens=2000,
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    api_base_url='https://api.deepseek.com/v1',
)

# Test batch mode
t0 = time.time()
results_batch = mut.mutate_batch(PARENTS, population_size=15)
t1 = time.time()
print(f'Batch mode (2 parents, 1 LLM call): {t1-t0:.1f}s')
for i, r in enumerate(results_batch):
    if r:
        print(f'  Variant {i+1}: {len(r)} chars')
        print(f'    snippet: {r[:120].strip()[:90]}...')
    else:
        print(f'  Variant {i+1}: None')

# Test 5-parent batch (production case)
P5 = [P1, P2, P1, P2, P1]
F5 = [25.0, 30.0, 28.0, 32.0, 22.0]
PARENTS5 = list(zip(P5, F5))
t2 = time.time()
results_5 = mut.mutate_batch(PARENTS5, population_size=15)
t3 = time.time()
print(f'\nBatch mode (5 parents, 1 LLM call): {t3-t2:.1f}s')
for i, r in enumerate(results_5):
    if r:
        print(f'  Variant {i+1}: {len(r)} chars')
    else:
        print(f'  Variant {i+1}: None')
