import time, sys
sys.path.insert(0, r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\.env')

from experiments.agents.code_agent import CodeAgent

CODE = r'''
def evaluate(current_reputation, observation, my_history, round_num):
    if observation["action"] == "A":
        return min(1.0, current_reputation + 0.2)
    else:
        return max(-1.0, current_reputation - 0.2)

def decide(recipient_reputation, round_num, my_history):
    return recipient_reputation >= 0.0
'''

agent = CodeAgent(agent_id=0, code=CODE)
print(f'agent compiled: {agent._compiled}')

t0 = time.time()
for _ in range(1000):
    agent.observe(
        donor_id=0,
        observation={'donor': 0, 'recipient': 1, 'action': 'A', 'round': 1, 'donor_reputation': 0.5, 'recipient_reputation': 0.5},
        round_num=1
    )
t1 = time.time()
for _ in range(1000):
    agent.decide(recipient_id=1, round_num=1, population_size=15)
t2 = time.time()

per_call_ms = (t2 - t0) / 1000 * 1000
print(f'1000 observe() calls: {(t1-t0)*1000:.0f}ms')
print(f'1000 decide() calls:  {(t2-t1)*1000:.0f}ms')
print(f'per-call total:       {per_call_ms:.2f}ms')
print()
# Estimate per trial
total_calls = 30 * 10 * (15 + 15*15)  # 30 rounds * 10 gens * (15 decide + 225 observe)
game_sec = total_calls * per_call_ms / 1000
print(f'Estimated game time:  {game_sec:.0f} sec = {game_sec/60:.1f} min')
print(f'LLM calls:            1 + 10*5 = 51')
llm_total = 21.6 * 60 - game_sec
print(f'LLM time:             {llm_total:.0f} sec = {llm_total/60:.1f} min')
print(f'Per-LLM-call:         {llm_total/51:.1f} sec')
