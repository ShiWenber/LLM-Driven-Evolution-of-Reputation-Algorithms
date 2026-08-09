import time
import sys
sys.path.insert(0, r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\.env')

from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url='https://api.deepseek.com/v1'
)

prompt = """You are designing agent strategies for a repeated economic game.

GAME RULES:
- 15 agents interact over 30 rounds.
- Each round, every agent is paired with a random recipient.
- Each round, the agent must choose one of TWO options: A or B. Exactly one is cooperative.

Each agent runs TWO functions: evaluate() and decide().

Write 5 DIFFERENT strategy pairs using different approaches. Some should be simple, some complex.
Each pair MUST contain exactly TWO functions named 'evaluate' and 'decide'.

Evaluate: takes (current_reputation, observation, my_history, round_num), returns float in [-1, 1].
Decide: takes (recipient_reputation, round_num, my_history), returns bool.

observation['action'] is 'A' or 'B'.
my_history: list of past interactions.

Separate each pair with '# ---' on its own line.
"""

# Test with max_tokens=8000 (current)
t0 = time.time()
response = client.chat.completions.create(
    model='deepseek-v4-flash',
    messages=[
        {'role': 'system', 'content': 'You are a Python programmer. Generate 5 strategy code pairs.'},
        {'role': 'user', 'content': prompt},
    ],
    temperature=0.9,
    max_tokens=8000
)
t1 = time.time()
out = response.choices[0].message.content
print(f'== Test 1: max_tokens=8000 ==')
print(f'Time: {t1-t0:.1f}s')
print(f'Output: {len(out)} chars')
print(f'Output tokens: {response.usage.completion_tokens}')
usage_details = response.usage.completion_tokens_details
if usage_details:
    print(f'Reasoning tokens: {usage_details.reasoning_tokens}')
print(f'Total tokens: {response.usage.total_tokens}')

# Test with max_tokens=2000 (smaller, faster)
t2 = time.time()
response2 = client.chat.completions.create(
    model='deepseek-v4-flash',
    messages=[
        {'role': 'system', 'content': 'You are a Python programmer. Generate 5 strategy code pairs.'},
        {'role': 'user', 'content': prompt},
    ],
    temperature=0.9,
    max_tokens=2000
)
t3 = time.time()
out2 = response2.choices[0].message.content
print(f'\n== Test 2: max_tokens=2000 ==')
print(f'Time: {t3-t2:.1f}s')
print(f'Output: {len(out2)} chars')
print(f'Output tokens: {response2.usage.completion_tokens}')
usage_details2 = response2.usage.completion_tokens_details
if usage_details2:
    print(f'Reasoning tokens: {usage_details2.reasoning_tokens}')

# Test with max_tokens=500
t4 = time.time()
response3 = client.chat.completions.create(
    model='deepseek-v4-flash',
    messages=[
        {'role': 'system', 'content': 'You are a Python programmer. Generate 5 strategy code pairs.'},
        {'role': 'user', 'content': prompt},
    ],
    temperature=0.9,
    max_tokens=500
)
t5 = time.time()
out3 = response3.choices[0].message.content
print(f'\n== Test 3: max_tokens=500 ==')
print(f'Time: {t5-t4:.1f}s')
print(f'Output: {len(out3)} chars')
print(f'Output tokens: {response3.usage.completion_tokens}')
usage_details3 = response3.usage.completion_tokens_details
if usage_details3:
    print(f'Reasoning tokens: {usage_details3.reasoning_tokens}')
