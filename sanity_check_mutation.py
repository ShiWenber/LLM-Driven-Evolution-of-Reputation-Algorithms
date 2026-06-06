"""Verify the model name exists. Try a real mutation-style prompt."""
import os
import urllib.request
import urllib.error
import json
from pathlib import Path

env = {}
for line in Path('.env').read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    env[k.strip()] = v.strip()

key = env.get('DEEPSEEK_API_KEY', '')
base = env.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
model = env.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')

# Test with a longer prompt to ensure the model gives us a code-shaped reply.
print(f"=== testing model '{model}' with mutation-style prompt ===")
prompt = """You are improving agent strategies for a repeated economic game.

GAME RULES:
- 15 agents interact over many rounds.
- Each round, every agent is paired with a random recipient.
- The agent can DONATE (pay 1, recipient gains 2) or NOT DONATE (pay 0, recipient gains 0).

Below is a strategy pair that performed well (score: 12.0).
It has TWO functions: evaluate (updates reputation from observations) and decide (makes donation decisions).

ORIGINAL CODE:
```python
def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return recipient_reputation >= 0.0
```

Your task: Create a VARIANT of this strategy pair.
The variant MUST contain both "evaluate" and "decide" functions.

MODIFICATION GUIDELINES:
- Change how evaluate updates reputation
- Change how decide uses reputation
- The variant should be recognizably related to the original but make DIFFERENT choices

Return ONLY the modified Python code (both functions), nothing else.
"""

req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.8
    }).encode(),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
)
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        usage = data.get('usage', {})
        print(f"Reply length: {len(content)} chars")
        print(f"Usage: {usage}")
        print("--- reply ---")
        print(content[:1200])
        print("--- end reply ---")
        if 'def evaluate' in content and 'def decide' in content:
            print("\nOK: reply contains both functions.")
        else:
            print("\nWARN: reply missing one or both functions.")
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode()[:500]}")
except Exception as e:
    print(f"FAILED: {e}")
