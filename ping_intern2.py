"""Direct API ping: bypass .env, hard-code key in test (less safe but for diagnostic)."""
import time
from openai import OpenAI

t0 = time.time()
# Pull key from .env file directly to bypass any shell env issues
import pathlib
env_path = pathlib.Path(r'C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation\.env')
key = ''
base = 'https://llmapi.paratera.com'
for line in env_path.read_text().splitlines():
    if line.startswith('ROUTER_API_KEY='):
        key = line.split('=', 1)[1].strip()
    elif line.startswith('ROUTER_API_BASE='):
        base = line.split('=', 1)[1].strip()
print(f'Key loaded: {key[:8]}... (len={len(key)})')
print(f'Base URL: {base}')

client = OpenAI(api_key=key, base_url=base)
try:
    print('Sending simple request to Intern-S2-Preview (max 30s)...')
    resp = client.chat.completions.create(
        model='Intern-S2-Preview',
        messages=[{'role': 'user', 'content': 'Say hi in 3 words.'}],
        max_tokens=20,
        timeout=30,
    )
    print(f'OK in {time.time()-t0:.1f}s')
    print(f'Response: {resp.choices[0].message.content}')
except Exception as e:
    print(f'FAIL in {time.time()-t0:.1f}s: {type(e).__name__}: {e}')