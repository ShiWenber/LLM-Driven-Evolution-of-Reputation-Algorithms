"""Direct API ping: 30s timeout to see if Intern responds at all."""
import os, time
from openai import OpenAI

t0 = time.time()
client = OpenAI(
    api_key=os.environ.get('ROUTER_API_KEY', ''),
    base_url='https://llmapi.paratera.com',
)
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