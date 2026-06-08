"""Check if deepseek-reasoner model is accessible with the current API key."""
import os
import urllib.request
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

for model in ('deepseek-reasoner', 'deepseek-chat', 'deepseek-coder'):
    print(f"=== testing model '{model}' ===")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
            "max_tokens": 10,
            "temperature": 0.0,
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
            reply = data.get('choices', [{}])[0].get('message', {}).get('content', '<none>')
            usage = data.get('usage', {})
            print(f"  OK. reply={reply!r}  usage={usage}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        print(f"  HTTPError {e.code}: {body}")
    except Exception as e:
        print(f"  FAILED: {e}")
    print()
