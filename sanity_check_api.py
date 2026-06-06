"""Sanity check: does the new DeepSeek key actually work?"""
import os
import urllib.request
import urllib.error
import json
from pathlib import Path

# load .env
env = {}
for line in Path('.env').read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    env[k.strip()] = v.strip()
os.environ.update(env)

key = env.get('DEEPSEEK_API_KEY', '')
base = env.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
model = env.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')

print(f"Key prefix: {key[:6]} ... suffix: ...{key[-4:]}")
print(f"Base: {base}")
print(f"Model: {model}")
print()

# 1. balance
print("=== /user/balance ===")
req = urllib.request.Request(
    f"{base}/user/balance",
    headers={"Authorization": f"Bearer {key}"}
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
        print(json.dumps(data, indent=2)[:600])
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode()[:300]}")
    raise SystemExit(1)
except Exception as e:
    print(f"FAILED: {e}")
    raise SystemExit(1)

print()

# 2. tiny chat
print("=== /chat/completions (tiny ping) ===")
req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 10,
        "temperature": 0.0
    }).encode(),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
        reply = data.get('choices', [{}])[0].get('message', {}).get('content', '<none>')
        usage = data.get('usage', {})
        print(f"Reply: {reply!r}")
        print(f"Usage: {usage}")
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode()[:300]}")
    raise SystemExit(1)
except Exception as e:
    print(f"FAILED: {e}")
    raise SystemExit(1)

print()
print("ALL GREEN. API key is alive.")
