"""Check remaining DeepSeek balance after the 36-trial rerun."""
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

req = urllib.request.Request(
    f"{base}/user/balance",
    headers={"Authorization": f"Bearer {key}"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    data = json.loads(r.read())
    print(json.dumps(data, indent=2))
