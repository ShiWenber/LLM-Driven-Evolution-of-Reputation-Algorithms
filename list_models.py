"""List which models are actually available on this deepseek account."""
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

# Try to list models (deepseek's API may or may not support /models)
for endpoint in ('/models', '/v1/models'):
    url = f"{base}{endpoint}"
    print(f"=== GET {url} ===")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
            print(json.dumps(data, indent=2)[:2000])
        break
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"HTTPError {e.code}: {body}")
    except Exception as e:
        print(f"FAILED: {e}")
    print()
