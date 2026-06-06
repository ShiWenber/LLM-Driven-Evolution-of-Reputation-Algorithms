import json
with open('results/_manifest.json') as f:
    m = json.load(f)

n_api_err = 0
n_fallback = 0
n_llm_used = 0
for e in m:
    tail = e.get('stdout_tail','')
    if 'invalid_api_key' in tail or 'invalid_request_error' in tail:
        n_api_err += 1
    if 'using random mutation fallback' in tail or 'using fallback strategies' in tail:
        n_fallback += 1
    if 'using random mutation fallback' not in tail and 'using fallback strategies' not in tail:
        n_llm_used += 1
print(f"Total trials: {len(m)}")
print(f"API errors: {n_api_err}")
print(f"Fallback used: {n_fallback}")
print(f"LLM actually used: {n_llm_used}")
print()
for e in m:
    tail = e.get('stdout_tail','')
    api_err = 'invalid_api_key' in tail or 'invalid_request_error' in tail
    fallback = 'using random mutation fallback' in tail or 'using fallback strategies' in tail
    print(f"  #{e['i']:2d} {e['run']:18s} {e['obs']:14s} seed={e['seed']}  api_err={api_err}  fallback={fallback}")
