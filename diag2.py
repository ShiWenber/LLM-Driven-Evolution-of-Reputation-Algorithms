import json
d = json.load(open('results/exp1_method/private_seed0/evo_private_deepseek-v4-flash_20260606_134009.json'))
print('top-level keys:', list(d.keys()))
for k in d.keys():
    v = d[k]
    if isinstance(v, list):
        print(f'  {k}: list len={len(v)}')
        if v and isinstance(v[0], dict):
            print(f'    first element keys: {list(v[0].keys())[:8]}')
    else:
        print(f'  {k}: type={type(v).__name__}, val={str(v)[:80]}')
