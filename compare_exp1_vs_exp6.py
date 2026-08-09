import json
d_orig = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp1_method_n10/full_seed0'
d_new = 'C:/Users/shiwenbo/.mavis/agents/mavis/workspace/llm-reputation-paper/llm-reputation/results/exp6_leading_eight'

with open(d_orig + '/evo_full_deepseek-v4-flash_20260610_052333.json') as f:
    orig = json.load(f)
with open(d_new + '/evo_full_deepseek-v4-flash_20260616_105442.json') as f:
    new = json.load(f)

orig_coops = [g['cooperation_rate_mean'] for g in orig['trajectory']]
new_coops = [g['cooperation_rate_mean'] for g in new['trajectory']]

print('=== ORIGINAL Exp 1 full (seed 0) trajectory ===')
for i, c in enumerate(orig_coops):
    print(f'  gen {i}: {c:.3f}')
print(f'gen-0: {orig_coops[0]:.3f}, gen-10: {orig_coops[-1]:.3f}')

print()
print('=== NEW Exp 6 (leading-8 interface) full (seed 0) trajectory ===')
for i, c in enumerate(new_coops):
    print(f'  gen {i}: {c:.3f}')
print(f'gen-0: {new_coops[0]:.3f}, gen-10: {new_coops[-1]:.3f}')

print()
print('=== Comparison ===')
print(f'                Original   New (leading-8)')
for i in range(11):
    o = orig_coops[i] if i < len(orig_coops) else None
    n = new_coops[i] if i < len(new_coops) else None
    print(f'  gen {i:2d}:        {o:.3f}       {n:.3f}' if o is not None and n is not None else '')
