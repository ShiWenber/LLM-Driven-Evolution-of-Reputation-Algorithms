import json, os
print('=== Full obs: A/B label ===')
for seed in range(3):
    d = f'results/exp6_sweep_AB/full_seed{seed}'
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
    final = coops[-1]
    status = 'OK' if final > 0.5 else 'FAIL'
    print(f'  seed{seed}: gen-10 = {final:.3f} ({status})')
print()
print('=== Full obs: donate label (original) ===')
for seed in range(3):
    d = f'results/exp1_method_n10/full_seed{seed}'
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
    final = coops[-1]
    status = 'OK' if final > 0.5 else 'FAIL'
    print(f'  seed{seed}: gen-10 = {final:.3f} ({status})')

# Also for partial 0.7
print()
print('=== Partial 0.7: A/B label ===')
for seed in range(3):
    d = f'results/exp6_sweep_AB/partial_0.7_seed{seed}'
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
    final = coops[-1]
    status = 'OK' if final > 0.5 else 'FAIL'
    print(f'  seed{seed}: gen-10 = {final:.3f} ({status})')

print()
print('=== Partial 0.7: donate label (original) ===')
for seed in range(3):
    d = f'results/exp1_method_n10/partial_0.7_seed{seed}'
    files = [f for f in os.listdir(d) if f.startswith('evo_') and f.endswith('.json')]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    with open(os.path.join(d, files[0])) as f:
        t = json.load(f)
    coops = [g['cooperation_rate_mean'] for g in t['trajectory']]
    final = coops[-1]
    status = 'OK' if final > 0.5 else 'FAIL'
    print(f'  seed{seed}: gen-10 = {final:.3f} ({status})')
