"""Show concrete data from the 3-gen x 1000 inter smoke."""
import json
from pathlib import Path

OUT = Path(r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation\results\quantitative_baseline\LLM_v3_g3_1000inter_smoke_seed0\evolutionary.json")
data = json.loads(OUT.read_text())

print("=" * 78)
print("  3 gen x 1000 inter/gen x seed=0  |  v3 type-2 LLM agent")
print("=" * 78)
print(f"  config: agent_type={data['config']['agent_type']}, "
      f"rounds={data['config']['num_rounds_per_gen']}, "
      f"target={data['config']['target_interactions_per_gen']}, "
      f"thinking={data['config']['llm_thinking']}, "
      f"max_tokens={data['config']['llm_max_tokens']}")
print(f"  FALLBACK: init={data['config']['fallback_init_count']}/15, "
      f"mutation={data['config']['fallback_mutation_count']}/10")
print()
print("  Per-gen trajectory:")
print(f"  {'gen':>3} | {'coop':>6} | {'fitness':>7} | {'n_inter':>7}")
print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")
for t in data['trajectory']:
    print(f"  {t['generation']:>3} | {t['cooperation_rate_mean']:>6.3f} | "
          f"{t['fitness_mean']:>7.2f} | {t['n_interactions']:>7}")
print()

print("  Final 15 agents (cooperation_rate, fitness, decide first-line):")
print(f"  {'aid':>4} | {'coop':>6} | {'fit':>6} | first-line-of-decide")
print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*60}")
for a in data['final_population']:
    coop = a['cooperation_rate']
    fit = a['fitness']
    decide_first = ""
    for line in a['code'].split('\n'):
        s = line.strip()
        if s.startswith('return') and 'self._rng' not in s and s not in ('return True', 'return False'):
            decide_first = s
            break
    if not decide_first:
        for line in a['code'].split('\n'):
            s = line.strip()
            if s.startswith('return'):
                decide_first = s
                break
    print(f"  {a['agent_id']:>4} | {coop:>6.3f} | {fit:>6.2f} | {decide_first[:60]}")
print()

def classify(a):
    code = a['code'].lower()
    has_rng = '_rng' in code or ('random' in code and 'randint' in code)
    has_payoff = 'payoff' in code or 'fitness' in code or 'score' in code
    has_image = 'image' in code or 'reput' in code
    has_memory = 'history' in code or 'memory' in code or 'past' in code or 'last' in code
    has_standing = 'standing' in code or 'judge' in code
    return has_rng, has_payoff, has_image, has_memory, has_standing

print("  Strategy classification (lower-cased code grep):")
families = {'A_image/reputation': 0, 'B_payoff/fitness-tracking': 0, 'C_memory/history': 0, 'D_random': 0, 'E_simple/other': 0}
for a in data['final_population']:
    has_rng, has_payoff, has_image, has_memory, has_standing = classify(a)
    if has_image:
        families['A_image/reputation'] += 1
    elif has_payoff:
        families['B_payoff/fitness-tracking'] += 1
    elif has_memory:
        families['C_memory/history'] += 1
    elif has_rng:
        families['D_random'] += 1
    else:
        families['E_simple/other'] += 1
for f, n in sorted(families.items(), key=lambda x: -x[1]):
    if n > 0:
        print(f"    {f}: {n}")

mean_coop = sum(a['cooperation_rate'] for a in data['final_population']) / len(data['final_population'])
mean_fit = sum(a['fitness'] for a in data['final_population']) / len(data['final_population'])
print()
print(f"  Mean final coop (per-agent): {mean_coop:.3f}")
print(f"  Mean final fitness:          {mean_fit:.1f}")
print()

# Show 1-2 representative class codes
print("=" * 78)
print("  Representative class code samples (gen 2, real LLM-generated):")
print("=" * 78)
for i, a in enumerate(data['final_population'][:3]):
    print(f"\n  --- agent_id={a['agent_id']}, coop={a['cooperation_rate']:.3f}, "
          f"fitness={a['fitness']:.1f} ---")
    print(a['code'])
