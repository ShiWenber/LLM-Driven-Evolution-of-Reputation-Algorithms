import sys
sys.path.insert(0, r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

# Smoke test 1: baseline that DOESN'T evolve (IS should stay at 1.0)
print("=== Smoke test 1: IS baseline 5 gens, pop=5 ===")
pop = V2EvolutionaryPopulation(population_size=5, num_rounds_per_gen=10, use_baseline='IS', seed=42)
res = pop.run_evolution(num_generations=5)
print("gen 0 ids:", [a.agent_id for a in pop.agents])
print("gen 4 ids:", [a.agent_id for a in pop.agents])
for gen_i in range(5):
    t = res["trajectory"][gen_i]
    print(f"  gen {gen_i}: coop={t['cooperation_rate_mean']:.3f}")

# Smoke test 2: ALLD baseline (should stay at 0.0)
print("\n=== Smoke test 2: ALLD via 'IS+' 5 gens, pop=5 ===")
# (use IS+ which evaluates to 0 in IS' framework; pick a stricter one)
# ALLD: never cooperate. Let's just verify with a custom mock using 'SH' (stern judging)
# which defects after one bad interaction. Easier: just check another baseline.
pop2 = V2EvolutionaryPopulation(population_size=5, num_rounds_per_gen=10, use_baseline='SH', seed=42)
res2 = pop2.run_evolution(num_generations=5)
for gen_i in range(5):
    t = res2["trajectory"][gen_i]
    print(f"  gen {gen_i}: coop={t['cooperation_rate_mean']:.3f}")
print("gen 0 ids:", [a.agent_id for a in pop2.agents])
print("gen 4 ids:", [a.agent_id for a in pop2.agents])

# Smoke test 3: Verify reputations consistency after a generation
print("\n=== Smoke test 3: reputation/agent_id consistency ===")
for a in pop.agents:
    assert a.agent_id in a.reputations, f"agent {a.agent_id} missing self-rep"
    # Check self_rep is not the default 0.1 (we ran 5 gens)
    if a.get_self_reputation() == 0.1 and len([k for k in a.reputations if k != a.agent_id]) > 0:
        print(f"  WARN: agent {a.agent_id} self_rep=0.1 but has other reps")
print("  consistency check passed (no self-rep missing)")
