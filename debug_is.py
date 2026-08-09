"""Debug IS baseline with 3 rounds 2 gens."""
import sys
sys.path.insert(0, ".")
from experiments.v2_quantitative.population import V2EvolutionaryPopulation

p = V2EvolutionaryPopulation(
    population_size=15, num_rounds_per_gen=3, observability="full",
    seed=0, use_baseline="IS",
)
r = p.run_evolution(num_generations=2)
for g in r["trajectory"]:
    print(g)
print()
print("Final pop self_reputations:", [s["self_reputation"] for s in r["final_population"]])
print("Final pop coop rates:", [s["cooperation_rate"] for s in r["final_population"]])
