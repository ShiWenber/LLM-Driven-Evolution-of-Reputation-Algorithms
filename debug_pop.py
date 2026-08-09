"""Run a tiny population (3 agents, 2 rounds, 2 gens) and trace."""
import sys
sys.path.insert(0, ".")
from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.baselines import get_baseline

p = V2EvolutionaryPopulation(
    population_size=3, num_rounds_per_gen=2, observability="full",
    seed=42, use_baseline="IS",
)

# Manually run 2 gens with detailed tracing
print("Initialize 3 agents with IS")
p._init_population_baseline()
print(f"After init:  self_reps = {[a.self_reputation for a in p.agents]}")
print(f"After init:  ratings[0] = {[a.reputations for a in p.agents]}")

from experiments.v2_quantitative.game import V2DonorGame
game = V2DonorGame(population_size=3, benefit=2.0, cost=1.0, observability="full", observability_p=1.0, seed=42)
game.setup_population(p.agents)

for gen in range(2):
    print(f"\n=== Gen {gen} ===")
    for r in range(2):
        game.round_num = 0
        game.payoffs = [0.0]*3
        game._global_log = []
        print(f"\n  Round {r+1} starts. State: self_reps={[a.self_reputation for a in p.agents]}")
        game.play_round()
        print(f"  After play_round: coop in this round = {sum(1 for i in game._global_log if i['donor_action'])}/3")
        game.distribute_observations_and_self_judgments()
        print(f"  After distribution: self_reps={[a.self_reputation for a in p.agents]}")
        print(f"  ratings[0] of agent 0: {p.agents[0].reputations}")
        print(f"  Agent 0 next decide: my={p.agents[0].self_reputation}, opp1={p.agents[0].get_reputation(1)} -> {p.agents[0]._executor.decide(p.agents[0].self_reputation, p.agents[0].get_reputation(1))}")
