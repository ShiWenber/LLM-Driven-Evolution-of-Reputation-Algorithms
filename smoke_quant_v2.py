"""Quick smoke test of the new 14-baseline set at G=15.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(r"C:\Users\shiwenbo\.mavis\agents\mavis\workspace\llm-reputation-paper\llm-reputation")))
from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.baselines import BASELINES

for name in BASELINES:
    t0 = time.time()
    pop = V2EvolutionaryPopulation(
        population_size=15,
        num_rounds_per_gen=30,
        observability="full",
        seed=0,
        use_baseline=name,
    )
    res = pop.run_evolution(num_generations=15)
    elapsed = time.time() - t0
    traj = res["trajectory"]
    coop_curve = [g["cooperation_rate_mean"] for g in traj]
    print(f"{name:<20} final coop g14={coop_curve[-1]:.3f}, "
          f"mean coop={sum(coop_curve)/len(coop_curve):.3f}, "
          f"({elapsed:.1f}s)")
