"""Debug IS baseline with detailed tracing."""
import sys
sys.path.insert(0, ".")
from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.baselines import get_baseline
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.agent import QuantitativeAgent

# Set up one IS agent
code = get_baseline("IS")
ex = V2StrategyExecutor(code)
a = QuantitativeAgent(0, code, ex)
print("Initial state:")
print(f"  self_rep = {a.self_reputation}")
print(f"  ratings (empty) = {a.reputations}")

# Gen 0 round 0: agent 0 donates to agent 1, both cooperate
print("\nGen 0 round 0: agent 0 as donor, agent 1 as recipient")
print(f"  before: a.self_rep = {a.self_reputation}, a.ratings[1] = {a.get_reputation(1)}")
a.self_judge(donor_action='cooperate', recipient_id=1, recipient_action='cooperate')
print(f"  after self_judge: a.self_rep = {a.self_reputation}")
# Now a observes another agent (e.g., agent 2) as donor in (2→3) cooperation
a.observe_and_judge(donor_id=2, donor_action='cooperate', recipient_id=3, recipient_action='cooperate')
print(f"  after observing 2->3 C: a.ratings[2] = {a.get_reputation(2)}")
# More observations
for i in range(3, 15):
    a.observe_and_judge(donor_id=i, donor_action='cooperate', recipient_id=(i+1) % 15, recipient_action='cooperate')
print(f"  after 14 observations: a.ratings[2]={a.get_reputation(2):.3f}, a.ratings[5]={a.get_reputation(5):.3f}, a.ratings[14]={a.get_reputation(14):.3f}")

# Now round 1: agent 0 is donor with random recipient
print("\nGen 0 round 1: agent 0 as donor, agent 7 as recipient")
print(f"  before: a.self_rep = {a.self_reputation}, a.ratings[7] = {a.get_reputation(7)}")
print(f"  decide(my_rep={a.self_reputation}, opp_rep={a.get_reputation(7)}) = ", end="")
action = a.choose(opponent_id=7, round_num=1)
print(action)
