"""Smoke test: verify mutation path keeps agent_id stable for survivors
and gives new (never-reused) agent_ids to children."""
import sys
sys.path.insert(0, r"C:\Users\shiwenbo\.minimax\agents\mavis\workspace\llm-reputation-paper\llm-reputation")
from experiments.v2_quantitative import population as popmod
from experiments.v2_quantitative.baselines import get_baseline

# Mock LLM-mutation path
SAMPLE_MUTATED_CODE = '''
def evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation):
    return donor_reputation
def decide(my_reputation, opponent_reputation):
    return opponent_reputation >= 0
'''

def fake_call_llm(self, system_msg, user_msg, max_retries=3):
    return SAMPLE_MUTATED_CODE

# Force init to use IS code (bypass LLM init) — same as calling
# _init_population_baseline with code 'IS'.
def fake_init_population_llm(self):
    code = get_baseline("IS")
    for _ in range(self.population_size):
        self.agents.append(self._new_agent(code))

popmod.V2EvolutionaryPopulation._call_llm = fake_call_llm
popmod.V2EvolutionaryPopulation._init_population_llm = fake_init_population_llm

pop = popmod.V2EvolutionaryPopulation(
    population_size=5, num_rounds_per_gen=10,
    elite_count=2, num_eliminate=2, tournament_size=2,
    seed=42, mutation_temperature=0.0,
)

# Trace counter at each stage
import unittest.mock as mock
orig_select = pop._select_and_reproduce
def traced_select():
    print(f"  [before select] counter={pop._next_agent_id}, agents={sorted(a.agent_id for a in pop.agents)}")
    orig_select()
    print(f"  [after select]  counter={pop._next_agent_id}, agents={sorted(a.agent_id for a in pop.agents)}")
pop._select_and_reproduce = traced_select

res = pop.run_evolution(num_generations=3)

print("\n=== Final state ===")
print(f"  counter = {pop._next_agent_id}")
print(f"  agents (sorted) = {sorted(a.agent_id for a in pop.agents)}")
print(f"  unique IDs = {len(set(a.agent_id for a in pop.agents))}")
print(f"  trajectory coops = {[t['cooperation_rate_mean'] for t in res['trajectory']]}")

# ASSERTIONS for our fix:
# 1. Counter is monotonic and matches expectation
expected_counter = 5 + 2 * 2  # 5 init + 2 mutations × 2 children each
assert pop._next_agent_id == expected_counter, \
    f"expected counter={expected_counter}, got {pop._next_agent_id}"

# 2. All current agents have self-rep in their reputations dict
for a in pop.agents:
    assert a.agent_id in a.reputations, \
        f"agent {a.agent_id} missing self-rep in reputations dict"

# 3. Dropped IDs are gone from reputations (i.e. no zombie entries)
all_ever_ids = set(range(expected_counter))
current_ids = {a.agent_id for a in pop.agents}
dropped = all_ever_ids - current_ids
for a in pop.agents:
    for dropped_id in dropped:
        assert dropped_id not in a.reputations, \
            f"agent {a.agent_id} still has rep for dropped id {dropped_id}"

# 4. agent_id in [0, expected_counter)
for a in pop.agents:
    assert 0 <= a.agent_id < expected_counter, \
        f"agent_id {a.agent_id} out of range"

print("\n[OK] All ID-stability checks passed")

# 5. Note: the survivors list can have duplicates (pre-existing bug in
# tournament selection). This is a separate issue; report it.
from collections import Counter
counts = Counter(a.agent_id for a in pop.agents)
dups = {aid: c for aid, c in counts.items() if c > 1}
if dups:
    print(f"\n[WARN] Pre-existing bug: same agent appears multiple times in self.agents")
    print(f"       Duplicate counts: {dups}")
    print(f"       This shrinks effective population size. Should be fixed in a follow-up.")
