import unittest

from experiments.v2_quantitative.population import V2EvolutionaryPopulation


TYPE1_CODE = """
def observe(donor_reputation, donor_action, recipient_reputation, recipient_action, my_reputation):
    return donor_reputation, recipient_reputation
def decide(my_reputation, opponent_reputation):
    return True
"""

TYPE2_CODE = """
class LLMAgent:
    def __init__(self, agent_id): self.agent_id = agent_id
    def decide(self): return True
    def observe(self, donor_id, donor_action, recipient_id, recipient_action): pass
"""


class ImitationLearningPromptTests(unittest.TestCase):
    def capture_prompt(self, agent_type: str, mode: str, fitness: float) -> str:
        population = V2EvolutionaryPopulation.__new__(V2EvolutionaryPopulation)
        population.agent_type = agent_type
        population.imitation_learning_mode = mode
        population._fallback_mutation_count = 0
        captured = []
        child = TYPE2_CODE if agent_type == "agent-type2" else TYPE1_CODE
        population._call_llm = lambda system, user: captured.append(user) or child
        population._make_agent = lambda code, preserve_id: (code, preserve_id)
        population._llm_small_mutate(child, fitness, 7)
        return captured[0]

    def test_real_parent_fitness_reaches_every_prompt(self):
        for agent_type in ("agent-type1", "agent-type2"):
            for mode in ("random", "deliberate"):
                with self.subTest(agent_type=agent_type, mode=mode):
                    self.assertIn("17.250", self.capture_prompt(agent_type, mode, 17.25))

    def test_random_and_deliberate_prompts_have_different_objectives(self):
        random_prompt = self.capture_prompt("agent-type1", "random", 3.0)
        deliberate_prompt = self.capture_prompt("agent-type1", "deliberate", 3.0)
        self.assertNotIn("adjust a single number or threshold", random_prompt)
        self.assertNotIn("higher fitness", random_prompt)
        self.assertIn("higher fitness", deliberate_prompt)


if __name__ == "__main__":
    unittest.main()
