"""Fermi imitation-learning prompt tests (pytest).

Verifies that the real parent fitness reaches every mutation prompt
and that the "random" vs "deliberate" imitation modes produce
qualitatively different objectives.
"""
import pytest

from experiments.v2_quantitative.population import V2EvolutionaryPopulation


TYPE1_CODE = """
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    return A_rep
def decide(my_reputation, opponent_reputation):
    return True
"""

TYPE2_CODE = """
class LLMAgent:
    def __init__(self, agent_id): self.agent_id = agent_id
    def decide(self): return True
    def observe(self, A_id, A_action, B_id, B_action): pass
"""


def _capture_prompt(agent_type: str, mode: str, fitness: float) -> str:
    """Capture the user prompt that _llm_small_mutate would send."""
    population = V2EvolutionaryPopulation.__new__(V2EvolutionaryPopulation)
    population.agent_type = agent_type
    population.imitation_learning_mode = mode
    population._fallback_mutation_count = 0
    captured = []
    child = TYPE2_CODE if agent_type == "agent-type2" else TYPE1_CODE
    population._call_llm = (
        lambda system, user, **_kwargs: captured.append(user) or child
    )
    population._validate_code = lambda code: None
    population._make_agent = lambda code, preserve_id: (code, preserve_id)
    population._llm_small_mutate(child, fitness, 7)
    return captured[0]


@pytest.mark.parametrize("agent_type", ["agent-type1", "agent-type2"])
@pytest.mark.parametrize("mode", ["random", "deliberate"])
def test_real_parent_fitness_reaches_every_prompt(agent_type, mode):
    assert "17.250" in _capture_prompt(agent_type, mode, 17.25)


def test_random_and_deliberate_prompts_have_different_objectives():
    random_prompt = _capture_prompt("agent-type1", "random", 3.0)
    deliberate_prompt = _capture_prompt("agent-type1", "deliberate", 3.0)
    assert "adjust a single number or threshold" not in random_prompt
    assert "higher fitness" not in random_prompt
    assert "higher fitness" in deliberate_prompt
