"""Fermi imitation-learning prompt tests (pytest).

Verifies that the real parent fitness reaches every mutation prompt
and that the "random" vs "deliberate" imitation modes produce
qualitatively different objectives.
"""
import pytest

from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative import prompts


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
    population.population_size = 16
    population.num_rounds_per_gen = 125
    population.num_generations = 30
    population.benefit = 2.0
    population.cost = 1.0
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


@pytest.mark.parametrize("mode", ["random", "deliberate"])
def test_type1_fermi_prompts_include_full_rules_without_cold_start_constraint(mode):
    prompt = _capture_prompt("agent-type1", mode, 3.0)
    assert "OVERALL GAME RULES" in prompt
    assert "drawn uniformly at random" in prompt
    assert "(C, C): 1 each" in prompt
    assert "Every third-party agent observes" in prompt
    assert "cold-start" not in prompt.lower()
    assert "initial reputation" not in prompt.lower()


def test_type1_init_and_ordinary_mutation_requests_share_overall_rules():
    population = V2EvolutionaryPopulation(
        population_size=4,
        num_rounds_per_gen=8,
        num_generations=2,
        agent_type="agent-type1",
    )
    captured = []
    population._call_llm = (
        lambda _system, prompt, **_kwargs: captured.append(prompt) or TYPE1_CODE
    )
    population._validate_code = lambda _code: None
    population._request_valid_code(population._init_prompt(), "init test")
    population._mutate(TYPE1_CODE, 1.0)

    assert len(captured) == 2
    # Derive the expected payoff row from the configured benefit/cost instead
    # of hardcoding it, so this test keeps checking "the rules block reflects
    # the configured payoffs" when the defaults change.
    cc_payoff = population.benefit - population.cost
    expected_cc = f"(C, C): {cc_payoff:g} each"
    expected_cd = f"defector +{population.benefit:g}"
    for prompt in captured:
        assert "OVERALL GAME RULES" in prompt
        assert "drawn uniformly at random" in prompt
        assert expected_cc in prompt
        assert expected_cd in prompt
        assert "cold-start" not in prompt.lower()
        assert "initial reputation" not in prompt.lower()


def test_task_templates_do_not_repeat_authoritative_game_rules():
    task_templates = (
        prompts.INIT_PROMPT_V2,
        prompts.MUTATION_PROMPT_V2,
        prompts.SMUTATION_PROMPT_V2,
        prompts.DELIBERATE_MUTATION_PROMPT_V2,
        prompts.INIT_PROMPT_V3,
        prompts.MUTATION_PROMPT_V3,
        prompts.SMALL_MUTATION_PROMPT_V3,
        prompts.DELIBERATE_MUTATION_PROMPT_V3,
    )
    repeated_rule_phrases = (
        "drawn uniformly at random",
        "rounds per generation",
        "pair payoffs",
        "third-party agent observes",
    )

    for template in task_templates:
        assert all(phrase not in template for phrase in repeated_rule_phrases)
