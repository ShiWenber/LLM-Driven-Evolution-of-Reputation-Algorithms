"""Unit tests for the fractional fitness window and per-action fitness."""
from __future__ import annotations

from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.evolution_architecture import (
    ReputationPrisonersDilemmaScenario,
)
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.game import DonorGame, resolve_fitness_window
from experiments.v2_quantitative.population import FALLBACK_STRATEGIES


class _Stub:
    """Minimal always-cooperate agent surface for engine-level tests."""

    def __init__(self, agent_id, cooperates=True):
        self.agent_id = agent_id
        self.cooperates = cooperates

    def choose(self, other, round_num=None):
        return self.cooperates

    def record_donation(self, *args, **kwargs):
        pass


def _play(game: DonorGame, interactions: int) -> None:
    game.payoffs = [0.0] * game.population_size
    game._global_log = []
    game._interaction_deltas = []
    for _ in range(interactions):
        game.play_interaction()


def test_disabled_window_counts_every_interaction():
    for fraction in (None, 0.0, -0.5, 1.0, 2.0):
        assert resolve_fitness_window(fraction, 1000, 8) is None


def test_default_share_is_twenty_percent():
    # 20% of 1250 rounds = 250 whole rounds = 2000 joint actions.
    assert resolve_fitness_window(0.2, 10000, 8) == 2000
    # 20% of 125 rounds = 25 rounds = 200 joint actions.
    assert resolve_fitness_window(0.2, 1000, 8) == 200


def test_window_is_floored_to_whole_rounds():
    # 15% of 125 rounds = 18.75 -> 18 whole rounds = 144 joint actions.
    window = resolve_fitness_window(0.15, 1000, 8)
    assert window == 144
    assert window % 8 == 0


def test_binary_float_error_does_not_lose_a_round():
    # 0.7 * 10 evaluates to 6.999999... in binary floating point.
    assert resolve_fitness_window(0.7, 10 * 4, 4) == 7 * 4


def test_tiny_share_keeps_at_least_one_round():
    assert resolve_fitness_window(0.001, 1000, 8) == 8


def test_share_covering_whole_generation_counts_everything():
    # A one-round generation cannot be split into a smaller window.
    assert resolve_fitness_window(0.2, 1, 1) is None


def test_degenerate_inputs_are_safe():
    assert resolve_fitness_window(0.2, 0, 8) is None
    assert resolve_fitness_window(0.2, -5, 8) is None
    assert resolve_fitness_window("not-a-number", 1000, 8) is None
    # A non-positive pairs-per-round is coerced to 1 rather than exploding.
    assert resolve_fitness_window(0.2, 1000, 0) == 200


def test_windowed_fitness_matches_hand_computation():
    population_size, steps, fraction = 8, 20, 0.25
    game = DonorGame(
        population_size=population_size,
        benefit=2.0,
        cost=1.0,
        fitness_window_fraction=fraction,
        seed=0,
    )
    game.setup_population([_Stub(i) for i in range(population_size)])
    _play(game, steps)

    # One pair per interaction, so the window is a plain interaction count:
    # 25% of 20 interactions = 5.
    window = resolve_fitness_window(fraction, len(game._global_log), 1)
    assert window == 5

    # Every counted action has payoff benefit - cost, regardless of how often
    # the agent was drawn. Undrawn agents have no denominator and score 0.
    counts = [0] * population_size
    for interaction in game._global_log[-window:]:
        for role in ("donor", "recipient"):
            counts[interaction[role]] += 1
    expected = [1.0 if count else 0.0 for count in counts]
    assert game.get_windowed_fitness() == expected

    # Draws are independent, so the window does NOT touch every agent equally;
    # that is the defining difference from a perfect-matching schedule.
    assert window < len(game._global_log)
    assert sum(counts) == 2 * window
    assert len(set(counts)) > 1


def test_disabled_window_divides_total_payoff_by_total_actions():
    game = DonorGame(
        population_size=4,
        benefit=2.0,
        cost=1.0,
        fitness_window_fraction=None,
        seed=1,
    )
    game.setup_population([_Stub(i) for i in range(4)])
    _play(game, 5)
    counts = [0] * 4
    for interaction in game._global_log:
        counts[interaction["donor"]] += 1
        counts[interaction["recipient"]] += 1
    assert game.get_windowed_fitness() == [
        game.payoffs[i] / counts[i] if counts[i] else 0.0 for i in range(4)
    ]


def test_window_uses_matching_payoff_and_action_count_slices():
    game = DonorGame(
        population_size=4, benefit=3.0, cost=1.0,
        fitness_window_fraction=0.5, seed=3,
    )
    game.setup_population([_Stub(i, cooperates=(i != 0)) for i in range(4)])
    _play(game, 20)
    window = resolve_fitness_window(0.5, 20, 1)
    totals = [0.0] * 4
    counts = [0] * 4
    for delta, interaction in zip(
        game._interaction_deltas[-window:], game._global_log[-window:]
    ):
        for pos, payoff in enumerate(delta):
            totals[pos] += payoff
        counts[interaction["donor"]] += 1
        counts[interaction["recipient"]] += 1
    assert game.get_windowed_fitness() == [
        totals[i] / counts[i] if counts[i] else 0.0 for i in range(4)
    ]
    assert any(
        total != 0.0 and total != fitness
        for total, fitness in zip(totals, game.get_windowed_fitness())
    )


def test_no_counted_actions_has_zero_fitness():
    game = DonorGame(population_size=3, fitness_window_fraction=0.5, seed=0)
    game.setup_population([_Stub(i) for i in range(3)])
    assert game.get_windowed_fitness() == [0.0, 0.0, 0.0]


def test_scenario_passes_average_payoff_to_evolution():
    source = FALLBACK_STRATEGIES[0]
    agents = [
        QuantitativeAgent(i, source, executor=V2StrategyExecutor(source))
        for i in range(8)
    ]
    scenario = ReputationPrisonersDilemmaScenario(
        population_size=8, benefit=3.0, cost=1.0,
        observability="full", observability_p=1.0,
        fitness_window_fraction=0.2, num_rounds_per_gen=100,
    )
    result = scenario.evaluate(agents, generation_seed=0, num_rounds=100)
    assert result.n_interactions == 100
    assert result.payoffs == (2.0,) * 8
    assert "divided by the agent's actual number of actions" in (
        scenario.overall_rules_prompt(
            num_generations=2, initial_reputation=0.0
        )
    )
