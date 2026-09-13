"""Unit tests for the fractional fitness window (window share / burn-in).

The window used to be an absolute interaction count; it is now a share of the
generation that is floored to a whole number of rounds, so every agent's
counted-interaction count stays identical under the engine's perfect matching.
"""
from __future__ import annotations

from experiments.v2_quantitative.game import DonorGame, resolve_fitness_window


class _Stub:
    """Minimal always-cooperate agent surface for engine-level tests."""

    def __init__(self, agent_id):
        self.agent_id = agent_id

    def choose(self, other, round_num=None):
        return True

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

    # Every counted interaction nets its two always-cooperating players
    # benefit - cost, so an agent's windowed fitness is just how often it was
    # drawn inside the window.
    expected = [0.0] * population_size
    for interaction in game._global_log[-window:]:
        for role in ("donor", "recipient"):
            expected[interaction[role]] += 2.0 - 1.0
    assert game.get_windowed_fitness() == expected

    # Draws are independent, so the window does NOT touch every agent equally;
    # that is the defining difference from a perfect-matching schedule.
    assert window < len(game._global_log)
    counts = [0] * population_size
    for interaction in game._global_log[-window:]:
        counts[interaction["donor"]] += 1
        counts[interaction["recipient"]] += 1
    assert sum(counts) == 2 * window


def test_windowed_fitness_equals_payoffs_when_disabled():
    game = DonorGame(
        population_size=4,
        benefit=2.0,
        cost=1.0,
        fitness_window_fraction=None,
        seed=1,
    )
    game.setup_population([_Stub(i) for i in range(4)])
    _play(game, 5)
    assert game.get_windowed_fitness() == list(game.payoffs)
