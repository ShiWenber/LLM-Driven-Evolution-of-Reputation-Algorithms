"""Matching and observation boundaries of the restored historical protocol."""
import random

from experiments.v2_quantitative.game import DonorGame
from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from tests.test_async_protocol import _CoinAgent


def make_game(n=16):
    game = DonorGame(n, seed=7)
    agents = [_CoinAgent(100 + i) for i in range(n)]
    game.setup_population(agents)
    return game, agents


def test_matching_is_historical_shuffle_and_each_agent_acts_once():
    game, agents = make_game()
    ids = [a.agent_id for a in agents]
    rng = random.Random(7)
    for _ in range(3):
        expected = ids.copy()
        rng.shuffle(expected)
        interactions = game.play_step()['interactions']
        actual = [(i['donor'], i['recipient']) for i in interactions]
        assert actual == list(zip(expected[::2], expected[1::2]))
        assert sorted(x for pair in actual for x in pair) == ids


def test_entire_round_decides_before_observations_and_next_round_sees_updates():
    game, agents = make_game()
    first = game.play_round()['interactions']
    assert [a.decide_reads for a in agents] == [[0.0]] * 16
    game.distribute_observations_and_self_judgments(first)
    game.play_round()
    assert all(a.decide_reads == [0.0, -1.0] for a in agents)


def test_odd_matching_has_one_sitout():
    game, agents = make_game(5)
    assert len(game.play_step()['interactions']) == 2
    assert sorted(a.total_decisions for a in agents) == [0, 1, 1, 1, 1]


def test_default_population_preserves_10000_pair_interactions():
    pop = V2EvolutionaryPopulation(population_size=16, target_interactions_per_gen=10000)
    assert pop.num_rounds_per_gen == 1250
    assert pop._result_config(100)['observation_schedule'] == 'synchronous'
    prompt = pop._overall_game_rules_prompt()
    assert '1250 rounds (10000 joint interactions)' in prompt
    assert 'final 2000' in prompt
    assert 'randomly partitioned' in prompt
    assert 'delivered immediately' not in prompt


def test_generation_and_window_are_balanced():
    """16 synchronous rounds: 16 x 8 = 128 interactions, each agent acting once
    per round, so exposure is equal across the population."""
    game, agents = make_game()
    for _ in range(16):
        game.distribute_observations_and_self_judgments(
            game.play_step()['interactions']
        )
    assert len(game._global_log) == 128
    assert [a.total_decisions for a in agents] == [16] * 16
    assert game.get_windowed_fitness() == [2.0] * 16

