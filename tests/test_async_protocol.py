"""Tests for the asynchronous observation protocol.

The archived asynchronous protocol: each interaction draws a single pair of
agents uniformly at random and delivers that pair's observations immediately,
before the next pair is drawn. Reputations therefore evolve continuously
within a generation, and a later pair already sees the effects of earlier ones.
"""
from __future__ import annotations

import pytest

from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.baselines import BASELINES
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.game import OBSERVATION_SCHEDULE, DonorGame
from experiments.v2_quantitative.population import V2EvolutionaryPopulation


class _CoinAgent:
    """Stub replaying ``QuantitativeAgent``'s surface with visible judgments.

    ``choose`` records the opponent reputation it decided from, so tests can
    observe exactly which information a decision had access to. ``observe``
    judges both participants to a recognizable extreme (-1.0), mirroring the
    real agent, which updates both the donor and the recipient.
    """

    def __init__(self, agent_id, cooperates=True):
        self.agent_id = agent_id
        self.cooperates = cooperates
        self.reputations = {agent_id: 0.0}
        self.decide_reads = []
        self.total_decisions = 0
        self.cooperations = 0

    def get_reputation(self, other_id):
        return self.reputations.get(other_id, 0.0)

    def get_self_reputation(self):
        return self.reputations.get(self.agent_id, 0.0)

    def update_reputation(self, other_id, new_rep):
        self.reputations[other_id] = max(-1.0, min(1.0, float(new_rep)))

    def choose(self, opponent_id, round_num=0):
        self.decide_reads.append(self.get_reputation(opponent_id))
        self.total_decisions += 1
        self.cooperations += int(self.cooperates)
        return self.cooperates

    def record_donation(self, partner_id, donated, round_num):
        pass

    def self_judge(self, donor_action, recipient_id, recipient_action):
        self.observe_and_judge(
            self.agent_id, donor_action, recipient_id, recipient_action
        )

    def observe_and_judge(
        self, donor_id, donor_action, recipient_id, recipient_action
    ):
        self.update_reputation(donor_id, -1.0)
        self.update_reputation(recipient_id, -1.0)


def _make_game(population_size, seed=0):
    game = DonorGame(
        population_size=population_size, observation_schedule="asynchronous",
        benefit=2.0,
        cost=1.0,
        fitness_window_fraction=None,
        seed=seed,
    )
    agents = [_CoinAgent(i) for i in range(population_size)]
    game.setup_population(agents)
    game.payoffs = [0.0] * population_size
    game._global_log = []
    game._interaction_deltas = []
    return game, agents


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------
def _play_steps(game, steps):
    """The engine's generation loop: play one step, then deliver its observations.

    Mirrors ``ReputationPrisonersDilemmaScenario.evaluate``; one iteration is
    one matching round under the synchronous protocol and one pair under the
    asynchronous one.
    """
    for _ in range(steps):
        game.distribute_observations_and_self_judgments(
            game.play_step()["interactions"]
        )


def test_one_step_plays_exactly_one_pair():
    game, _ = _make_game(16)
    step = game.play_interaction()
    assert len(step["interactions"]) == 1
    assert game.round_num == 1


def test_pair_members_are_distinct_and_in_range():
    game, _ = _make_game(16)
    for _ in range(50):
        inter = game.play_interaction()["interactions"][0]
        assert inter["donor"] != inter["recipient"]
        assert 0 <= inter["donor"] < 16
        assert 0 <= inter["recipient"] < 16


def test_per_agent_counts_are_not_forced_equal():
    """Draws are independent, so exposure is multinomial, not a matching."""
    game, _ = _make_game(16)
    counts = {i: 0 for i in range(16)}
    for _ in range(32):
        inter = game.play_interaction()["interactions"][0]
        counts[inter["donor"]] += 1
        counts[inter["recipient"]] += 1
    # 32 interactions = 64 agent-slots among 16 agents; equality would be an
    # absurd coincidence for independent draws.
    assert len(set(counts.values())) > 1


def test_every_agent_acts_given_enough_draws():
    game, _ = _make_game(16)
    for _ in range(400):
        game.play_interaction()
    acted = {inter["donor"] for inter in game._global_log}
    acted |= {inter["recipient"] for inter in game._global_log}
    assert acted == set(range(16))


def test_rejects_a_population_too_small_to_pair():
    game, _ = _make_game(1)
    with pytest.raises(ValueError, match="at least 2 agents"):
        game.play_interaction()


# --------------------------------------------------------------------------
# The defining property: immediate delivery
# --------------------------------------------------------------------------
def test_later_decisions_see_updates_from_earlier_ones():
    """Within one generation, a later decision reads a post-update reputation.

    The RNG is seeded, so this outcome is deterministic rather than flaky.
    """
    game, agents = _make_game(8)
    for _ in range(12):
        step = game.play_interaction()["interactions"]
        game.distribute_observations_and_self_judgments(step)
    reads = [r for a in agents for r in a.decide_reads]
    assert reads
    assert any(r != 0.0 for r in reads), (
        "no decision ever saw an update made earlier in the generation"
    )


def test_the_very_first_decision_starts_from_the_initial_snapshot():
    """Nothing has been observed yet, so the first pair reads neutral."""
    game, agents = _make_game(8)
    game.play_interaction()
    first_reads = [a.decide_reads[0] for a in agents if a.decide_reads]
    assert first_reads
    assert all(r == 0.0 for r in first_reads)


def test_both_members_decide_before_either_update_lands():
    """A pair acts simultaneously: neither half sees the other's effect."""
    game, agents = _make_game(4)
    inter = game.play_interaction()["interactions"][0]
    # Both decisions were recorded before any observation for this step exists.
    assert game._global_log[-1] is inter
    assert agents[inter["donor"]].decide_reads[-1] == 0.0
    assert agents[inter["recipient"]].decide_reads[-1] == 0.0


def test_delivery_is_reproducible_for_a_fixed_seed():
    def run():
        game, _ = _make_game(12, seed=7)
        for _ in range(40):
            game.distribute_observations_and_self_judgments(
                game.play_interaction()["interactions"]
            )
        return list(game.payoffs), [
            (i["donor"], i["recipient"]) for i in game._global_log
        ]

    assert run() == run()


# --------------------------------------------------------------------------
# Window / bookkeeping
# --------------------------------------------------------------------------
def test_window_counts_individual_interactions():
    game, _ = _make_game(16)
    for _ in range(20):
        game.play_interaction()
    assert len(game._interaction_deltas) == 20
    assert len(game.get_windowed_fitness()) == 16


def test_one_delta_and_one_log_entry_per_interaction():
    game, _ = _make_game(16)
    for _ in range(48):
        game.play_interaction()
    assert len(game._global_log) == 48
    assert len(game._interaction_deltas) == 48


def test_each_step_plays_one_pair_and_advances_the_round():
    """Asynchronous: one step is one interaction, and round_num counts them."""
    game, _ = _make_game(9)
    _play_steps(game, 9)
    assert len(game._global_log) == 9
    assert game.round_num == 9


# --------------------------------------------------------------------------
# Population wiring
# --------------------------------------------------------------------------
def _population(**kwargs):
    """A population with the LLM path stubbed out."""
    return V2EvolutionaryPopulation(
        population_size=8, num_generations=1, observation_schedule="asynchronous", **kwargs
    )


def test_default_protocol_is_synchronous():
    assert OBSERVATION_SCHEDULE == "synchronous"


def test_target_interactions_is_the_step_count():
    """One pair plays per interaction, so the target IS the step count."""
    pop = _population(num_rounds_per_gen=0, target_interactions_per_gen=1000)
    assert pop.num_rounds_per_gen == 1000
    assert pop.observation_schedule == "asynchronous"


def test_schedule_is_recorded_in_config():
    config = _population(target_interactions_per_gen=10)._result_config(
        num_generations=1
    )
    assert config["observation_schedule"] == "asynchronous"


def test_rules_prompt_describes_the_protocol():
    prompt = _population(target_interactions_per_gen=250)._overall_game_rules_prompt()
    assert "OVERALL GAME RULES" in prompt
    assert "drawn uniformly at random" in prompt
    assert "delivered immediately" in prompt
    # The removed synchronous wording must not survive anywhere.
    assert "randomly partitioned" not in prompt
    assert "sits out" not in prompt


def test_scenario_reports_no_schedule_choice():
    scenario = _population(target_interactions_per_gen=40)._get_game_scenario()
    params = scenario.simulation_parameters(
        num_generations=1, initial_reputation=0.0
    )
    assert params["num_rounds_per_gen"] == 40
    # These described the synchronous matching and are gone.
    assert "num_pairs" not in params
    assert "observation_schedule" not in params


# --------------------------------------------------------------------------
# The CLI switch between the two protocols
# --------------------------------------------------------------------------
def test_cli_defaults_to_synchronous_and_accepts_asynchronous():
    from experiments import run_fermi_v3 as cli

    parser = cli.build_parser()
    assert parser.parse_args([]).observation_schedule == "synchronous"
    chosen = parser.parse_args(["--observation-schedule", "asynchronous"])
    assert chosen.observation_schedule == "asynchronous"
    with pytest.raises(SystemExit):
        parser.parse_args(["--observation-schedule", "nope"])


def test_only_the_non_default_protocol_changes_the_label():
    """Synchronous must keep its historical label, or archived runs break."""
    from experiments.run_fermi_v3 import schedule_suffix

    assert schedule_suffix("synchronous") == ""
    assert schedule_suffix("asynchronous") == "_asynchronous"


def test_run_one_seed_forwards_the_schedule_to_the_engine(monkeypatch, tmp_path):
    """A flag that never reaches the engine would be decorative."""
    from experiments import run_fermi_v3 as cli

    captured = {}

    class _Stub:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_evolution(self, num_generations):
            raise RuntimeError("stop before any LLM work")

    monkeypatch.setattr(cli, "V2EvolutionaryPopulation", _Stub)
    monkeypatch.setattr(cli, "get_api_key", lambda provider: "test-key")
    args = cli.build_parser().parse_args(
        ["--observation-schedule", "asynchronous", "--gens", "1",
         "--output-root", str(tmp_path)]
    )
    cli.run_one_seed(args, 0, "label", tmp_path)
    assert captured["observation_schedule"] == "asynchronous"


def test_generation_runs_end_to_end():
    code = BASELINES["L1"]
    pop = _population(target_interactions_per_gen=64, benefit=2.0, cost=1.0)
    pop.agents = [
        QuantitativeAgent(i, code, V2StrategyExecutor(code))
        for i in range(pop.population_size)
    ]
    assert pop._run_one_generation()["n_interactions"] == 64
