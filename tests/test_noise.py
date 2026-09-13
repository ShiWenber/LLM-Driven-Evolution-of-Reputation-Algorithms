"""Action (execution) error and reputation (perception) error in the engine.

Two independent noise sources, both off by default:

  * ``action_error_probability`` -- execution error. A player's intended
    action is mis-executed. The executed action drives payoffs and is what
    every observer sees.
  * ``observation_error_probability`` -- perception error. An observer
    misperceives an action while rating it, corrupting the reputation it
    writes. Payoffs are untouched, and every observer draws independently,
    the two participants included.

The load-bearing property tested here is that a run with both knobs at 0.0 is
bit-identical to a run from before these knobs existed. That is what keeps the
archived results reproducible, and it is why both flips must return before
touching the RNG.
"""
from __future__ import annotations

import json

import pytest

from experiments.evolution_log import (
    F_CONFIG_ACTION_ERROR,
    F_CONFIG_OBSERVATION_ERROR,
    SCHEMA_VERSION,
)
from experiments.run_fermi_v3 import build_parser, noise_suffix
from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.baselines import BASELINES
from experiments.v2_quantitative.evolution_architecture import (
    ReputationPrisonersDilemmaScenario,
)
from experiments.v2_quantitative.executor import V2StrategyExecutor
from experiments.v2_quantitative.game import DonorGame, flip_action
from experiments.v2_quantitative.population import V2EvolutionaryPopulation


class _Recorder:
    """Agent that always intends to cooperate and records what it is shown."""

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.reputations = {agent_id: 0.0}
        self.seen = []
        self.total_decisions = 0
        self.cooperations = 0

    def get_reputation(self, other_id):
        return self.reputations.get(other_id, 0.0)

    def get_self_reputation(self):
        return self.reputations.get(self.agent_id, 0.0)

    def update_reputation(self, other_id, new_rep):
        self.reputations[other_id] = max(-1.0, min(1.0, float(new_rep)))

    def choose(self, opponent_id, round_num=0):
        self.total_decisions += 1
        self.cooperations += 1
        return True

    def record_donation(self, partner_id, donated, round_num):
        pass

    def observe_and_judge(
        self, donor_id, donor_action, recipient_id, recipient_action
    ):
        self.seen.append((donor_id, donor_action, recipient_id, recipient_action))
        self.update_reputation(
            donor_id, 1.0 if donor_action == "cooperate" else -1.0
        )
        self.update_reputation(
            recipient_id, 1.0 if recipient_action == "cooperate" else -1.0
        )

    def self_judge(self, donor_action, recipient_id, recipient_action):
        self.observe_and_judge(
            self.agent_id, donor_action, recipient_id, recipient_action
        )

    def reset_for_generation(self):
        self.reputations = {self.agent_id: 0.0}
        self.total_decisions = 0
        self.cooperations = 0


def _game(population_size=2, agents=None, **kwargs):
    game = DonorGame(
        population_size=population_size,
        benefit=3.0,
        cost=1.0,
        fitness_window_fraction=None,
        **{"seed": 0, **kwargs},
    )
    game.setup_population(agents or [_Recorder(i) for i in range(population_size)])
    game.payoffs = [0.0] * population_size
    game._global_log = []
    game._interaction_deltas = []
    return game, game.agents


def _population(**kwargs):
    return V2EvolutionaryPopulation(population_size=4, num_generations=1, **kwargs)


# --------------------------------------------------------------------------
# flip_action: the single definition both noise sources go through
# --------------------------------------------------------------------------
def test_flip_action_never_touches_rng_at_zero_probability():
    """Off means off: no draw is consumed, so the stream cannot shift."""
    game, _ = _game()
    before = game.rng.getstate()
    for _ in range(50):
        assert flip_action("cooperate", game.rng, 0.0) == "cooperate"
    assert game.rng.getstate() == before


def test_flip_action_flips_certainly_at_probability_one():
    game, _ = _game()
    assert flip_action("cooperate", game.rng, 1.0) == "defect"
    assert flip_action("defect", game.rng, 1.0) == "cooperate"


def test_flip_action_is_reproducible_and_hits_the_requested_rate():
    game, _ = _game()
    flips = [
        flip_action("cooperate", game.rng, 0.25) == "defect" for _ in range(4000)
    ]
    assert 0.22 < sum(flips) / len(flips) < 0.28
    # Same seed -> same draw sequence.
    other, _ = _game()
    assert flips == [
        flip_action("cooperate", other.rng, 0.25) == "defect" for _ in range(4000)
    ]


# --------------------------------------------------------------------------
# Backward compatibility: zero noise is the old engine, exactly
# --------------------------------------------------------------------------
def test_zero_noise_reproduces_the_pre_noise_engine_bit_for_bit():
    def run(**kwargs):
        game, _ = _game(6, **kwargs)
        for _ in range(40):
            game.distribute_observations_and_self_judgments(
                game.play_interaction()["interactions"]
            )
        return list(game.payoffs), [dict(entry) for entry in game._global_log]

    # Omitting the knobs entirely must equal passing 0.0 explicitly.
    assert run() == run(action_error_probability=0.0, observation_error_probability=0.0)


def test_zero_noise_generation_matches_a_scenario_without_noise():
    agents = [
        QuantitativeAgent(i, BASELINES["L1"], V2StrategyExecutor(BASELINES["L1"]))
        for i in range(4)
    ]
    plain = DonorGame(population_size=4, fitness_window_fraction=None, seed=11)
    plain.setup_population(agents)
    for _ in range(20):
        plain.distribute_observations_and_self_judgments(
            plain.play_interaction()["interactions"]
        )

    fresh_agents = [
        QuantitativeAgent(i, BASELINES["L1"], V2StrategyExecutor(BASELINES["L1"]))
        for i in range(4)
    ]
    noisy = DonorGame(
        population_size=4,
        fitness_window_fraction=None,
        seed=11,
        action_error_probability=0.0,
        observation_error_probability=0.0,
    )
    noisy.setup_population(fresh_agents)
    for _ in range(20):
        noisy.distribute_observations_and_self_judgments(
            noisy.play_interaction()["interactions"]
        )

    assert plain.payoffs == noisy.payoffs
    assert plain._global_log == noisy._global_log


# --------------------------------------------------------------------------
# Action error: execution error
# --------------------------------------------------------------------------
def test_action_error_at_one_turns_intended_cooperation_into_defection():
    game, agents = _game(2, action_error_probability=1.0)
    inter = game.play_interaction()["interactions"][0]
    # Both intended to cooperate; both mis-executed.
    assert inter["donor_action"] == "defect"
    assert inter["recipient_action"] == "defect"
    # Mutual defection pays nothing to either player.
    assert game.payoffs == [0.0, 0.0]


def test_no_action_error_keeps_mutual_cooperation_paying():
    game, _ = _game(2, action_error_probability=0.0)
    game.play_interaction()
    # (C, C) -> each benefit - cost.
    assert game.payoffs == [2.0, 2.0]


def test_action_error_is_recorded_as_the_executed_action():
    """The logged action is what happened, so cooperation metrics stay honest."""
    game, _ = _game(2, action_error_probability=1.0)
    game.play_interaction()
    assert game._global_log[0]["donor_action"] == "defect"
    assert game._global_log[0]["recipient_action"] == "defect"


def test_action_error_at_one_makes_a_generation_fully_defect():
    game, _ = _game(6, action_error_probability=1.0)
    stats = game.run_generation()
    assert stats["cooperation_rate_mean"] == 0.0


def test_action_error_reaches_observers_not_just_payoffs():
    """A mis-execution is visible: everyone rates the executed action."""
    game, agents = _game(3, action_error_probability=1.0)
    game.distribute_observations_and_self_judgments(
        game.play_interaction()["interactions"]
    )
    # Every recorded action in every observer's view is the defect.
    assert all(
        donor_action == "defect" and recipient_action == "defect"
        for agent in agents
        for _, donor_action, _, recipient_action in agent.seen
    )


# --------------------------------------------------------------------------
# Reputation error: perception error
# --------------------------------------------------------------------------
def test_observation_error_does_not_change_payoffs_or_the_log():
    """Perception corrupts ratings, not outcomes."""

    def play(observation_error):
        game, agents = _game(3, observation_error_probability=observation_error)
        # Same explicit pair, so the pairing RNG stream is not a confound.
        game.distribute_observations_and_self_judgments(
            [game._play_pair(0, 1)]
        )
        return list(game.payoffs), [dict(e) for e in game._global_log], agents

    clean_payoffs, clean_log, _ = play(0.0)
    noisy_payoffs, noisy_log, _ = play(1.0)
    assert noisy_payoffs == clean_payoffs
    assert noisy_log == clean_log


def test_observation_error_at_one_does_corrupt_the_reputation():
    game, agents = _game(3, observation_error_probability=1.0)
    game.distribute_observations_and_self_judgments([game._play_pair(0, 1)])
    # Nobody cooperated in anyone's view, so every rating is -1.
    assert all(
        agent.get_reputation(0) == -1.0 and agent.get_reputation(1) == -1.0
        for agent in agents
    )


def test_observation_error_can_make_a_self_judgment_wrong():
    """The participants are observers too, so self-ratings are fallible.

    This matches the invasion / fixation harness, whose observer loop also
    covers the two participants.
    """
    game, agents = _game(2, observation_error_probability=1.0)
    game.distribute_observations_and_self_judgments([game._play_pair(0, 1)])
    # Agent 0 intended and executed cooperation, yet rated itself as a defector.
    assert agents[0].get_self_reputation() == -1.0
    assert agents[1].get_self_reputation() == -1.0


def test_observation_error_is_independent_across_observers():
    """Two agents can walk away with opposite views of the same interaction."""
    game, agents = _game(4, observation_error_probability=0.5)
    disagreed = False
    for _ in range(40):
        for agent in agents:
            agent.seen.clear()
        game.distribute_observations_and_self_judgments([game._play_pair(0, 1)])
        views = {
            tuple(entry[1::2]) for entry in (a.seen[-1] for a in agents)
        }
        if len(views) > 1:
            disagreed = True
            break
    assert disagreed, "every observer misperceived in lockstep"


def test_observation_error_is_reproducible_for_a_fixed_seed():
    def run():
        game, agents = _game(5, observation_error_probability=0.4, seed=3)
        for _ in range(15):
            game.distribute_observations_and_self_judgments(
                game.play_interaction()["interactions"]
            )
        return [dict(a.reputations) for a in agents]

    assert run() == run()


def test_partial_observability_still_applies_perception_error():
    """Noise composes with the observability gate instead of bypassing it."""
    game, agents = _game(
        4,
        observation_error_probability=1.0,
        observability="partial",
        observability_p=0.0,
    )
    game.distribute_observations_and_self_judgments([game._play_pair(0, 1)])
    # Nobody observed, so third parties never rated the pair...
    for agent in agents[2:]:
        assert agent.get_reputation(0) == 0.0
    # ...while the participants still mis-rated themselves.
    assert agents[0].get_self_reputation() == -1.0


def test_private_observability_still_applies_self_judgment_noise():
    game, agents = _game(
        3, observation_error_probability=1.0, observability="private"
    )
    game.distribute_observations_and_self_judgments([game._play_pair(0, 1)])
    assert agents[0].get_self_reputation() == -1.0
    assert agents[2].get_reputation(0) == 0.0


# --------------------------------------------------------------------------
# Scenario / population wiring
# --------------------------------------------------------------------------
def test_scenario_forwards_noise_to_the_engine():
    agents = [
        QuantitativeAgent(i, BASELINES["ALLC"], V2StrategyExecutor(BASELINES["ALLC"]))
        for i in range(4)
    ]
    scenario = ReputationPrisonersDilemmaScenario(
        population_size=4,
        benefit=3.0,
        cost=1.0,
        observability="full",
        observability_p=1.0,
        fitness_window_fraction=None,
        num_rounds_per_gen=10,
        action_error_probability=1.0,
        observation_error_probability=0.0,
    )
    result = scenario.evaluate(agents, generation_seed=0, num_rounds=10)
    # Always-cooperate agents under certain execution error defect every time.
    assert result.cooperation_rate_mean == 0.0


def test_scenario_forwards_perception_error_to_the_engine():
    """The scenario must hand the perception knob to the engine, not drop it."""
    agents = [_Recorder(i) for i in range(4)]
    scenario = ReputationPrisonersDilemmaScenario(
        population_size=4,
        benefit=3.0,
        cost=1.0,
        observability="full",
        observability_p=1.0,
        fitness_window_fraction=None,
        num_rounds_per_gen=10,
        action_error_probability=0.0,
        observation_error_probability=1.0,
    )
    scenario.evaluate(agents, generation_seed=0, num_rounds=10)
    # Every agent cooperated, yet every agent's perception inverted it.
    assert all(
        agent.get_reputation(agent.agent_id) == -1.0 for agent in agents
    )


def test_population_records_both_noise_rates_in_config():
    pop = _population(
        target_interactions_per_gen=10,
        action_error_probability=0.01,
        observation_error_probability=0.02,
    )
    config = pop._result_config(num_generations=1)
    assert config[F_CONFIG_ACTION_ERROR] == 0.01
    assert config[F_CONFIG_OBSERVATION_ERROR] == 0.02


def test_population_defaults_to_zero_noise():
    config = _population(target_interactions_per_gen=10)._result_config(
        num_generations=1
    )
    assert config[F_CONFIG_ACTION_ERROR] == 0.0
    assert config[F_CONFIG_OBSERVATION_ERROR] == 0.0


def test_population_passes_noise_into_its_default_scenario():
    pop = _population(
        target_interactions_per_gen=10,
        action_error_probability=0.05,
        observation_error_probability=0.07,
    )
    scenario = pop._get_game_scenario()
    assert scenario.action_error_probability == 0.05
    assert scenario.observation_error_probability == 0.07


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_population_rejects_out_of_range_noise(bad):
    with pytest.raises(ValueError, match="action_error_probability"):
        _population(action_error_probability=bad)
    with pytest.raises(ValueError, match="observation_error_probability"):
        _population(observation_error_probability=bad)


def test_zero_noise_population_generation_is_unchanged_by_the_feature():
    def run(**kwargs):
        pop = _population(target_interactions_per_gen=64, benefit=2.0, cost=1.0, **kwargs)
        pop.agents = [
            QuantitativeAgent(i, BASELINES["L1"], V2StrategyExecutor(BASELINES["L1"]))
            for i in range(pop.population_size)
        ]
        return pop._run_one_generation()

    assert run() == run(action_error_probability=0.0, observation_error_probability=0.0)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_parser_defaults_to_no_noise():
    args = build_parser().parse_args([])
    assert args.action_error == 0.0
    assert args.observation_error == 0.0


def test_parser_accepts_the_noise_flags():
    args = build_parser().parse_args(
        ["--action-error", "0.01", "--observation-error", "0.02"]
    )
    assert args.action_error == 0.01
    assert args.observation_error == 0.02


def test_reputation_error_is_an_alias_for_observation_error():
    args = build_parser().parse_args(["--reputation-error", "0.03"])
    assert args.observation_error == 0.03


@pytest.mark.parametrize("value", ["-0.1", "1.5"])
def test_main_rejects_out_of_range_noise(value):
    from experiments.run_fermi_v3 import main

    with pytest.raises(SystemExit):
        main(["--action-error", value, "--dry-run"])
    with pytest.raises(SystemExit):
        main(["--observation-error", value, "--dry-run"])


def test_noise_suffix_is_empty_without_noise_and_names_the_condition():
    assert noise_suffix(0.0, 0.0) == ""
    assert noise_suffix(0.01, 0.01) == "_ae0p01_oe0p01"
    assert noise_suffix(0.0, 0.05) == "_ae0_oe0p05"


def test_dry_run_label_carries_the_noise_level(capsys):
    """A noisy run must not land in a noise-free run's output directory."""
    from experiments.run_fermi_v3 import main

    main(["--action-error", "0.01", "--observation-error", "0.01", "--dry-run"])
    printed = capsys.readouterr().out
    assert "ae0p01_oe0p01" in printed
    assert "noise: action_error=0.01, observation_error=0.01" in printed


# --------------------------------------------------------------------------
# Resume: the noise model is inherited from the log, never re-specified
# --------------------------------------------------------------------------
def _write_log(path, noise_keys):
    population = [
        {
            "agent_id": agent_id,
            "code": "def decide(my_reputation, opponent_reputation):\n    return True\n",
            "fitness": 1.0,
            "cooperation_rate": 0.0,
            "self_reputation": 0.0,
            "lineage_id": 100 + agent_id,
            "parent_id": None,
            "parent_lineage_id": None,
            "origin": "initial",
            "birth_gen": 0,
        }
        for agent_id in (0, 1)
    ]
    path.write_text(
        json.dumps(
            {
                "trajectory": [
                    {
                        "generation": 0,
                        "cooperation_rate_mean": 0.5,
                        "n_interactions": 2,
                        "fitness_mean": 1.0,
                        "fitness_max": 1.0,
                        "population": population,
                    }
                ],
                "final_population": population,
                "lineage_events": [],
                "config": {
                    "schema_version": SCHEMA_VERSION,
                    "agent_type": "agent-type1",
                    "seed": 5,
                    "population_size": 2,
                    "learning_method": "fermi",
                    "observation_schedule": "asynchronous",
                    **noise_keys,
                },
            }
        ),
        encoding="utf-8",
    )


def _valid_result(previous):
    """A stub resume result that still satisfies the log's schema contract."""
    return {
        "trajectory": previous["trajectory"],
        "final_population": previous["final_population"],
        "lineage_events": [],
        "config": {
            "schema_version": SCHEMA_VERSION,
            "agent_type": "agent-type1",
            "seed": 5,
            "population_size": 2,
            "resume": {"rng_mode": "checkpoint"},
        },
    }


def _run_stubbed_resume(tmp_path, monkeypatch, noise_keys, label, extra=()):
    """Drive ``run_one_resume`` with the population and the API key stubbed.

    Returns the kwargs the real code would have built the population with:
    that is where an inherited noise level shows up.
    """
    from experiments import run_fermi_v3 as cli

    source = tmp_path / "old.json"
    _write_log(source, noise_keys)

    captured = {}

    class _StubPopulation:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def resume_evolution(self, previous, additional, *, source_path=None):
            return _valid_result(previous)

    monkeypatch.setattr(cli, "V2EvolutionaryPopulation", _StubPopulation)
    monkeypatch.setattr(cli, "get_api_key", lambda provider: "test-key")

    args = cli.build_parser().parse_args(
        ["--resume-json", str(source), "--additional-gens", "1",
         "--label", label, "--output-root", str(tmp_path / "out"), *extra]
    )
    cli.run_one_resume(args, source, label, tmp_path / "out")
    return captured


@pytest.mark.parametrize(
    "noise_keys, expected",
    [
        ({F_CONFIG_ACTION_ERROR: 0.01, F_CONFIG_OBSERVATION_ERROR: 0.02}, (0.01, 0.02)),
        # A log written before noise existed must resume noise-free.
        ({}, (0.0, 0.0)),
    ],
)
def test_resume_inherits_the_recorded_noise(
    tmp_path, monkeypatch, noise_keys, expected
):
    captured = _run_stubbed_resume(tmp_path, monkeypatch, noise_keys, "resumed")

    assert captured["action_error_probability"] == expected[0]
    assert captured["observation_error_probability"] == expected[1]


def test_resume_ignores_cli_noise_because_the_lineage_already_fixed_it(
    tmp_path, monkeypatch
):
    """Passing --action-error on resume must not rewrite a recorded level."""
    captured = _run_stubbed_resume(
        tmp_path, monkeypatch, {}, "resumed2", extra=["--action-error", "0.5"]
    )

    assert captured["action_error_probability"] == 0.0
