"""Private signal information boundary and shared experiment integration."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.v2_quantitative.agent_signal import (
    ALLC_SIGNAL_SOURCE, ALLD_SIGNAL_SOURCE, SignalAgent,
)
from experiments.v2_quantitative.signal_executor import (
    SIGNAL_INTERFACE_VERSION, SignalExecutionError, SignalStrategyExecutor,
)
from experiments.v2_quantitative.code_contract import CodeContractError
from experiments.v2_quantitative.population import V2EvolutionaryPopulation
from experiments.v2_quantitative.game import DonorGame
from experiments.evolution_log import validate_evolution_results, write_evolution_json, load_evolution_json
from experiments.analysis.invasion.core import Competitor, EvolvedSource, norm_source, KIND_NORM
from experiments.analysis.invasion.run_invasion import load_candidate, run_one
from experiments.analysis.invasion.run_fixation_benchmark import stationary_mixture


STRUCTURED = '''
from dataclasses import dataclass, field
@dataclass
class Signal:
    observations: int = 0
    context: tuple = ()
    history: list = field(default_factory=list)
class LLMAgent:
    def __init__(self):
        pass
    def observe(self, target_signal, target_action, partner_signal, partner_action, self_signal):
        history = target_signal.history + [target_action]
        return Signal(target_signal.observations + 1,
                      (partner_signal.observations, self_signal.observations), history)
    def decide(self, opponent_signal):
        return opponent_signal.observations == 0 or opponent_signal.history[-1] == 'cooperate'
'''


def agent(code=STRUCTURED, agent_id=10):
    return SignalAgent(agent_id, SignalStrategyExecutor(code), code)


def source(code=STRUCTURED):
    return EvolvedSource("agent-type2-signal", Path("<test-signal>"), -1, -1, -1,
                         0, 0.0, code, hashlib.sha256(code.encode()).hexdigest())


def test_own_observation_uses_pre_event_self_signal_for_both_updates():
    a = agent()
    a.self_judge("cooperate", 20, "defect")
    a.self_judge("defect", 20, "cooperate")
    assert a.get_signal(10).observations == a.get_signal(20).observations == 2
    assert a.get_signal(10).context == a.get_signal(20).context == (1, 1)
    assert a.get_signal(10).history == ["cooperate", "defect"]
    assert a.choose(20) is True
    assert not hasattr(a, "_ctx_opponent_id")
    assert vars(a._executor.brain) == {}


def test_third_party_observation_does_not_overwrite_self_signal():
    a = agent()
    a.self_judge("cooperate", 20, "defect")
    a.observe_and_judge(20, "defect", 30, "cooperate")
    assert a.get_signal(10).observations == 1
    assert a.get_signal(20).context == (0, 1)
    assert a.get_signal(30).context == (1, 1)


def test_aliases_are_broken_even_when_self_is_a_participant():
    code = STRUCTURED.replace(
        "history = target_signal.history + [target_action]",
        "target_signal.history.append(target_action)\n        history = self_signal.history.copy()",
    ).replace(
        "return opponent_signal.observations == 0 or opponent_signal.history[-1] == 'cooperate'",
        "return True",
    )
    a = agent(code)
    a.self_judge("cooperate", 20, "defect")
    assert a.get_signal(10).history == a.get_signal(20).history == []


def test_decision_cannot_mutate_framework_memory():
    code = STRUCTURED.replace(
        "return opponent_signal.observations == 0 or opponent_signal.history[-1] == 'cooperate'",
        "opponent_signal.history.append('defect')\n        opponent_signal.observations = 999\n        return True",
    )
    a = agent(code)
    a.choose(20)
    assert a.get_signal(20).history == []
    assert a.get_signal(20).observations == 0


def test_relabeling_ids_preserves_all_signal_values_and_actions():
    left, right = agent(agent_id=10), agent(agent_id=902)
    mapping = {10: 902, 20: -55, 30: 17001}
    for x, ax, y, ay in [(10, "cooperate", 20, "defect"), (20, "defect", 30, "cooperate"),
                         (30, "cooperate", 10, "cooperate")]:
        left.observe_and_judge(x, ax, y, ay)
        right.observe_and_judge(mapping[x], ax, mapping[y], ay)
    for key in mapping:
        assert left.choose(key) == right.choose(mapping[key])
        assert left._executor.signal_record(left.get_signal(key)) == right._executor.signal_record(right.get_signal(mapping[key]))


def test_heterogeneous_schemas_are_private_and_wrong_protocol_is_rejected():
    a, b = agent(), agent(ALLC_SIGNAL_SOURCE, 20)
    a.self_judge("cooperate", 20, "defect")
    b.self_judge("defect", 10, "cooperate")
    assert a.get_signal(20).history == ["defect"]
    assert b.get_signal(10).observations == 1
    with pytest.raises(SignalExecutionError, match="this strategy's Signal"):
        b._executor.decide(a.get_signal(20))
    assert b._executor.errors["decide"] == 1


@pytest.mark.parametrize("body", [
    "return self.agent_id == 0", "return self._ctx_opponent_id == 1",
    "return id(opponent_signal) == 0", "return hash(opponent_signal) == 0",
    "return opponent_signal.__class__ is Signal", "return globals()['agent_id'] == 0",
    "return getattr(opponent_signal, '__class__') == Signal",
    "self.memory = {}\n        return True", "x = self\n        return True",
    "return opponent_signal is Signal()", "return '{0.__class__}'.format(opponent_signal) == ''",
])
def test_identity_reflection_and_hidden_memory_are_rejected(body):
    code = ALLC_SIGNAL_SOURCE.replace("return True", body)
    with pytest.raises(CodeContractError):
        SignalStrategyExecutor(code)


@pytest.mark.parametrize("replacement", ["return 1", "return None", "return Signal()"])
def test_decide_requires_an_actual_bool(replacement):
    with pytest.raises(CodeContractError, match="bool"):
        SignalStrategyExecutor(ALLC_SIGNAL_SOURCE.replace("return True", replacement))


def test_dataclass_default_protocol_instance_methods_and_imports():
    invalid = [
        ALLC_SIGNAL_SOURCE.replace("@dataclass\n", ""),
        ALLC_SIGNAL_SOURCE.replace("observations: int = 0", "observations: int"),
        ALLC_SIGNAL_SOURCE.replace("    def decide", "    @staticmethod\n    def decide"),
        "import os\n" + ALLC_SIGNAL_SOURCE,
        ALLC_SIGNAL_SOURCE.replace("observations: int = 0", "agent_id: int = 0"),
        ALLC_SIGNAL_SOURCE.replace("observations: int = 0", "observations: int = field(default_factory=lambda x=[]: len(x))"),
    ]
    for code in invalid:
        with pytest.raises(CodeContractError):
            SignalStrategyExecutor(code)
    SignalStrategyExecutor("from __future__ import annotations\n" + STRUCTURED)


def test_budget_stops_a_nonterminating_rule():
    with pytest.raises(CodeContractError, match="budget"):
        SignalStrategyExecutor(ALLC_SIGNAL_SOURCE.replace("return True", "while True:\n            pass"))


def test_helper_methods_and_math_are_available_without_policy_state():
    code = ALLC_SIGNAL_SOURCE.replace("from dataclasses", "import math\nfrom dataclasses").replace(
        "return True", "return self.accept(opponent_signal.observations)"
    ) + "\n    def accept(self, count):\n        return math.tanh(count) >= 0\n"
    assert agent(code).choose(50) is True


def test_random_stream_is_repeatable_without_ids_or_validation_side_effects():
    code = "import random\n" + ALLC_SIGNAL_SOURCE.replace("return True", "return random.random() < 0.5")
    left = SignalAgent(1, SignalStrategyExecutor(code, seed=41), code)
    right = SignalAgent(987654, SignalStrategyExecutor(code, seed=41), code)
    expected_rng = random.Random(41)
    expected = [expected_rng.random() < 0.5 for _ in range(30)]
    assert [left.choose(2) for _ in expected] == expected
    assert [right.choose(-200) for _ in expected] == expected
    assert left.signal_metadata()["signal_rng_seed"] == 41
    for expression in ("random.seed(12)", "random.getstate()", "random.Random(12)"):
        bad = code.replace("random.random() < 0.5", expression)
        with pytest.raises(CodeContractError, match="attribute access"):
            SignalStrategyExecutor(bad)


def test_private_signal_population_seed_repeats_stochastic_evolution():
    code = "import random\n" + ALLC_SIGNAL_SOURCE.replace("return True", "return random.random() < 0.5")
    def run():
        pop = population()
        pop._request_valid_code = lambda prompt, label: code
        return pop.run_evolution(2)
    first, second = run(), run()
    assert first == second


def test_runtime_error_does_not_partially_commit_or_silently_defect():
    code = STRUCTURED.replace(
        "history = target_signal.history + [target_action]",
        "if target_signal.observations > 0 and target_action == 'defect':\n            return None\n        history = target_signal.history + [target_action]",
    )
    a = agent(code)
    a.self_judge("cooperate", 20, "cooperate")
    with pytest.raises(SignalExecutionError, match="observe failed"):
        a.self_judge("cooperate", 20, "defect")
    assert a.get_signal(10).observations == a.get_signal(20).observations == 1
    assert a._executor.errors["observe"] == 1


def test_defaults_reset_and_replacement_do_not_share_state():
    a, b = agent(), agent(agent_id=20)
    a.self_judge("cooperate", 20, "defect")
    assert b.get_signal(10).history == []
    a.handle_agents_replaced([20], [30])
    assert 20 not in a.signals
    a.reset_for_generation()
    assert list(a.signals) == [10]
    assert a.get_signal(10).observations == 0


def test_private_observability_still_updates_both_participants():
    agents = [agent(ALLC_SIGNAL_SOURCE, i) for i in range(3)]
    game = DonorGame(population_size=3, observability="private", seed=4)
    game.setup_population(agents)
    interactions = game.play_interaction()["interactions"]
    game.distribute_observations_and_self_judgments(interactions)
    pair = {interactions[0]["donor"], interactions[0]["recipient"]}
    for a in agents:
        assert a.get_signal(a.agent_id).observations == int(a.agent_id in pair)


def population(**kwargs):
    defaults = dict(agent_type="agent-type2-signal", population_size=4, elite_count=1,
                    num_eliminate=1, tournament_size=2, num_rounds_per_gen=12,
                    learning_method="fermi", updates_per_gen=2, seed=9)
    defaults.update(kwargs)
    pop = V2EvolutionaryPopulation(**defaults)
    pop._llm_client = object()
    pop._request_valid_code = lambda prompt, label: ALLC_SIGNAL_SOURCE
    return pop


@pytest.mark.parametrize("learning_method", ["fermi", "tournament"])
def test_evolution_logs_and_generation_cleanup(learning_method, tmp_path):
    pop = population(learning_method=learning_method)
    result = pop.run_evolution(2)
    validate_evolution_results(result)
    record = result["final_population"][0]
    assert record["self_reputation"] is None
    assert record["self_signal"]["dataclass"] == "Signal"
    assert record["signal_schema"][0]["name"] == "observations"
    assert result["config"]["signal_interface_version"] == SIGNAL_INTERFACE_VERSION
    assert result["config"]["signal_observations_include_participants"] is True
    path = write_evolution_json(tmp_path / "evolutionary.json", result)
    assert load_evolution_json(path)["config"]["agent_type"] == "agent-type2-signal"
    candidate = load_candidate("signal", "agent-type2-signal", str(path))
    assert candidate.agent_type == "agent-type2-signal"


def test_signal_fallback_is_valid_and_counted():
    pop = population()
    pop._request_valid_code = lambda prompt, label: None
    result = pop.run_evolution(1)
    assert result["config"]["fallback_init_count"] == 4
    assert all(a.code in (ALLC_SIGNAL_SOURCE, ALLD_SIGNAL_SOURCE) for a in pop.agents)


def test_mutation_prompt_routes_all_signal_variants():
    pop = population()
    prompts = []
    pop._request_valid_code = lambda prompt, label: prompts.append(prompt) or ALLC_SIGNAL_SOURCE
    pop._mutate_code(ALLC_SIGNAL_SOURCE, 1.0)
    pop._llm_small_mutate_code(ALLC_SIGNAL_SOURCE, 1.0, 4)
    pop.imitation_learning_mode = "deliberate"
    pop._llm_small_mutate_code(ALLC_SIGNAL_SOURCE, 1.0, 4)
    assert "self_signal" in pop._init_prompt()
    assert all("self_signal" in p and "@dataclass" in p for p in prompts)
    assert all("_ctx_opponent_id" not in p for p in prompts)


def test_invasion_py_source_retains_selected_agent_family(tmp_path):
    path = tmp_path / "signal.py"
    path.write_text(ALLC_SIGNAL_SOURCE, encoding="utf-8")
    candidate = load_candidate("s", "agent-type2-signal", str(path))
    assert candidate.agent_type == "agent-type2-signal"
    run = run_one(candidate, "s", norm_source("L1"), "L1", KIND_NORM, 2, 0,
                  generations=2, interactions=20, population_size=4)
    assert len(run["trajectory"]) >= 1
    assert run["candidate_source"]["agent_type"] == "agent-type2-signal"


def test_fixation_uses_accumulating_signal_memory_and_matches_scalar_allc():
    result = stationary_mixture(source(ALLC_SIGNAL_SOURCE), norm_source("ALLC"),
                                2, 0, population_size=4, burn_in=20, measure=20)
    assert result["payoff_difference"] == pytest.approx(0.0)
    counting = STRUCTURED.replace(
        "return opponent_signal.observations == 0 or opponent_signal.history[-1] == 'cooperate'",
        "return opponent_signal.observations < 25",
    )
    # All agents witness every event. A tracked target can accumulate >25
    # observations only when memory persists throughout the stationary run.
    tracked = Competitor.create(0, "mutant", "s", source(counting))
    for _ in range(26):
        tracked.observe(1, "cooperate", 2, "defect")
    assert tracked.choose(1) is False


def test_fixation_stochastic_signal_uses_the_run_seed_not_global_random():
    code = "import random\n" + ALLC_SIGNAL_SOURCE.replace("return True", "return random.random() < 0.5")
    candidate = source(code)
    random.seed(111)
    first = stationary_mixture(candidate, norm_source("ALLC"), 2, 13,
                               population_size=4, burn_in=20, measure=50)
    random.seed(999)
    second = stationary_mixture(candidate, norm_source("ALLC"), 2, 13,
                                population_size=4, burn_in=20, measure=50)
    assert first == second


def test_type1_record_format_and_behavior_stay_scalar():
    pop = population(agent_type="agent-type1", use_baseline="ALLC")
    result = pop.run_evolution(1)
    record = result["final_population"][0]
    assert isinstance(record["self_reputation"], float)
    assert not any(key.startswith("signal_") for key in record)
    assert "self_signal" not in record
    assert "signal_interface_version" not in result["config"]
    assert result["trajectory"][0]["cooperation_rate_mean"] == 1.0


def test_cli_signal_mode_is_opt_in():
    from experiments.run_fermi_v3 import build_parser
    parser = build_parser()
    assert parser.parse_args([]).agent_type == "agent-type2"
    assert parser.parse_args(["--agent-type", "agent-type2-signal"]).agent_type == "agent-type2-signal"


def test_signal_benchmark_cache_requires_interface_version():
    from experiments.analysis.invasion.run_invasion import cache_matches
    from experiments.analysis.invasion.run_fixation_benchmark import cache_matches as fixation_cache
    candidate, resident = source(ALLC_SIGNAL_SOURCE), norm_source("L1")
    run = run_one(candidate, "s", resident, "L1", KIND_NORM, 2, 0,
                  generations=1, interactions=20, population_size=4)
    args = (1, 20, 0.2, 0.0, 0.0, 4)
    assert cache_matches(run, candidate, resident, "L1", *args)
    run["candidate_source"]["signal_interface_version"] = "obsolete"
    assert not cache_matches(run, candidate, resident, "L1", *args)
    settings = SimpleNamespace(population_size=4, burn_in=20, measure=20,
        beta=1.0, action_error=0.0, observation_error=0.0, replicates=1,
        probes=["L1"], benefit=2.0, cost=1.0, observation_schedule="synchronous")
    stored = {
        "config": {"population_size": 4, "burn_in_interactions": 20,
                   "measure_interactions": 20, "beta": 1.0,
                   "action_error_probability": 0.0, "observation_error_probability": 0.0,
                   "benefit": 2.0, "cost": 1.0, "observation_schedule": "synchronous"},
        "candidate": {"label": "s", "code_sha256": candidate.code_sha256,
                      "agent_type": candidate.agent_type,
                      "signal_interface_version": SIGNAL_INTERFACE_VERSION},
        "probes": ["L1"],
    }
    assert fixation_cache(stored, settings, candidate, "s")
    del stored["candidate"]["signal_interface_version"]
    assert not fixation_cache(stored, settings, candidate, "s")
