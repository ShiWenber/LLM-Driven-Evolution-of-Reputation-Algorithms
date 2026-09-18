"""Check fixation scheduling against the main engine, including noise draws."""
import random

import pytest

from experiments.analysis.invasion.core import Competitor
from experiments.analysis.invasion.run_fixation_benchmark import probe_source, stationary_mixture
from experiments.v2_quantitative.game import DonorGame


@pytest.mark.parametrize("n", [4, 5])
@pytest.mark.parametrize("error", [0.0, 0.2])
def test_synchronous_mixture_matches_main_engine(n, error, monkeypatch):
    seed, k, burn, measure = 31, 2, 20, 100
    candidate, resident = probe_source("L6"), probe_source("ALLD")
    created = []
    original = Competitor.create

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(Competitor, "create", capture)
    measured = stationary_mixture(candidate, resident, k, seed, n, burn, measure,
                                  action_error=error, observation_error=error,
                                  benefit=2, cost=1, observation_schedule="synchronous")
    actual = list(created)
    rng = random.Random(seed)
    slots = list(range(n))
    rng.shuffle(slots)
    mutants = set(slots[:k])
    population = [original(slot, "mutant" if slot in mutants else "resident", "reference",
                           candidate if slot in mutants else resident) for slot in slots]
    game = DonorGame(n, benefit=2, cost=1, action_error_probability=error,
                     observation_error_probability=error)
    game.setup_population([member.agent for member in population])
    game.rng.setstate(rng.getstate())
    for _ in range((burn + measure) // (n // 2)):
        interactions = game.play_round()["interactions"]
        game.distribute_observations_and_self_judgments(interactions)
    for left, right in zip(actual, population):
        assert left.agent.reputations == right.agent.reputations
    for kind in ("mutant", "resident"):
        positions = [i for i, member in enumerate(population) if member.kind == kind]
        total = sum(sum(delta[i] for i in positions) for delta in game._interaction_deltas[burn:])
        assert measured[f"payoff_{kind}"] == total / measured[f"{kind}_participation"]


def test_synchronous_actions_precede_all_observations(monkeypatch):
    events = []
    original_choose, original_observe = Competitor.choose, Competitor.observe

    def choose(self, other):
        events.append("action")
        return original_choose(self, other)

    def observe(self, *args):
        events.append("observation")
        return original_observe(self, *args)

    monkeypatch.setattr(Competitor, "choose", choose)
    monkeypatch.setattr(Competitor, "observe", observe)
    for schedule, expected in [("synchronous", ["action"] * 4),
                               ("asynchronous", ["action", "action", "observation", "observation"])]:
        events.clear()
        stationary_mixture(probe_source("L1"), probe_source("ALLD"), 1, 3,
                           population_size=4, burn_in=0, measure=2, observation_schedule=schedule)
        assert events[:4] == expected
