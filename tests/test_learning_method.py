"""Configuration and dispatch tests for population learning rules."""

import pytest

from experiments.run_fermi_v3 import build_parser
from experiments.v2_quantitative.population import V2EvolutionaryPopulation


def test_cli_defaults_to_fermi_and_accepts_tournament():
    parser = build_parser()

    assert parser.parse_args([]).learning_method == "fermi"
    assert (
        parser.parse_args(["--learning-method", "tournament"]).learning_method
        == "tournament"
    )


def test_legacy_use_fermi_is_still_supported():
    assert V2EvolutionaryPopulation(population_size=4).learning_method == "tournament"
    assert (
        V2EvolutionaryPopulation(population_size=4, use_fermi=True).learning_method
        == "fermi"
    )


def test_explicit_learning_method_is_authoritative():
    population = V2EvolutionaryPopulation(
        population_size=4,
        use_fermi=True,
        learning_method="tournament",
    )

    assert population.learning_method == "tournament"
    assert population.use_fermi is False


def test_learning_method_validation():
    with pytest.raises(ValueError, match="learning_method"):
        V2EvolutionaryPopulation(population_size=4, learning_method="unknown")


@pytest.mark.parametrize(
    ("method", "expected"),
    [("fermi", "fermi"), ("tournament", "tournament")],
)
def test_learning_method_dispatch(method, expected, monkeypatch):
    population = V2EvolutionaryPopulation(
        population_size=4, learning_method=method
    )
    calls = []
    monkeypatch.setattr(
        population,
        "_select_and_reproduce_fermi",
        lambda next_gen=None: calls.append(("fermi", next_gen)),
    )
    monkeypatch.setattr(
        population,
        "_select_and_reproduce",
        lambda next_gen=None: calls.append(("tournament", next_gen)),
    )

    population._select_and_reproduce_by_method(next_gen=7)

    assert calls == [(expected, 7)]


def test_result_config_records_new_and_legacy_fields():
    population = V2EvolutionaryPopulation(
        population_size=4, learning_method="tournament"
    )

    config = population._result_config(num_generations=2)

    assert config["learning_method"] == "tournament"
    assert config["use_fermi"] is False
