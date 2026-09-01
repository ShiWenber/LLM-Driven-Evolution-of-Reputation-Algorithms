"""Lineage-tree builder regression tests.

Synthetic lineage data (4 agents, 4 generations, Fermi run) mirrors the
historical ``--selftest`` fixture that used to live in
``experiments/analysis/lineage/build.py``. These tests are the regression
gate for ``build_lineage_tree``.
"""
import pytest

from experiments.analysis.lineage.build import build_lineage_tree


def _synthetic_data() -> dict:
    """4-agent, 4-generation Fermi run."""
    events = [
        {"lineage_id": 0, "parent_lineage_id": None, "parent_id": None, "origin": "initial", "birth_gen": 0},
        {"lineage_id": 1, "parent_lineage_id": None, "parent_id": None, "origin": "initial", "birth_gen": 0},
        {"lineage_id": 2, "parent_lineage_id": None, "parent_id": None, "origin": "initial", "birth_gen": 0},
        {"lineage_id": 3, "parent_lineage_id": None, "parent_id": None, "origin": "initial", "birth_gen": 0},
        {"lineage_id": 4, "parent_lineage_id": 3, "parent_id": 3, "origin": "imitate", "birth_gen": 1},
        {"lineage_id": 5, "parent_lineage_id": 3, "parent_id": 3, "origin": "imitate", "birth_gen": 1},
        {"lineage_id": 6, "parent_lineage_id": None, "parent_id": None, "origin": "independent_init", "birth_gen": 2},
        {"lineage_id": 7, "parent_lineage_id": 4, "parent_id": 0, "origin": "imitate", "birth_gen": 2},
        {"lineage_id": 8, "parent_lineage_id": 7, "parent_id": 3, "origin": "imitate", "birth_gen": 3},
        {"lineage_id": 9, "parent_lineage_id": None, "parent_id": None, "origin": "independent_init", "birth_gen": 3},
    ]
    trajectory = [
        {"generation": 0, "population": [
            {"agent_id": 0, "lineage_id": 0}, {"agent_id": 1, "lineage_id": 1},
            {"agent_id": 2, "lineage_id": 2}, {"agent_id": 3, "lineage_id": 3}]},
        {"generation": 1, "population": [
            {"agent_id": 0, "lineage_id": 4}, {"agent_id": 1, "lineage_id": 5},
            {"agent_id": 2, "lineage_id": 2}, {"agent_id": 3, "lineage_id": 3}]},
        {"generation": 2, "population": [
            {"agent_id": 0, "lineage_id": 4}, {"agent_id": 1, "lineage_id": 6},
            {"agent_id": 2, "lineage_id": 7}, {"agent_id": 3, "lineage_id": 3}]},
        {"generation": 3, "population": [
            {"agent_id": 0, "lineage_id": 4}, {"agent_id": 1, "lineage_id": 9},
            {"agent_id": 2, "lineage_id": 8}, {"agent_id": 3, "lineage_id": 7}]},
    ]
    final_population = [
        {"agent_id": 0, "lineage_id": 4}, {"agent_id": 1, "lineage_id": 9},
        {"agent_id": 2, "lineage_id": 8}, {"agent_id": 3, "lineage_id": 7},
    ]
    return {"lineage_events": events, "trajectory": trajectory, "final_population": final_population}


@pytest.fixture(scope="module")
def synthetic_tree():
    return build_lineage_tree(_synthetic_data())


def test_synthetic_tree_lineage_count(synthetic_tree):
    # roots: 0, 1, 2, 3, 6, 9
    assert synthetic_tree["n_lineages"] == 6


def test_root_lineage_members_and_lifetime(synthetic_tree):
    root3 = synthetic_tree["lineages"]["3"]
    assert set(root3["members"]) == {3, 4, 5, 7, 8}
    assert root3["birth_gen"] == 0 and root3["death_gen"] == 3


def test_survivor_ancestry(synthetic_tree):
    surv = {s["agent_id"]: s for s in synthetic_tree["survivors"]}
    assert surv[2]["root_lineage_id"] == 3  # lineage 8 -> root 3
    assert surv[1]["root_lineage_id"] == 9  # independent init
