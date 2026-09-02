"""Focused pytest suite for the one-directional observe() interface.

The agent-type1 strategy interface changed from a two-sided observe:

    observe(donor_rep, donor_action, recipient_rep, recipient_action,
            my_rep) -> (new_donor_rep, new_recipient_rep)

to a ONE-DIRECTIONAL judge:

    observe(A_rep, A_action, B_rep, B_action, my_rep) -> new_A_rep

The framework calls observe() twice per joint action (once judging the
donor, once judging the recipient with roles swapped), so the LLM only
writes a single judging rule and never repeats a symmetric update.

These tests pin down:
  * executor.observe returns a single clamped float
  * observe_and_judge calls observe() twice and updates BOTH reputations
  * self-judgment uses the same one-directional call
  * the leading-eight baselines compile and judge one target only
  * clamping, NaN and exception safety
"""
from __future__ import annotations

import pytest

from experiments.v2_quantitative.agent import QuantitativeAgent
from experiments.v2_quantitative.baselines import (
    ALLC, ALLD, IS, IS_PLUS, SC, SH, SJ, SJ_PLUS, SS, SS_PLUS, get_baseline,
)
from experiments.v2_quantitative.executor import V2StrategyExecutor

ONE_WAY_CODE = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    # one-directional IS-like update for player A only
    if A_action == 'cooperate':
        new = A_rep + 0.5
    else:
        new = A_rep - 0.5
    return max(-1.0, min(1.0, new))

def decide(my_reputation, opponent_reputation):
    return my_reputation > 0.0
'''


@pytest.fixture
def executor():
    return V2StrategyExecutor(ONE_WAY_CODE)


def _agent(agent_id: int, code: str = ONE_WAY_CODE) -> QuantitativeAgent:
    return QuantitativeAgent(agent_id, code, executor=V2StrategyExecutor(code))


# ---------------------------------------------------------------- executor


def test_executor_observe_returns_single_float(executor):
    """One-directional observe returns a float, not a tuple."""
    out = executor.observe(0.2, "cooperate", 0.4, "defect", 0.1)
    assert isinstance(out, float)
    assert out == pytest.approx(0.7)


def test_executor_observe_clamps_to_unit_interval(executor):
    assert executor.observe(0.9, "cooperate", 0.0, "defect", 0.0) == pytest.approx(1.0)
    assert executor.observe(-0.9, "defect", 0.0, "cooperate", 0.0) == pytest.approx(-1.0)


def test_executor_observe_nan_becomes_zero():
    code = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    return float('nan')
def decide(my_reputation, opponent_reputation):
    return True
'''
    ex = V2StrategyExecutor(code)
    assert ex.observe(0.3, "cooperate", 0.3, "defect", 0.0) == 0.0


def test_executor_observe_raises_falls_back_to_input():
    code = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    raise RuntimeError("boom")
def decide(my_reputation, opponent_reputation):
    return True
'''
    ex = V2StrategyExecutor(code)
    assert ex.observe(0.3, "cooperate", 0.3, "defect", 0.0) == pytest.approx(0.3)


# ---------------------------------------------------- observe_and_judge


def test_observe_and_judge_updates_both_reputations():
    """One joint action -> observe() called twice, both players updated."""
    agent = _agent(0)
    agent.reputations = {0: 0.1, 1: 0.3, 2: -0.2}
    agent.observe_and_judge(
        donor_id=1, donor_action="cooperate",
        recipient_id=2, recipient_action="defect",
    )
    # donor cooperated: 0.3 + 0.5 = 0.8
    assert agent.reputations[1] == pytest.approx(0.8)
    # recipient defected: -0.2 - 0.5 = -0.7
    assert agent.reputations[2] == pytest.approx(-0.7)


def test_observe_and_judge_roles_are_swapped():
    """The second call judges the recipient with (A=recipient, B=donor)."""
    calls = []

    def spy(A_rep, A_action, B_rep, B_action, my_reputation):
        calls.append((A_rep, A_action, B_rep, B_action))
        return A_rep

    code = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    return A_rep
def decide(my_reputation, opponent_reputation):
    return True
'''
    agent = QuantitativeAgent(0, code, executor=V2StrategyExecutor(code))
    agent._executor._observe = spy
    agent.reputations = {0: 0.1, 1: 0.3, 2: -0.2}
    agent.observe_and_judge(
        donor_id=1, donor_action="cooperate",
        recipient_id=2, recipient_action="defect",
    )
    assert len(calls) == 2
    # first: judge donor (A=donor), second: judge recipient (A=recipient)
    assert calls[0] == (0.3, "cooperate", -0.2, "defect")
    assert calls[1] == (-0.2, "defect", 0.3, "cooperate")


def test_self_judge_uses_one_directional_observe():
    """Self-judgment judges self first (A=self), then the partner."""
    agent = _agent(1)
    agent.reputations = {1: 0.0, 2: 0.5}
    agent.self_judge(donor_action="cooperate", recipient_id=2, recipient_action="defect")
    # self cooperated: 0.0 + 0.5 = 0.5
    assert agent.reputations[1] == pytest.approx(0.5)
    # partner defected: 0.5 - 0.5 = 0.0
    assert agent.reputations[2] == pytest.approx(0.0)


def test_observe_and_judge_symmetric_interaction():
    """Both players cooperating raises both reputations symmetrically."""
    agent = _agent(0)
    agent.reputations = {0: 0.1, 1: 0.3, 2: 0.2}
    agent.observe_and_judge(
        donor_id=1, donor_action="cooperate",
        recipient_id=2, recipient_action="cooperate",
    )
    assert agent.reputations[1] == pytest.approx(0.8)
    assert agent.reputations[2] == pytest.approx(0.7)


# ------------------------------------------------------------ baselines


@pytest.mark.parametrize(
    "name, action, expected_step",
    [
        ("IS", "cooperate", 0.333),
        ("IS", "defect", -0.333),
        ("SC", "cooperate", 0.333),
        ("SC", "defect", -0.333),
        ("ALLC", "cooperate", 0.333),
        ("ALLD", "defect", -0.333),
    ],
)
def test_baseline_observe_is_one_directional(name, action, expected_step):
    """Baselines judge a single target A from A's own action only."""
    ex = V2StrategyExecutor(get_baseline(name))
    start = 0.2
    out = ex.observe(start, action, 0.5, "cooperate", 0.1)
    assert out == pytest.approx(max(-1.0, min(1.0, start + expected_step)), abs=1e-6)


def test_all_baselines_compile_and_return_float():
    """Every leading-eight baseline compiles and returns a single float."""
    for name in ("ALLC", "ALLD", "IS", "SS", "SJ", "SC", "SH", "IS+", "SS+", "SJ+"):
        ex = V2StrategyExecutor(get_baseline(name))
        out = ex.observe(0.2, "cooperate", 0.4, "defect", 0.1)
        assert isinstance(out, float)
        assert -1.0 <= out <= 1.0
