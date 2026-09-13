"""Tests for the static contract validation of LLM-generated strategy code.

Covers both contracts:

* validate_v2_contract: observe()/decide() top-level functions.
* validate_v3_contract: LLMAgent class with __init__/decide/observe.

Also guards that contract validation never rejects valid baseline code
(regression guard for the V2 executor path).
"""
from __future__ import annotations

import pytest

from experiments.v2_quantitative.baselines import (
    ALLC, ALLD, IS, SC, SH, SJ, SS, get_baseline,
)
from experiments.v2_quantitative.code_contract import (
    CodeContractError,
    validate_v2_contract,
    validate_v3_contract,
)
from experiments.v2_quantitative.executor import V2StrategyExecutor

# A minimal, valid V2 strategy.
VALID_V2 = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    return A_rep

def decide(my_reputation, opponent_reputation):
    return True
'''

# A minimal, valid V3 LLMAgent class.
VALID_V3 = '''
class LLMAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def decide(self) -> bool:
        return True

    def observe(self, donor_id, donor_action, recipient_id, recipient_action) -> None:
        return None
'''


# --------------------------------------------------------------- v2 contract


def test_v2_valid_code_passes():
    validate_v2_contract(VALID_V2)


def test_v2_missing_observe_fails():
    code = '''
def decide(my_reputation, opponent_reputation):
    return True
'''
    with pytest.raises(CodeContractError, match="must define observe"):
        validate_v2_contract(code)


def test_v2_missing_decide_fails():
    code = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation):
    return A_rep
'''
    with pytest.raises(CodeContractError, match="must define decide"):
        validate_v2_contract(code)


def test_v2_observe_missing_param_fails():
    code = '''
def observe(A_rep, A_action, my_reputation):
    return A_rep

def decide(my_reputation, opponent_reputation):
    return True
'''
    with pytest.raises(CodeContractError, match="observe\\(\\) missing parameter"):
        validate_v2_contract(code)


def test_v2_syntax_error_fails():
    code = "def observe(A_rep, :\n    return A_rep"
    with pytest.raises(CodeContractError, match="syntax error"):
        validate_v2_contract(code)


def test_v2_too_long_fails():
    code = "def observe(A_rep, A_action, B_rep, B_action, my_reputation):\n"
    code += "    return A_rep\n" * 400  # far exceeds 3000 chars
    code += "def decide(my_reputation, opponent_reputation):\n    return True\n"
    with pytest.raises(CodeContractError, match="too long"):
        validate_v2_contract(code)


def test_v2_nested_functions_do_not_satisfy_the_top_level_contract():
    code = '''
def wrapper():
    def observe(A_rep, A_action, B_rep, B_action, my_reputation):
        return A_rep
    def decide(my_reputation, opponent_reputation):
        return True
'''
    with pytest.raises(CodeContractError, match="top-level observe"):
        validate_v2_contract(code)


def test_v2_rejects_parameter_order_that_breaks_positional_calls():
    code = '''
def observe(A_action, A_rep, B_rep, B_action, my_reputation):
    return A_rep

def decide(opponent_reputation, my_reputation):
    return True
'''
    with pytest.raises(CodeContractError, match="parameter order"):
        validate_v2_contract(code)


def test_v2_rejects_additional_required_parameters():
    code = '''
def observe(A_rep, A_action, B_rep, B_action, my_reputation, context):
    return A_rep

def decide(my_reputation, opponent_reputation):
    return True
'''
    with pytest.raises(CodeContractError, match="additional required"):
        validate_v2_contract(code)


# --------------------------------------------------------------- v3 contract


def test_v3_valid_code_passes():
    validate_v3_contract(VALID_V3)


def test_v3_missing_class_fails():
    code = "def decide():\n    return True\n"
    with pytest.raises(CodeContractError, match="class LLMAgent"):
        validate_v3_contract(code)


def test_v3_missing_decide_fails():
    code = '''
class LLMAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
'''
    with pytest.raises(CodeContractError, match="must define decide"):
        validate_v3_contract(code)


def test_v3_observe_missing_param_fails():
    code = '''
class LLMAgent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def decide(self) -> bool:
        return True

    def observe(self, donor_id, donor_action) -> None:
        return None
'''
    with pytest.raises(CodeContractError, match="observe\\(\\) missing parameter"):
        validate_v3_contract(code)


def test_v3_nested_agent_class_does_not_satisfy_the_contract():
    code = '''
def factory():
    class LLMAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
        def decide(self):
            return True
        def observe(self, donor_id, donor_action, recipient_id, recipient_action):
            return None
    return LLMAgent
'''
    with pytest.raises(CodeContractError, match="top-level class LLMAgent"):
        validate_v3_contract(code)


def test_v3_constructor_requires_self_before_agent_id():
    code = '''
class LLMAgent:
    def __init__(agent_id):
        pass
    def decide(self):
        return True
    def observe(self, donor_id, donor_action, recipient_id, recipient_action):
        return None
'''
    with pytest.raises(CodeContractError, match="self|parameter order"):
        validate_v3_contract(code)


# ----------------------------------------------------------- executor wiring


def test_executor_rejects_missing_observe():
    """V2StrategyExecutor rejects invalid code on construction."""
    code = '''
def decide(my_reputation, opponent_reputation):
    return True
'''
    # Match the message the executor actually raises. An earlier revision of
    # this test expected "invalid v2 strategy", which appears nowhere in the
    # source; the behaviour under test (refuse to construct) was already right.
    with pytest.raises(ValueError, match="must define both observe"):
        V2StrategyExecutor(code)


@pytest.mark.parametrize("name", ["ALLC", "ALLD", "IS", "SS", "SJ", "SC", "SH"])
def test_all_baselines_still_compile(name):
    """Contract validation never rejects a valid baseline."""
    ex = V2StrategyExecutor(get_baseline(name))
    assert ex.observe(0.2, "cooperate", 0.4, "defect", 0.1) is not None
