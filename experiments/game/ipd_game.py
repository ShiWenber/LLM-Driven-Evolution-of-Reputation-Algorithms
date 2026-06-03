"""Iterated Prisoner's Dilemma (IPD) — Willis 2025 baseline implementation.

Implements the two-player repeated Prisoner's Dilemma with full
information, used as a baseline in PAPER_DRAFT.md Experiment 5 to
contrast with the private-observation N-player donor game.

Payoff matrix (R, S, T, P):
    T (temptation) = 5  — defect vs cooperator
    R (reward)     = 3  — mutual cooperation
    P (punishment) = 1  — mutual defection
    S (sucker)     = 0  — cooperate vs defector

Strategy interface: this game does NOT use the CodeAgent wrapper.
Instead, IPDGame.play_match calls a *strategy function* directly with
`decide(my_history, round_num) -> bool` and tracks history itself.
This keeps the IPD path simple and avoids cross-pollination with the
donor-game reputation API.
"""
from __future__ import annotations

import random
import textwrap
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field


# IPD canonical payoffs (Axelrod tournament standard)
T_PAYOFF = 5
R_PAYOFF = 3
P_PAYOFF = 1
S_PAYOFF = 0


# A strategy callable type: decide(my_history, round_num) -> bool
IPDStrategy = Callable[[list, int], bool]


@dataclass
class IPDMatchResult:
    """Outcome of a single IPD match between two strategies."""
    cooperator_id: int
    defector_id: int
    rounds: int
    coop_actions: List[bool] = field(default_factory=list)
    payoff_a: float = 0.0
    payoff_b: float = 0.0

    @property
    def cooperation_rate(self) -> float:
        if not self.coop_actions:
            return 0.0
        return sum(self.coop_actions) / len(self.coop_actions)


class IPDGame:
    """Iterated Prisoner's Dilemma between two IPD strategy callables."""

    def __init__(self, num_rounds: int = 1000, noise: float = 0.0):
        self.num_rounds = num_rounds
        self.noise = noise

    def play_match(
        self,
        strategy_a: IPDStrategy,
        strategy_b: IPDStrategy,
        seed: int | None = None,
    ) -> IPDMatchResult:
        rng = random.Random(seed)
        history_a: list = []
        history_b: list = []
        coop_a: list = []
        coop_b: list = []
        payoff_a = 0.0
        payoff_b = 0.0

        for r in range(self.num_rounds):
            decision_a = bool(strategy_a(history_a, r))
            decision_b = bool(strategy_b(history_b, r))

            if self.noise > 0 and rng.random() < self.noise:
                decision_a = not decision_a
            if self.noise > 0 and rng.random() < self.noise:
                decision_b = not decision_b

            if decision_a and decision_b:
                payoff_a += R_PAYOFF
                payoff_b += R_PAYOFF
            elif decision_a and not decision_b:
                payoff_a += S_PAYOFF
                payoff_b += T_PAYOFF
            elif not decision_a and decision_b:
                payoff_a += T_PAYOFF
                payoff_b += S_PAYOFF
            else:
                payoff_a += P_PAYOFF
                payoff_b += P_PAYOFF

            action_a = "donate" if decision_a else "not_donate"
            action_b = "donate" if decision_b else "not_donate"
            history_a.append({
                "round": r, "role": "donor", "partner": 1,
                "action": action_a, "partner_action": action_b,
            })
            history_b.append({
                "round": r, "role": "donor", "partner": 0,
                "action": action_b, "partner_action": action_a,
            })
            coop_a.append(decision_a)
            coop_b.append(decision_b)

        return IPDMatchResult(
            cooperator_id=0,
            defector_id=1,
            rounds=self.num_rounds,
            coop_actions=coop_a,
            payoff_a=payoff_a,
            payoff_b=payoff_b,
        )

    def all_play_all(
        self,
        strategies: List[IPDStrategy],
        seed: int | None = None,
    ) -> Dict[str, Any]:
        n = len(strategies)
        total_payoff = np.zeros(n)
        matches_played = np.zeros(n)
        cooperation_counts: list = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                result = self.play_match(strategies[i], strategies[j], seed=seed)
                total_payoff[i] += result.payoff_a
                matches_played[i] += 1
                cooperation_counts[i].extend(result.coop_actions)

        per_agent = []
        for i in range(n):
            mean_payoff = float(total_payoff[i] / matches_played[i])
            coop_rate = (
                sum(cooperation_counts[i]) / len(cooperation_counts[i])
                if cooperation_counts[i] else 0.0
            )
            per_agent.append({
                "agent_id": i,
                "mean_payoff": mean_payoff,
                "cooperation_rate": coop_rate,
                "matches_played": int(matches_played[i]),
            })

        return {
            "num_agents": n,
            "rounds_per_match": self.num_rounds,
            "noise": self.noise,
            "per_agent": per_agent,
            "tournament_mean_cooperation": float(
                np.mean([a["cooperation_rate"] for a in per_agent])
            ),
            "tournament_mean_payoff": float(
                np.mean([a["mean_payoff"] for a in per_agent])
            ),
        }


def ipd_tournament_to_fitness(
    tournament_result: Dict[str, Any],
    num_agents: int,
) -> np.ndarray:
    payoffs = np.array([a["mean_payoff"] for a in tournament_result["per_agent"]])
    return payoffs - payoffs.min() + 1.0


def load_ipd_strategy_from_code(code: str) -> Optional[IPDStrategy]:
    """Compile LLM-generated Python code into an IPDStrategy callable.

    Supports two signatures for `decide`:
        (a) `decide(my_history, round_num) -> bool`  — pure IPD style
        (b) `decide(recipient_reputation, round_num, my_history) -> bool`
            — donor-game style (LLM templates use this; we ignore
            recipient_reputation in IPD)

    Returns None if compilation fails.
    """
    try:
        namespace: dict = {}
        full_src = code
        exec(full_src, namespace)
        if "decide" not in namespace:
            return None
        # Inspect signature
        import inspect
        sig = inspect.signature(namespace["decide"])
        params = list(sig.parameters.keys())

        if len(params) == 2 and "my_history" in params:
            # Pure IPD signature: decide(my_history, round_num)
            def strategy(my_history, round_num):
                return bool(namespace["decide"](my_history, round_num))
        elif len(params) == 3 and "my_history" in params:
            # Donor-game signature: decide(recipient_reputation, round_num, my_history)
            def strategy(my_history, round_num):
                return bool(namespace["decide"](0.0, round_num, my_history))
        else:
            return None
        return strategy
    except Exception:
        return None


def load_ipd_strategies_from_code(code_strings: List[str]) -> List[Optional[IPDStrategy]]:
    """Compile a list of code strings; None for un-compilable code."""
    return [load_ipd_strategy_from_code(c) for c in code_strings]


# Pre-built classical strategies for baseline tournaments
def tit_for_tat(history, round_num):
    if not history:
        return True
    return history[-1].get("partner_action") == "donate"


def always_cooperate(history, round_num):
    return True


def always_defect(history, round_num):
    return False


def pavlov(history, round_num):
    """Win-Stay, Lose-Shift: cooperate if last round was (C,*), defect if (D,*)."""
    if not history:
        return True
    last = history[-1]
    if last["action"] == "donate":
        return True  # stayed
    return False  # shifted


def grim_trigger(history, round_num):
    if any(h["partner_action"] == "not_donate" for h in history):
        return False
    return True


CLASSICAL_STRATEGIES = {
    "tit_for_tat": tit_for_tat,
    "always_cooperate": always_cooperate,
    "always_defect": always_defect,
    "pavlov": pavlov,
    "grim_trigger": grim_trigger,
}
