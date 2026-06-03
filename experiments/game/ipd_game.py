"""Iterated Prisoner's Dilemma (IPD) — Willis 2025 baseline implementation.

Implements the two-player repeated Prisoner's Dilemma with full
information, used as a baseline in PAPER_DRAFT.md Experiment 5 to
contrast with the private-observation N-player donor game.

Payoff matrix (R, S, T, P):
    T (temptation) = 5  — defect vs cooperator
    R (reward)     = 3  — mutual cooperation
    P (punishment) = 1  — mutual defection
    S (sucker)     = 0  — cooperate vs defector

Strategy interface matches donor_game: each agent runs the same
`evaluate()` / `decide()` pair, but in the IPD these are repurposed:
    - `evaluate()` is unused (no third-party observation in two-player game)
    - `decide()` takes recipient_reputation but in practice only history matters
    - Strategies consume my_history to decide C/D each round
"""
from __future__ import annotations

import random
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field


# IPD canonical payoffs (Axelrod tournament standard)
T_PAYOFF = 5
R_PAYOFF = 3
P_PAYOFF = 1
S_PAYOFF = 0


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
    """Iterated Prisoner's Dilemma between two LLM-coded strategies.

    Each round both agents simultaneously decide C or D based on
    opponent's full action history (full information).
    """

    def __init__(self, num_rounds: int = 1000, noise: float = 0.0):
        self.num_rounds = num_rounds
        self.noise = noise  # Probability that an agent's action is flipped

    def play_match(
        self,
        agent_a,
        agent_b,
        seed: int | None = None,
    ) -> IPDMatchResult:
        """Play an IPD match between two CodeAgent-like objects.

        Each round both agents call decide(recipient_reputation, round, history).
        In IPD we treat recipient_reputation as a constant 0.0 placeholder
        and rely on my_history for decision logic.
        """
        rng = random.Random(seed)
        history_a: list[dict] = []
        history_b: list[dict] = []
        coop_a: list[bool] = []
        coop_b: list[bool] = []
        payoff_a = 0.0
        payoff_b = 0.0

        for r in range(self.num_rounds):
            # Both agents decide simultaneously
            decision_a = agent_a.decide(
                recipient_reputation=0.0,
                round_num=r,
                my_history=history_a,
            )
            decision_b = agent_b.decide(
                recipient_reputation=0.0,
                round_num=r,
                my_history=history_b,
            )

            # Apply noise: each agent's action flipped independently
            if self.noise > 0 and rng.random() < self.noise:
                decision_a = not decision_a
            if self.noise > 0 and rng.random() < self.noise:
                decision_b = not decision_b

            # Compute payoff for this round
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

            # Record histories
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
        agents: list,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        """Run an all-play-all IPD tournament.

        Each agent plays against every other agent (and itself).
        Returns aggregated per-agent statistics.
        """
        n = len(agents)
        total_payoff = np.zeros(n)
        matches_played = np.zeros(n)
        cooperation_counts: list[list[bool]] = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                result = self.play_match(agents[i], agents[j], seed=seed)
                total_payoff[i] += result.payoff_a
                matches_played[i] += 1
                # Track i's cooperation rate across the tournament
                cooperation_counts[i].extend(result.coop_actions)

        # Per-agent statistics
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
    """Convert tournament per-agent payoff into fitness for selection.

    Fitness = mean_payoff - min_payoff + 1, so all are positive and
    relative differences are preserved (Moran-style).
    """
    payoffs = np.array([a["mean_payoff"] for a in tournament_result["per_agent"]])
    return payoffs - payoffs.min() + 1.0
