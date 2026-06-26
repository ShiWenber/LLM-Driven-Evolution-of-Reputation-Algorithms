"""Donor Game with indirect reciprocity and observability control.

Agents interact in rounds. Each agent acts as donor once per round.
Observability controls what interactions each agent witnesses, enabling
the study of private (decentralized) vs public reputation.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional

from ..agents.code_agent import CodeAgent


class DonorGame:
    """
    Iterated Donor Game with configurable observability.

    Observability conditions:
    - "private": agents see only their own interactions
    - "partial_X": agents randomly observe fraction X of other interactions
    - "full": agents observe all interactions (central reputation limit)
    """

    def __init__(
        self,
        population_size: int = 20,
        benefit: int = 2,
        cost: int = 1,
        num_rounds: int = 30,
        observability: str = "full",
        observability_p: float = 0.3,
        recent_window: int = 0,
        reputation_noise: float = 0.0,
    ):
        self.population_size = population_size
        self.benefit = benefit
        self.cost = cost
        self.num_rounds = num_rounds
        self.observability = observability
        self.observability_p = observability_p
        self.recent_window = recent_window  # 0 = off; >0 = inject last N actions in observation
        self.reputation_noise = reputation_noise  # 0 = off; >0 = symmetric noise on observed reputation

        # Game state
        self.agents: List[CodeAgent] = []
        self.round_num = 0
        self.history: List[Dict[str, Any]] = []
        self.payoffs = np.zeros(population_size)

        # Global interaction log (used for observation distribution)
        self._global_log: List[Dict[str, Any]] = []
        # Recent-actions ring buffer per agent (for recent_window feature)
        self._recent_actions: Dict[int, List[Dict[str, Any]]] = {}

    def setup_population(self, agents: List[CodeAgent]):
        """Set up the agent population."""
        if len(agents) != self.population_size:
            raise ValueError(
                f"Expected {self.population_size} agents, got {len(agents)}"
            )
        self.agents = agents
        self.payoffs = np.zeros(self.population_size)
        self.history = []
        self._global_log = []
        self.round_num = 0

        # Reset agents
        for agent in self.agents:
            pass  # Agent state reset handled in population.reset_for_generation()

    def play_round(self) -> Dict[str, Any]:
        """Play one round of the donor game."""
        round_data = {
            "round": self.round_num,
            "interactions": []
        }

        # Shuffle donor order
        donors = list(range(self.population_size))
        random.shuffle(donors)

        for donor_id in donors:
            # Choose random recipient (not self)
            possible = [i for i in range(self.population_size) if i != donor_id]
            recipient_id = random.choice(possible)

            agent = self.agents[donor_id]

            # Agent decides
            decision = agent.decide(
                recipient_id=recipient_id,
                round_num=self.round_num,
                population_size=self.population_size
            )

            # Execute decision
            if decision:
                self.payoffs[donor_id] -= self.cost
                self.payoffs[recipient_id] += self.benefit

            # Record for agents. Use explicit semantic action labels
            # ("cooperate" / "defect") so that the LLM has direct access
            # to the game's underlying semantics — no neutral-token indirection.
            action = "cooperate" if decision else "defect"
            recipient_action = None
            # Find if recipient was donor in same round (for my_history)
            for prev_interaction in round_data["interactions"]:
                if prev_interaction["donor"] == recipient_id:
                    recipient_action = prev_interaction["action"]
                    break

            agent.record_interaction(
                round_num=self.round_num,
                role="donor",
                partner_id=recipient_id,
                action=action,
                partner_action=recipient_action
            )

            # Recipient records too
            recipient_agent = self.agents[recipient_id]
            donor_action_for_recipient = action
            recipient_agent.record_interaction(
                round_num=self.round_num,
                role="recipient",
                partner_id=donor_id,
                action=recipient_action or "defect",
                partner_action=donor_action_for_recipient
            )

            # Log interaction globally (use observation-compatible keys)
            interaction = {
                "donor": donor_id,
                "recipient": recipient_id,
                "action": action,
                "round": self.round_num
            }
            round_data["interactions"].append(interaction)
            self._global_log.append(interaction)

            # Maintain per-observer recent-actions ring buffer (for recent_window feature)
            if self.recent_window > 0:
                obs_view = {"donor": donor_id, "action": action, "round": self.round_num}
                for observer in self.agents:
                    if observer.agent_id == donor_id or observer.agent_id == recipient_id:
                        continue  # Skip self/recipient — they're in their own my_history
                    buf = self._recent_actions.setdefault(observer.agent_id, [])
                    buf.append(obs_view)
                    if len(buf) > self.recent_window * 2:  # 2x cap for safety
                        buf = buf[-self.recent_window:]

        self.round_num += 1
        self.history.append(round_data)
        return round_data

    def _distribute_observations(self):
        """After each round, distribute observations to agents based on condition."""
        if self.observability == "private":
            # No observations — agents only know what they participated in
            return

        round_interactions = self._global_log[
            -(self.population_size):  # Last N interactions (this round)
        ]

        for agent in self.agents:
            for interaction in round_interactions:
                donor_id = interaction["donor"]

                # Skip own interactions (already in my_history)
                if donor_id == agent.agent_id:
                    continue
                if interaction["recipient"] == agent.agent_id:
                    continue

                # Inject recent-actions window (if feature enabled)
                obs_to_send = dict(interaction)  # shallow copy
                if self.recent_window > 0:
                    recent = self._recent_actions.get(agent.agent_id, [])
                    obs_to_send["recent_window"] = list(recent[-self.recent_window:])

                # Inject symmetric reputation noise (if feature enabled)
                if self.reputation_noise > 0:
                    noise = random.uniform(-self.reputation_noise, self.reputation_noise)
                    obs_to_send["_reputation_noise"] = noise  # for downstream, not used yet

                # Determine if agent observes this interaction
                if self.observability == "full":
                    agent.observe(
                        donor_id=donor_id,
                        observation=obs_to_send,
                        round_num=interaction["round"]
                    )
                elif self.observability.startswith("partial"):
                    if random.random() < self.observability_p:
                        agent.observe(
                            donor_id=donor_id,
                            observation=obs_to_send,
                            round_num=interaction["round"]
                        )

    def run_simulation(self) -> Dict[str, Any]:
        """Run full simulation and return results."""
        for _ in range(self.num_rounds):
            self.play_round()
            self._distribute_observations()

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        agent_states = []
        cooperation_rates = []
        final_payoffs = []

        for agent in self.agents:
            cr = agent.cooperation_rate
            donations_given = agent.total_donations
            donations_received = 0  # tracked via payoff
            total_rounds = agent.total_decisions

            cooperation_rates.append(cr)
            final_payoffs.append(float(self.payoffs[agent.agent_id]))

            agent_states.append({
                "agent_id": agent.agent_id,
                "cooperation_rate": cr,
                "donations_given": donations_given,
                "donations_received": donations_received,
                "total_rounds": total_rounds,
                "payoff": float(self.payoffs[agent.agent_id])
            })

        return {
            "population_size": self.population_size,
            "num_rounds": self.num_rounds,
            "benefit": self.benefit,
            "cost": self.cost,
            "observability": self.observability,
            "cooperation_rate_mean": float(np.mean(cooperation_rates)),
            "cooperation_rate_std": float(np.std(cooperation_rates)),
            "payoff_mean": float(np.mean(final_payoffs)),
            "payoff_std": float(np.std(final_payoffs)),
            "agent_states": agent_states,
            # "history": self.history,  # optionally save for debug
        }

    def __repr__(self):
        return (
            f"DonorGame(N={self.population_size}, rounds={self.num_rounds}, "
            f"b/c={self.benefit}/{self.cost}, obs={self.observability})"
        )
