"""Donor Game engine for the v2 quantitative interface.

Public reputation store (single dict shared by all agents) + each agent's
own private self-rating + private ratings of others.

Observability levels (same as v1):
  - "private": agents observe only their own interactions
  - "partial_X": agents observe a random fraction X of third-party interactions
  - "full": agents observe every third-party interaction

Per interaction, the framework calls:
  1. donor's `decide()` -> True/False
  2. donor's `self_judge()` -> updates donor's self_reputation
  3. recipient's `self_judge()` -> updates recipient's self_reputation
  4. for each observer X (per observability rules):
       X.observe_and_judge(donor=donor, ...) -> updates X.reputations[donor]
"""
from __future__ import annotations
import random
from typing import Dict, List, Tuple
from .agent import QuantitativeAgent
from .executor import V2StrategyExecutor


class V2DonorGame:
    def __init__(
        self,
        population_size: int,
        benefit: float = 2.0,
        cost: float = 1.0,
        observability: str = "full",
        observability_p: float = 1.0,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.benefit = benefit
        self.cost = cost
        self.observability = observability
        self.observability_p = observability_p
        self.seed = seed
        self.rng = random.Random(seed)
        self.agents: List[QuantitativeAgent] = []
        self.round_num = 0
        # Global log of every interaction in the current generation
        self._global_log: List[Dict] = []
        # Payoffs
        self.payoffs = [0.0] * population_size

    def setup_population(self, agents: List[QuantitativeAgent]):
        self.agents = agents
        # NOTE: agent_id is a STABLE global identity (assigned monotonically
        # by V2EvolutionaryPopulation._new_agent). We do NOT reassign it to
        # list positions. List index in self.agents is just iteration order.
        # Build a lookup so play_round can resolve agent_id -> agent object
        # without scanning the list each time.
        self._agent_by_id = {a.agent_id: a for a in self.agents}

    def play_round(self) -> Dict:
        """Play one round: each agent acts as donor once with random recipient."""
        self.round_num += 1
        round_log = []
        # Each agent (by stable agent_id) in shuffled order takes a turn
        agent_ids = [a.agent_id for a in self.agents]
        donor_order = list(agent_ids)
        self.rng.shuffle(donor_order)
        for donor_id in donor_order:
            # Choose random recipient (by agent_id, not list position)
            recipient_id = agent_ids[self.rng.randrange(self.population_size)]
            while recipient_id == donor_id:
                recipient_id = agent_ids[self.rng.randrange(self.population_size)]
            donor = self._agent_by_id[donor_id]
            recipient = self._agent_by_id[recipient_id]
            action = donor.choose(recipient_id, round_num=self.round_num)
            donor.record_donation(recipient_id, action, self.round_num)
            # Payoffs (use list position to index the payoffs array)
            donor_pos = self.agents.index(donor)
            recipient_pos = self.agents.index(recipient)
            if action:
                self.payoffs[donor_pos] -= self.cost
                self.payoffs[recipient_pos] += self.benefit
            # Store action as STRING ('cooperate' / 'defect') so that the
            # strategy code can pattern-match on it in evaluate().
            action_str = "cooperate" if action else "defect"
            recipient_action = action_str
            interaction = {
                "round": self.round_num,
                "donor": donor_id,
                "recipient": recipient_id,
                "donor_action": action_str,
                "recipient_action": recipient_action,
            }
            self._global_log.append(interaction)
            round_log.append(interaction)
        return {"round": self.round_num, "interactions": round_log}

    def distribute_observations_and_self_judgments(self):
        """After each round: distribute observations and update self-ratings.

        Order of operations:
          1. donor.self_judge()  (donor updates their own self-rating)
          2. for each observer (per observability rules), call
             observer.observe_and_judge(donor, ...) which updates
             observer's private rating of the donor.
        """
        recent = self._global_log[-self.population_size:]
        # Step 1: self-judgments. Only the DONOR took an action so only the
        # donor's self_reputation is updated here. The recipient's self-
        # reputation will update in a future round when they themselves act.
        for inter in recent:
            donor_id = inter["donor"]
            recipient_id = inter["recipient"]
            self._agent_by_id[donor_id].self_judge(
                donor_action=inter["donor_action"],
                recipient_id=recipient_id,
                recipient_action=inter["recipient_action"],
            )
        # Step 2: distribute third-party observations per observability rules
        if self.observability == "private":
            return
        all_agent_ids = [a.agent_id for a in self.agents]
        for inter in recent:
            donor_id = inter["donor"]
            recipient_id = inter["recipient"]
            donor_action = inter["donor_action"]
            recipient_action = inter["recipient_action"]
            # All other agents (by stable agent_id) are potential observers
            for obs_id in all_agent_ids:
                if obs_id == donor_id or obs_id == recipient_id:
                    continue
                if self.observability == "full":
                    self._agent_by_id[obs_id].observe_and_judge(
                        donor_id=donor_id,
                        donor_action=donor_action,
                        recipient_id=recipient_id,
                        recipient_action=recipient_action,
                    )
                elif self.observability.startswith("partial"):
                    if self.rng.random() < self.observability_p:
                        self._agent_by_id[obs_id].observe_and_judge(
                            donor_id=donor_id,
                            donor_action=donor_action,
                            recipient_id=recipient_id,
                            recipient_action=recipient_action,
                        )

    def run_generation(self) -> Dict:
        """Run a full generation: T rounds, then return aggregate stats."""
        self.round_num = 0
        self.payoffs = [0.0] * self.population_size
        self._global_log = []
        for _ in range(self.population_size):  # T = N (one round per donor)
            self.play_round()
            self.distribute_observations_and_self_judgments()
        # Stats
        coop_count = sum(1 for inter in self._global_log if inter["donor_action"] == "cooperate")
        coop_rate = coop_count / max(1, len(self._global_log))
        return {
            "cooperation_rate_mean": coop_rate,
            "n_interactions": len(self._global_log),
            "round_num": self.population_size,
            "payoffs": list(self.payoffs),
        }
