"""2-player Prisoner's Dilemma game engine for the v2 quantitative interface.

Game model:
  - N agents (15 in the default config), each is its own instance with a
    private reputation matrix `reputations: dict[int, float]` keyed by
    agent_id, with `reputations[agent_id]` being the self-rating.
  - Each round: form a random matching of the N agents into pairs (one
    agent sits out if N is odd).
  - In each pair, BOTH players simultaneously choose C (cooperate) or
    D (defect). Payoffs (benefit=2, cost=1):
      (C, C) -> each +1
      (C, D) -> C gets -1, D gets +2
      (D, C) -> symmetric
      (D, D) -> each 0
  - After all pairs decide, distribute observations per observability
    rules (full / partial / private). For each observed joint action,
    the framework calls observer.observe_and_judge(donor_id, donor_action,
    recipient_id, recipient_action) on the observer. The agent's
    observe_and_judge internally calls its observe() function once
    with both players in the joint action.

Backward-compatible with the type1 QuantitativeAgent interface
(choose / observe_and_judge / self_judge / record_donation / etc.).
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple
from .agent import QuantitativeAgent
from .executor import V2StrategyExecutor


class V2DonorGame:
    """2-player simultaneous-PD game with reputation tracking.

    Note: despite the historical name "V2DonorGame" (kept for backward
    compat with existing imports), the underlying game is a 2-player
    symmetric prisoner's dilemma, not a donor game.
    """

    def __init__(
        self,
        population_size: int,
        benefit: float = 2.0,
        cost: float = 1.0,
        observability: str = "full",
        observability_p: float = 1.0,
        seed: int = 42,
        fitness_window_interactions: Optional[int] = 200,
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
        # Global log of every joint action in the current generation
        self._global_log: List[Dict] = []
        # Payoffs (indexed by list position)
        self.payoffs = [0.0] * population_size
        # Per-interaction payoff deltas (one entry per joint action played
        # in the current generation). Used to compute windowed fitness
        # (= sum of payoffs over only the last N interactions, treating
        # the first M as burn-in). Stored as a list of length-15 lists.
        self._interaction_deltas: List[List[float]] = []
        # Fitness window size: how many of the latest interactions to
        # count toward the agent's fitness for selection. If None or
        # <= 0 or larger than the total interaction count, all
        # interactions count (legacy behavior).
        self.fitness_window_interactions = fitness_window_interactions

    def setup_population(self, agents: List[QuantitativeAgent]):
        self.agents = agents
        # agent_id is a STABLE global identity (assigned monotonically by
        # the population manager). List index is just iteration order.
        # Build a lookup so play_round can resolve agent_id -> agent object
        # without scanning the list each time.
        self._agent_by_id = {a.agent_id: a for a in self.agents}

    def _form_pairs(self) -> List[Tuple[int, int]]:
        """Randomly partition agents into pairs. If odd, one sits out."""
        agent_ids = [a.agent_id for a in self.agents]
        self.rng.shuffle(agent_ids)
        pairs = []
        for i in range(0, len(agent_ids) - 1, 2):
            pairs.append((agent_ids[i], agent_ids[i + 1]))
        return pairs

    def play_round(self) -> Dict:
        """Play one round: form random pairs, each pair plays a simultaneous PD."""
        self.round_num += 1
        round_log = []
        pairs = self._form_pairs()
        for donor_id, recipient_id in pairs:
            donor = self._agent_by_id[donor_id]
            recipient = self._agent_by_id[recipient_id]
            # Both players choose simultaneously. We call choose() in
            # sequence but both decisions are based on each player's
            # own (my_rep, opp_rep) at the start of the round — no
            # information leaks between the two calls.
            action1 = donor.choose(recipient_id, round_num=self.round_num)
            action2 = recipient.choose(donor_id, round_num=self.round_num)
            donor.record_donation(recipient_id, action1, self.round_num)
            recipient.record_donation(donor_id, action2, self.round_num)
            # Payoffs (use list position to index the payoffs array)
            donor_pos = self.agents.index(donor)
            recipient_pos = self.agents.index(recipient)
            # Per-interaction deltas (for windowed fitness / burn-in).
            # Initialize zeros for all agents; only the two players
            # in this pair have nonzero entries.
            pair_delta = [0.0] * self.population_size
            # Cost: each cooperator pays cost
            if action1:
                self.payoffs[donor_pos] -= self.cost
                pair_delta[donor_pos] -= self.cost
            if action2:
                self.payoffs[recipient_pos] -= self.cost
                pair_delta[recipient_pos] -= self.cost
            # Benefit: each cooperator gives benefit to the other
            if action1:
                self.payoffs[recipient_pos] += self.benefit
                pair_delta[recipient_pos] += self.benefit
            if action2:
                self.payoffs[donor_pos] += self.benefit
                pair_delta[donor_pos] += self.benefit
            # Record this interaction's per-agent payoff deltas for the
            # windowed-fitness computation (the first
            # `total - window` interactions count as burn-in).
            self._interaction_deltas.append(pair_delta)
            # Store actions as STRING so the strategy code can
            # pattern-match on them in observe().
            donor_action_str = "cooperate" if action1 else "defect"
            recipient_action_str = "cooperate" if action2 else "defect"
            interaction = {
                "round": self.round_num,
                "donor": donor_id,
                "recipient": recipient_id,
                "donor_action": donor_action_str,
                "recipient_action": recipient_action_str,
            }
            self._global_log.append(interaction)
            round_log.append(interaction)
        return {"round": self.round_num, "interactions": round_log}

    def distribute_observations_and_self_judgments(self):
        """After each round: distribute observations and self-judgments.

        For each joint action in the round:
          - donor observes (self-judgment) via self_judge
          - recipient observes (self-judgment) via self_judge
          - for each third-party observer (per observability rules),
            call observer.observe_and_judge(...)
        """
        recent = self._global_log[-max(1, len(self._global_log)):]
        recent = [i for i in recent if i["round"] == self.round_num]
        # Step 1: self-judgments for BOTH players in each pair
        for inter in recent:
            donor_id = inter["donor"]
            recipient_id = inter["recipient"]
            donor_action = inter["donor_action"]
            recipient_action = inter["recipient_action"]
            # Donor's self-judgment
            self._agent_by_id[donor_id].self_judge(
                donor_action=donor_action,
                recipient_id=recipient_id,
                recipient_action=recipient_action,
            )
            # Recipient's self-judgment (in PD, recipient also acts)
            self._agent_by_id[recipient_id].self_judge(
                donor_action=recipient_action,
                recipient_id=donor_id,
                recipient_action=donor_action,
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
                    continue  # already self-judged
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
        """Run a full generation: T rounds, then return aggregate stats.

        With population_size=16 and T=16 (T = N), each round we form
        8 pairs and nobody sits out. Total of 8*16 = 128 joint actions
        per generation; each player participates in all 16 rounds.
        """
        self.round_num = 0
        self.payoffs = [0.0] * self.population_size
        self._global_log = []
        self._interaction_deltas = []
        for _ in range(self.population_size):  # T = N (one round per donor, by default)
            self.play_round()
            self.distribute_observations_and_self_judgments()
        # Stats
        coop_count = sum(
            1
            for inter in self._global_log
            if inter["donor_action"] == "cooperate"
        )
        coop_count += sum(
            1
            for inter in self._global_log
            if inter["recipient_action"] == "cooperate"
        )
        coop_rate = coop_count / max(1, 2 * len(self._global_log))
        return {
            "cooperation_rate_mean": coop_rate,
            "n_interactions": len(self._global_log),
            "round_num": self.population_size,
            "payoffs": list(self.payoffs),
        }

    def get_windowed_fitness(self) -> List[float]:
        """Return per-agent fitness summed over only the LAST
        `fitness_window_interactions` interactions.

        The first `total - window` interactions are treated as
        burn-in: they were played (so strategies got experience via
        `observe()` and reputations evolved), but their payoffs do
        not count toward the fitness used for selection.

        If `fitness_window_interactions` is None, <= 0, or larger
        than the total interaction count, all interactions count
        (legacy behavior, equivalent to `self.payoffs`).
        """
        window = self.fitness_window_interactions
        deltas = self._interaction_deltas
        n_total = len(deltas)
        if window is None or window <= 0 or window >= n_total:
            return list(self.payoffs)
        # Sum the deltas of the LAST `window` interactions only.
        windowed = [0.0] * self.population_size
        for delta in deltas[-window:]:
            for pos, d in enumerate(delta):
                windowed[pos] += d
        return windowed
