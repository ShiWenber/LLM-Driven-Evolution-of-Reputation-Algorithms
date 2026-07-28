"""v2 quantitative interface agent (simplified).

Each agent runs the SAME code for two functions:

  def evaluate(
      donor_reputation: float,
      recipient_reputation: float,
      donor_action: str,
      recipient_action: str,
      my_reputation: float
  ) -> float:
      # Returns new donor_reputation (clamped to [-1, 1])

  def decide(
      my_reputation: float,
      opponent_reputation: float
  ) -> bool:
      # Returns True to donate, False to defect

Architecture:
  - Single private reputation matrix `reputations: dict[int, float]`.
    The agent's own self-rating is `reputations[agent_id]`. This way
    `reputations` is a uniform dict; no special field needed.
  - When the agent is the donor in a third-party observation, the
    framework calls `evaluate(...)` with donor=self.agent_id,
    which updates `reputations[self_id]` (the self-rating). Same
    function, same args.
  - Population turnover drops entries of removed IDs.
"""
from __future__ import annotations
from typing import Dict


INITIAL_REPUTATION = 0.1  # default for unseen (incl. self at start)


class QuantitativeAgent:
    def __init__(self, agent_id: int, code: str, executor=None):
        self.agent_id = agent_id
        self.code = code
        self._executor = executor
        # Private reputation matrix; includes self at key agent_id
        self.reputations: Dict[int, float] = {agent_id: INITIAL_REPUTATION}
        # Tracking
        self.fitness: float = 0.0
        self.total_donations: int = 0
        self.total_decisions: int = 0
        self.cooperations: int = 0

    @property
    def cooperation_rate(self) -> float:
        return (self.cooperations / self.total_decisions) if self.total_decisions else 0.0

    # --- Reputation accessors ----------------------------------------------
    def get_reputation(self, other_id: int) -> float:
        return self.reputations.get(other_id, INITIAL_REPUTATION)

    def get_self_reputation(self) -> float:
        # Self is just an entry in the same reputations dict.
        return self.reputations.get(self.agent_id, INITIAL_REPUTATION)

    def update_reputation(self, other_id: int, new_rep: float):
        new_rep = max(-1.0, min(1.0, new_rep))
        self.reputations[other_id] = new_rep

    def update_self_reputation(self, new_rep: float):
        self.update_reputation(self.agent_id, new_rep)

    # --- Framework-driven actions -----------------------------------------
    def observe_and_judge(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ):
        """Update my view of `donor_id` based on the observed interaction.

        Note: when the framework calls this with donor_id == self.agent_id,
        this updates my self-rating (the same function is used for both
        observing others and judging yourself).
        """
        if self._executor is None:
            return
        donor_rep = self.get_reputation(donor_id)
        recipient_rep = self.get_reputation(recipient_id)
        my_rep = self.get_self_reputation()
        try:
            new_rep = self._executor.evaluate(
                donor_reputation=donor_rep,
                recipient_reputation=recipient_rep,
                donor_action=donor_action,
                recipient_action=recipient_action,
                my_reputation=my_rep,
            )
        except Exception:
            return
        self.update_reputation(donor_id, new_rep)

    # The legacy `self_judge` is now identical to `observe_and_judge`
    # but with donor_id == self.agent_id. Provided as a thin alias.
    def self_judge(
        self,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ):
        self.observe_and_judge(
            donor_id=self.agent_id,
            donor_action=donor_action,
            recipient_id=recipient_id,
            recipient_action=recipient_action,
        )

    def choose(self, opponent_id: int, round_num: int = 0) -> bool:
        if self._executor is None:
            return False
        my_rep = self.get_self_reputation()
        opp_rep = self.get_reputation(opponent_id)
        try:
            return bool(self._executor.decide(
                my_reputation=my_rep,
                opponent_reputation=opp_rep,
            ))
        except Exception:
            return False

    # --- Generation tracking -----------------------------------------------
    def reset_for_generation(self):
        self.total_donations = 0
        self.total_decisions = 0
        self.cooperations = 0
        # NOTE: do NOT reset reputations; they accumulate across generations
        # within a trial.

    def record_donation(self, partner_id: int, donated: bool, round_num: int):
        self.total_decisions += 1
        if donated:
            self.cooperations += 1

    def handle_agents_replaced(self, old_ids, new_ids):
        for old in old_ids:
            self.reputations.pop(old, None)
