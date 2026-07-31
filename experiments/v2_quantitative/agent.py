"""v2 quantitative interface agent (simplified).

Each agent runs the SAME code for two functions:

  def evaluate(
      target_reputation: float,     # observer's current rating of the target
      target_action: str,            # target's last action: 'cooperate' or 'defect'
      my_reputation: float           # observer's own self-rating
  ) -> float:
      # Returns new target_reputation (clamped to [-1, 1])

  def decide(
      my_reputation: float,
      opponent_reputation: float
  ) -> bool:
      # Returns True to cooperate, False to defect

Note: the game is a 2-player simultaneous Prisoner's Dilemma (not a
donor game). For each joint action (donor_action, recipient_action),
the framework calls observe_and_judge on each observer, which
internally calls evaluate() twice (once for donor, once for recipient),
using each player's own action as target_action.

Architecture:
  - Single private reputation matrix `reputations: dict[int, float]`.
    The agent's own self-rating is `reputations[agent_id]`. This way
    `reputations` is a uniform dict; no special field needed.
  - The framework's observe_and_judge calls evaluate() twice per
    joint action: once for donor (target_action = donor_action),
    once for recipient (target_action = recipient_action).
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
    def _call_evaluate(self, target_rep: float, target_action: str, my_rep: float) -> float:
        """Call the executor's evaluate and return the new target rep.
        On any exception, return the unchanged target rep."""
        if self._executor is None:
            return target_rep
        try:
            return self._executor.evaluate(target_rep, target_action, my_rep)
        except Exception:
            return target_rep

    def observe_and_judge(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ):
        """Update my view of `donor_id` based on donor_action, and my view
        of `recipient_id` based on recipient_action.

        In the PD game, both donor and recipient are active players, so
        the observer judges BOTH of them: this method calls evaluate()
        twice — once with target=(donor_id, donor_action) and once with
        target=(recipient_id, recipient_action). When the observer
        itself is one of the two players (self-judgment), it
        transparently updates its own self-rating because
        `reputations[self.agent_id]` is the same dict entry.
        """
        my_rep = self.get_self_reputation()
        # Update donor's view based on donor's action
        donor_rep = self.get_reputation(donor_id)
        new_donor_rep = self._call_evaluate(donor_rep, donor_action, my_rep)
        self.update_reputation(donor_id, new_donor_rep)
        # Update recipient's view based on recipient's action
        recipient_rep = self.get_reputation(recipient_id)
        new_recipient_rep = self._call_evaluate(
            recipient_rep, recipient_action, my_rep
        )
        self.update_reputation(recipient_id, new_recipient_rep)

    # Self-judgment is just observe_and_judge with donor_id == self.agent_id.
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
