"""v2 quantitative interface agent (simplified).

Each agent runs the SAME code for two functions:

  def observe(
      A_rep: float,        # observer's current rating of player A (target)
      A_action: str,       # A's action: 'cooperate' or 'defect'
      B_rep: float,        # observer's current rating of player B (partner)
      B_action: str,       # B's action: 'cooperate' or 'defect'
      my_rep: float        # observer's own self-rating
  ) -> float:
      # Returns A's NEW reputation only, clamped to [-1, 1] by the
      # framework. The interface is ONE-DIRECTIONAL: it judges a single
      # target player, so the strategy code does not have to repeat a
      # symmetric update for both players.

  def decide(
      my_reputation: float,
      opponent_reputation: float
  ) -> bool:
      # Returns True to cooperate, False to defect

Note: the game is a 2-player simultaneous Prisoner's Dilemma (not a
 donor game). For each joint action (A_action, B_action), the framework
 calls observe_and_judge on each observer, which internally calls
 observe() TWICE — once judging A, once judging B with the roles
 swapped:
     new_A_rep = observe(A_rep, A_action, B_rep, B_action, my_rep)
     new_B_rep = observe(B_rep, B_action, A_rep, A_action, my_rep)

Architecture:
  - Single private reputation matrix `reputations: dict[int, float]`.
    The agent's own self-rating is `reputations[agent_id]`. This way
    `reputations` is a uniform dict; no special field needed.
  - The framework's observe_and_judge calls observe() twice per
    joint action with both sides of the interaction.
  - Population turnover drops entries of removed IDs.
"""
INITIAL_REPUTATION = 0.0  # every generation starts neutral (incl. unseen/self)


class QuantitativeAgent:
    def __init__(self, agent_id: int, code: str, executor=None):
        self.agent_id = agent_id
        self.code = code
        self._executor = executor
        # Private reputation matrix; includes self at key agent_id
        self.reputations: dict[int, float] = {agent_id: INITIAL_REPUTATION}
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
        # Fast path. Executor output is already a float clamped to [-1, 1], so
        # the common case needs neither the float() conversion nor the
        # min()/max() pair -- and this is the hottest clamp in the sweep, called
        # millions of times per run.
        if type(new_rep) is float and -1.0 <= new_rep <= 1.0:
            self.reputations[other_id] = new_rep
            return
        self.reputations[other_id] = max(-1.0, min(1.0, float(new_rep)))

    def update_self_reputation(self, new_rep: float):
        self.update_reputation(self.agent_id, new_rep)

    # --- Framework-driven actions -----------------------------------------
    def _call_observe(
        self,
        A_rep: float,
        A_action: str,
        B_rep: float,
        B_action: str,
        my_rep: float,
    ) -> float:
        """Call strategy observe for ONE target player A; returns A's new reputation.

        The executor exposes the 5-arg one-directional observe(); it already
        coerces the strategy's return value to a float and clamps it, and it
        swallows strategy exceptions itself, so this wrapper adds neither.
        """
        executor = self._executor
        if executor is None:
            return A_rep
        try:
            return executor.observe(A_rep, A_action, B_rep, B_action, my_rep)
        except Exception:  # noqa: BLE001 - strategy code may raise arbitrary exceptions
            return A_rep

    def observe_and_judge(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ):
        """Update my view of both players from one symmetric PD interaction.

        The strategy's observe() is one-directional (it judges a single
        target player), so the framework calls it twice — once for the
        donor and once for the recipient with the roles swapped.
        """
        # Read all three pre-interaction values up front. The dict is aliased
        # rather than reached through the getters because this method runs once
        # per (observer, interaction) -- hundreds of millions of times per
        # sweep -- and the getters are pure lookups.
        reputations = self.reputations
        my_rep = reputations.get(self.agent_id, INITIAL_REPUTATION)
        A_rep = reputations.get(donor_id, INITIAL_REPUTATION)
        B_rep = reputations.get(recipient_id, INITIAL_REPUTATION)
        new_A_rep = self._call_observe(
            A_rep, donor_action, B_rep, recipient_action, my_rep
        )
        new_B_rep = self._call_observe(
            B_rep, recipient_action, A_rep, donor_action, my_rep
        )
        self.update_reputation(donor_id, new_A_rep)
        self.update_reputation(recipient_id, new_B_rep)

    # Self-judgment is just observe_and_judge with A_id = self.agent_id and B_id = other_id.
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
        except Exception:  # noqa: BLE001 - strategy code may raise arbitrary exceptions
            return False

    # --- Generation tracking -----------------------------------------------
    def reset_for_generation(self):
        self.total_donations = 0
        self.total_decisions = 0
        self.cooperations = 0
        self.reputations = {self.agent_id: INITIAL_REPUTATION}

    def record_donation(self, partner_id: int, donated: bool, round_num: int):
        self.total_decisions += 1
        if donated:
            self.cooperations += 1

    def handle_agents_replaced(self, old_ids, new_ids):
        for old in old_ids:
            self.reputations.pop(old, None)
