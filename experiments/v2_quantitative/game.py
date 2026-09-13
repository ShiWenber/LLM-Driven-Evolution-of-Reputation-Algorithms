"""2-player Prisoner's Dilemma game engine for the v2 quantitative interface.

Game model:
  - N agents (15 in the default config), each is its own instance with a
    private reputation matrix `reputations: dict[int, float]` keyed by
    agent_id, with `reputations[agent_id]` being the self-rating.
  - Each interaction: draw one random pair of agents independently of earlier
    draws. Both players simultaneously choose C (cooperate) or D (defect).
    Payoffs (benefit=3, cost=1):
      (C, C) -> each +2
      (C, D) -> C gets -1, D gets +3
      (D, C) -> symmetric
      (D, D) -> each 0
  - After all pairs decide, distribute observations per observability
    rules (full / partial / private). For each observed joint action,
    the framework calls observer.observe_and_judge(donor_id, donor_action,
    recipient_id, recipient_action) on the observer. The agent's
    observe_and_judge internally calls its observe() function TWICE
    (once judging the donor, once judging the recipient with roles
    swapped) because the strategy's observe is one-directional.

Noise (both default off, both drawn from ``self.rng`` so a generation stays
reproducible from its seed):

  - ``action_error_probability`` -- execution error. Each player's intended
    action is flipped with this probability before it is used. The executed
    action drives the payoffs and is what everyone observes, so a slip is a
    real defection/cooperation.
  - ``observation_error_probability`` -- perception error. Every observer,
    the two participants included, independently misperceives each action it
    sees with this probability before rating it. Payoffs are untouched, so a
    slip here produces a wrong reputation rather than a wrong outcome.

Both are the same two knobs the invasion / fixation / consensus measurement
modules expose, so evolution and measurement can be run under one noise model.

Backward-compatible with the type1 QuantitativeAgent interface
(choose / observe_and_judge / self_judge / record_donation / etc.).
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Optional
from .agent import QuantitativeAgent


# The one observation protocol the framework implements. Recorded in result
# configs so an archived run states which protocol produced it.
OBSERVATION_SCHEDULE = "asynchronous"


def flip_action(action: str, rng, probability: float) -> str:
    """Return ``action``, flipped to the other PD action with ``probability``.

    The single definition of an action flip in the engine. Both noise sources
    below are expressed through it, so execution and perception noise cannot
    drift apart:

      * execution error (``action_error_probability``) is one flip applied to
        the *executed* action, drawn once per player per interaction;
      * perception error (``observation_error_probability``) is a flip applied
        to the *observed* action, drawn once per observer per seen action.

    ``probability <= 0`` returns ``action`` without touching ``rng``. That is
    what keeps a noise-free run's random stream bit-identical to the stream of
    a run recorded before this feature existed, so archived results stay
    reproducible.
    """
    if probability <= 0.0:
        return action
    if rng.random() >= probability:
        return action
    return "defect" if action == "cooperate" else "cooperate"


def resolve_fitness_window(
    fraction: Optional[float],
    total_interactions: int,
    pairs_per_round: int,
) -> Optional[int]:
    """Resolve a fractional fitness window to an absolute interaction count.

    ``fraction`` is the share of a generation's joint actions whose payoffs
    count toward selection fitness; the rest is burn-in (still played, so
    ``observe()`` and reputations evolve, but not scored).

    Returns how many of the generation's LAST joint actions to count, or
    ``None`` when every interaction counts (no burn-in). ``None``, ``0``, or
    any value outside ``(0, 1)`` disables the window.

    The window is floored to a whole number of rounds (never fewer than one).
    Matching-based callers can pass their pairs-per-round count to preserve
    round boundaries; this asynchronous game passes 1 and does not assume
    equal per-agent exposure.
    """
    if fraction is None:
        return None
    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        return None
    if not (0.0 < fraction < 1.0):
        return None
    pairs = max(1, int(pairs_per_round))
    total = int(total_interactions)
    if total <= 0:
        return None
    rounds = total // pairs
    if rounds <= 0:
        return None
    # The epsilon absorbs binary-float representation error so that e.g.
    # 0.7 * 10 -> 6.999999999999999 still floors to 7.
    rounds_in_window = max(1, math.floor(fraction * rounds + 1e-9))
    window = rounds_in_window * pairs
    if window >= total:
        return None
    return window


def prisoners_dilemma_payoff(
    *,
    my_cooperates: bool,
    other_cooperates: bool,
    benefit: float,
    cost: float,
) -> float:
    """One player's payoff for a single symmetric two-player PD interaction.

    Cooperating costs ``cost``; a player receives ``benefit`` for every partner
    action that cooperates. This is the single definition of the payoff matrix
    used by the evolutionary engine and by the fixed-strategy invasion
    measurement, so the matrix cannot drift between them.
    """
    return (
        (benefit if other_cooperates else 0.0)
        - (cost if my_cooperates else 0.0)
    )


class DonorGame:
    """2-player simultaneous-PD game with reputation tracking.

    One step plays a single pair of agents, drawn uniformly at random. That
    pair's observations are delivered before the next pair is drawn, so
    reputations evolve continuously within a generation and a later pair
    already sees the effects of earlier ones.
    """

    def __init__(
        self,
        population_size: int,
        benefit: float = 3.0,
        cost: float = 1.0,
        observability: str = "full",
        observability_p: float = 1.0,
        seed: int = 42,
        fitness_window_fraction: Optional[float] = 0.2,
        action_error_probability: float = 0.0,
        observation_error_probability: float = 0.0,
    ):
        self.population_size = population_size
        self.benefit = benefit
        self.cost = cost
        self.observability = observability
        self.observability_p = observability_p
        self.seed = seed
        self.rng = random.Random(seed)
        # Execution error: the probability that a player's intended action is
        # mis-executed. The executed action -- not the intention -- determines
        # the payoffs and is what every observer (including the actor itself)
        # sees, so an execution slip is a real event.
        self.action_error_probability = action_error_probability
        # Perception error: the probability that an observer misperceives an
        # action while judging it. Each observer draws independently for each
        # action it sees, so two observers can disagree about the same
        # interaction. A slip here corrupts a rating; it does not change what
        # was actually played or paid.
        self.observation_error_probability = observation_error_probability
        self.agents: List[QuantitativeAgent] = []
        self.round_num = 0
        # Global log of every joint action in the current generation
        self._global_log: List[Dict] = []
        # Payoffs (indexed by list position)
        self.payoffs = [0.0] * population_size
        # Per-interaction payoff deltas (one entry per joint action played
        # in the current generation). Used to compute windowed fitness
        # (= sum of payoffs over only the last N interactions, treating
        # the first M as burn-in).
        self._interaction_deltas: List[List[float]] = []
        # Fitness window: the share of the generation's joint actions whose
        # payoffs count toward selection fitness. ``None``, ``0``, or any
        # value outside (0, 1) counts every interaction (no burn-in). See
        # ``resolve_fitness_window`` for how the share becomes an absolute
        # interaction count.
        self.fitness_window_fraction = fitness_window_fraction

    def setup_population(self, agents: List[QuantitativeAgent]):
        self.agents = agents
        # agent_id is a STABLE global identity (assigned monotonically by
        # the population manager). List index is just iteration order.
        # Build a lookup so play_interaction can resolve agent_id -> agent
        # object without scanning the list each time.
        self._agent_by_id = {a.agent_id: a for a in self.agents}

    def _play_pair(self, donor_id: int, recipient_id: int) -> Dict:
        """Play one simultaneous PD between two agents and record it.

        Both players decide from the reputations as they stand right now, so
        neither decision sees the other's effect.
        """
        donor = self._agent_by_id[donor_id]
        recipient = self._agent_by_id[recipient_id]
        # Both players choose simultaneously. We call choose() in
        # sequence but both decisions are based on each player's
        # own (my_rep, opp_rep) before any update is applied — no
        # information leaks between the two calls.
        action1 = donor.choose(recipient_id, round_num=self.round_num)
        action2 = recipient.choose(donor_id, round_num=self.round_num)
        # Execution error. The flip is applied to the intended action, before
        # anything else consumes it, so the executed action is what pays out,
        # what the pair's own members observe about each other, and what the
        # strategy's own bookkeeping records. Note the flips are drawn for
        # BOTH players before either is used, so a slip by one player cannot
        # shift the other's draw.
        donor_action_str = flip_action(
            "cooperate" if action1 else "defect",
            self.rng,
            self.action_error_probability,
        )
        recipient_action_str = flip_action(
            "cooperate" if action2 else "defect",
            self.rng,
            self.action_error_probability,
        )
        donor.record_donation(
            recipient_id, donor_action_str == "cooperate", self.round_num
        )
        recipient.record_donation(
            donor_id, recipient_action_str == "cooperate", self.round_num
        )
        # Payoffs (use list position to index the payoffs array)
        donor_pos = self.agents.index(donor)
        recipient_pos = self.agents.index(recipient)
        # Per-interaction deltas (for windowed fitness / burn-in).
        # Initialize zeros for all agents; only the two players
        # in this pair have nonzero entries.
        pair_delta = [0.0] * self.population_size
        # Payoffs come from the shared matrix definition so the engine and
        # the fixed-strategy measurements cannot disagree about it. They use
        # the EXECUTED actions: a trembling hand that turns cooperation into
        # defection really does defraud the partner.
        donor_payoff = prisoners_dilemma_payoff(
            my_cooperates=donor_action_str == "cooperate",
            other_cooperates=recipient_action_str == "cooperate",
            benefit=self.benefit,
            cost=self.cost,
        )
        recipient_payoff = prisoners_dilemma_payoff(
            my_cooperates=recipient_action_str == "cooperate",
            other_cooperates=donor_action_str == "cooperate",
            benefit=self.benefit,
            cost=self.cost,
        )
        self.payoffs[donor_pos] += donor_payoff
        pair_delta[donor_pos] += donor_payoff
        self.payoffs[recipient_pos] += recipient_payoff
        pair_delta[recipient_pos] += recipient_payoff
        # Record this interaction's per-agent payoff deltas for the
        # windowed-fitness computation (the first
        # `total - window` interactions count as burn-in).
        self._interaction_deltas.append(pair_delta)
        # Store actions as STRING so the strategy code can
        # pattern-match on them in observe().
        interaction = {
            "round": self.round_num,
            "donor": donor_id,
            "recipient": recipient_id,
            "donor_action": donor_action_str,
            "recipient_action": recipient_action_str,
        }
        self._global_log.append(interaction)
        return interaction

    def play_interaction(self) -> Dict:
        """Play one step: one pair drawn uniformly at random.

        Two distinct agents are sampled uniformly, they play one simultaneous
        PD from the current reputations, and the caller is expected to deliver
        the resulting observations before the next pair is drawn. Repeated
        draws are independent, so the same agents can interact many times and
        per-agent interaction counts are multinomial rather than equal.
        """
        if self.population_size < 2:
            raise ValueError(
                "an interaction needs at least 2 agents, got "
                f"{self.population_size}"
            )
        self.round_num += 1
        agent_ids = [a.agent_id for a in self.agents]
        donor_id, recipient_id = self.rng.sample(agent_ids, 2)
        return {
            "round": self.round_num,
            "interactions": [self._play_pair(donor_id, recipient_id)],
        }

    def _perceive(self, action: str) -> str:
        """The action one observer perceives, subject to perception error.

        Called once per observer per action, with fresh draws, so observers
        misperceive independently -- two agents can walk away from the same
        interaction with opposite views of what happened.
        """
        return flip_action(
            action, self.rng, self.observation_error_probability
        )

    def distribute_observations_and_self_judgments(
        self, interactions: Optional[List[Dict]] = None,
    ):
        """Deliver one step's observations and self-judgments.

        ``interactions`` is this step's joint actions as returned by
        ``play_interaction`` -- pass it. It defaults to a scan of the generation
        log for the current step, which is only there for callers that discard
        ``play_interaction``'s return value and costs O(generation length) per
        step.

        For each joint action:
          - donor observes (self-judgment) via self_judge
          - recipient observes (self-judgment) via self_judge
          - for each third-party observer (per observability rules),
            call observer.observe_and_judge(...)

        Perception error is applied here, at delivery time: every observer --
        the two participants included -- sees each action through its own
        independent flip. That is why a self-judgment can be wrong about the
        agent's own action, matching the invasion / fixation measurements in
        ``experiments/analysis`` where the observer loop also covers the two
        participants.
        """
        recent = self._round_interactions(interactions)
        # Step 1: self-judgments for BOTH players in each pair
        for inter in recent:
            donor_id = inter["donor"]
            recipient_id = inter["recipient"]
            donor_action = inter["donor_action"]
            recipient_action = inter["recipient_action"]
            # Donor's self-judgment. Each of the two actions it perceives is
            # flipped independently; this does not mutate the logged actions.
            self._agent_by_id[donor_id].self_judge(
                donor_action=self._perceive(donor_action),
                recipient_id=recipient_id,
                recipient_action=self._perceive(recipient_action),
            )
            # Recipient's self-judgment (in PD, recipient also acts)
            self._agent_by_id[recipient_id].self_judge(
                donor_action=self._perceive(recipient_action),
                recipient_id=donor_id,
                recipient_action=self._perceive(donor_action),
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
                        donor_action=self._perceive(donor_action),
                        recipient_id=recipient_id,
                        recipient_action=self._perceive(recipient_action),
                    )
                elif self.observability.startswith("partial"):
                    if self.rng.random() < self.observability_p:
                        self._agent_by_id[obs_id].observe_and_judge(
                            donor_id=donor_id,
                            donor_action=self._perceive(donor_action),
                            recipient_id=recipient_id,
                            recipient_action=self._perceive(recipient_action),
                        )

    def _round_interactions(
        self, interactions: Optional[List[Dict]],
    ) -> List[Dict]:
        """The joint actions to distribute observations for.

        The explicit argument avoids rescanning the generation log on every
        step, which made a generation cost O(steps^2) during observation.
        """
        if interactions is not None:
            return interactions
        return [i for i in self._global_log if i["round"] == self.round_num]

    def run_generation(self) -> Dict:
        """Run a full generation: one step per agent, then aggregate stats.

        Interaction counts per agent are multinomial, so ``population_size``
        steps is the natural default rather than an equality guarantee: each
        agent is drawn a mean of twice per generation (once as donor, once as
        recipient).
        """
        self.round_num = 0
        self.payoffs = [0.0] * self.population_size
        self._global_log = []
        self._interaction_deltas = []
        for _ in range(self.population_size):
            self.distribute_observations_and_self_judgments(
                self.play_interaction()["interactions"]
            )
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
        """Return each agent's mean payoff per action in the fitness window.

        Both the payoff numerator and the action-count denominator use the
        same trailing joint-action window. Earlier interactions are burn-in:
        they still affect reputations, but not selection fitness. An agent
        that was never drawn in the window receives fitness 0.0.

        Random independent pairing gives agents different exposure counts;
        dividing by actual actions prevents extra draws alone from raising
        an otherwise identical strategy's fitness.
        """
        deltas = self._interaction_deltas
        n_total = len(deltas)
        window = resolve_fitness_window(self.fitness_window_fraction, n_total, 1)
        selected_deltas = deltas if window is None else deltas[-window:]
        selected_interactions = (
            self._global_log if window is None else self._global_log[-window:]
        )
        payoff_totals = [0.0] * self.population_size
        action_counts = [0] * self.population_size
        position_by_id = {agent.agent_id: pos for pos, agent in enumerate(self.agents)}
        for delta in selected_deltas:
            for pos, d in enumerate(delta):
                payoff_totals[pos] += d
        for interaction in selected_interactions:
            action_counts[position_by_id[interaction["donor"]]] += 1
            action_counts[position_by_id[interaction["recipient"]]] += 1
        return [
            payoff_totals[pos] / count if count else 0.0
            for pos, count in enumerate(action_counts)
        ]
