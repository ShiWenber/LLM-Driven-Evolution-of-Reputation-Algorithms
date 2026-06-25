"""Code-based agent with private reputation store and evolved strategy functions.

Each agent runs TWO compiled functions:
- evaluate(): Updates private reputation scores from observations
- decide(): Makes donation decisions based on reputation

The reputation store is PRIVATE: each agent independently tracks its own
assessment of every other agent. No central reputation signal exists.
"""

from typing import Optional
import hashlib

from ..sandbox.executor import StrategyExecutor, create_strategy_executor

# Initial reputation for agents we have never observed.
# Must be slightly positive (> 0.0) so that strategies using "> 0.0"
# as a donation threshold will donate on first encounter (cold start).
# Without this, all agents see reputation=0.0 for everyone, nobody donates,
# and cooperation irreversibly collapses.
INITIAL_REPUTATION = 0.01


class CodeAgent:
    """Agent powered by evolved evaluate/decide strategy functions."""

    def __init__(
        self,
        agent_id: int,
        code: str,
        strategy_id: Optional[str] = None
    ):
        """
        Initialize agent with strategy code.

        Args:
            agent_id: Unique agent identifier
            code: Python source code defining evaluate() and decide() functions
            strategy_id: Optional identifier for the strategy (hash if not provided)
        """
        self.agent_id = agent_id
        self.code = code
        self.strategy_id = strategy_id or _hash_code(code)

        # Compile the strategy pair
        self._executor = create_strategy_executor(code)
        self._compiled = self._executor is not None

        # Agent state
        self.fitness = 0.0
        self.total_donations = 0
        self.total_decisions = 0
        self.generation = 0
        self.parent_id: Optional[str] = None

        # Private reputation store: agent_id -> reputation score
        self.reputations: dict[int, float] = {}

        # Personal interaction history
        self.my_history: list = []

    @property
    def cooperation_rate(self) -> float:
        """Fraction of decisions that were donations."""
        if self.total_decisions == 0:
            return 0.0
        return self.total_donations / self.total_decisions

    @property
    def is_valid(self) -> bool:
        """Whether the strategy code compiled successfully."""
        return self._compiled

    def decide(
        self,
        recipient_id: int,
        round_num: int,
        population_size: int = 20
    ) -> bool:
        """
        Make a donation decision using the evolved decide() function.

        Uses the agent's PRIVATE reputation assessment of the recipient.
        If the recipient has never been observed, uses INITIAL_REPUTATION.

        Args:
            recipient_id: ID of the potential recipient
            round_num: Current round number
            population_size: Total population size

        Returns:
            True to donate, False to not donate
        """
        if not self._compiled or self._executor is None:
            self.total_decisions += 1
            return False

        rep = self.reputations.get(recipient_id, INITIAL_REPUTATION)

        try:
            result = self._executor.decide(
                recipient_reputation=rep,
                round_num=round_num,
                my_history=self.my_history
            )
        except Exception:
            result = False

        self.total_decisions += 1
        if result:
            self.total_donations += 1

        return result

    def observe(
        self,
        donor_id: int,
        observation: dict,
        round_num: int
    ):
        """
        Observe an interaction and update private reputation of the donor.

        Called whenever the agent witnesses an interaction involving other agents.
        Runs the evaluate() function to update the donor's reputation score.

        Args:
            donor_id: The agent whose action is being evaluated
            observation: Dict with round, donor, recipient, action keys
            round_num: Current round number
        """
        if not self._compiled or self._executor is None:
            return

        current_rep = self.reputations.get(donor_id, INITIAL_REPUTATION)

        # Augment observation with observer-private reputation fields so that
        # LLM-evolved strategies can implement reputation norms that depend on
        # the recipient's standing (e.g. Simple Standing, Judging, IS+ from the
        # "leading eight" of indirect-reciprocity norms). The donor's reputation
        # in the observer's own store is also exposed; both default to
        # INITIAL_REPUTATION for agents the observer has never seen.
        observation = dict(observation)  # shallow copy to avoid mutating caller's dict
        observation["donor_reputation"] = current_rep
        observation["recipient_reputation"] = self.reputations.get(
            observation.get("recipient"), INITIAL_REPUTATION
        )

        try:
            new_rep = self._executor.evaluate(
                current_reputation=current_rep,
                observation=observation,
                my_history=self.my_history,
                round_num=round_num
            )
        except Exception:
            # On evaluation failure, keep current reputation
            return

        self.reputations[donor_id] = new_rep

    def record_interaction(
        self,
        round_num: int,
        role: str,
        partner_id: int,
        action: str,
        partner_action: Optional[str] = None
    ):
        """Record an interaction the agent personally participated in."""
        self.my_history.append({
            "round": round_num,
            "role": role,
            "partner": partner_id,
            "action": action,
            "partner_action": partner_action
        })

    def handle_agent_removed(self, agent_id: int):
        """
        Handle removal of an agent from the population.

        Resets reputation for the removed agent to INITIAL_REPUTATION,
        and removes from the reputation store to free memory.
        """
        if agent_id in self.reputations:
            del self.reputations[agent_id]

    def handle_agents_replaced(self, old_ids: list[int], new_ids: list[int]):
        """
        Handle population turnover after selection.

        Removes reputation entries for eliminated agents.
        New agents will be assigned new IDs; they start with INITIAL_REPUTATION
        when first observed.

        Args:
            old_ids: IDs of agents that were removed
            new_ids: IDs of new agents (not yet in reputation store)
        """
        for aid in old_ids:
            if aid in self.reputations:
                del self.reputations[aid]

    def reset_for_generation(self):
        """Reset agent state for a new generation (keeps code, resets history and reputations)."""
        self.fitness = 0.0
        self.total_donations = 0
        self.total_decisions = 0
        self.my_history = []
        self.reputations = {}

    def clone_with_code(self, new_code: str) -> "CodeAgent":
        """Create a child agent with mutated code but same lineage info."""
        child = CodeAgent(
            agent_id=-1,  # Will be reassigned by population
            code=new_code
        )
        child.generation = self.generation + 1
        child.parent_id = self.strategy_id
        return child

    def __repr__(self):
        status = "valid" if self._compiled else "BROKEN"
        return (
            f"CodeAgent(id={self.agent_id}, "
            f"strategy={self.strategy_id[:8]}, "
            f"gen={self.generation}, "
            f"fitness={self.fitness:.1f}, "
            f"coop={self.cooperation_rate:.2f}, "
            f"{status})"
        )


def _hash_code(code: str) -> str:
    """Generate a short hash for strategy identification."""
    return hashlib.md5(code.encode()).hexdigest()[:12]
