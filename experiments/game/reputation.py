"""Reputation mechanisms for indirect reciprocity."""

from abc import ABC, abstractmethod


class ReputationMechanism(ABC):
    """Abstract base class for reputation mechanisms."""

    @abstractmethod
    def calculate_reputation_delta(
        self,
        action: str,
        recipient_reputation: float
    ) -> float:
        """
        Calculate the reputation change for a donor's action.

        Args:
            action: "donate" or "not_donate"
            recipient_reputation: Reputation score of the recipient

        Returns:
            Reputation delta (positive or negative)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the mechanism."""
        pass


class ImageScoring(ReputationMechanism):
    """
    Image Scoring reputation mechanism (Nowak & Sigmund, 1998).

    Simple mechanism: reputation score = number of donations - number of defections.
    """

    def __init__(self, delta_cooperate: float = 1.0, delta_defect: float = -1.0):
        """
        Initialize Image Scoring.

        Args:
            delta_cooperate: Reputation change for donating
            delta_defect: Reputation change for not donating
        """
        self.delta_cooperate = delta_cooperate
        self.delta_defect = delta_defect

    def calculate_reputation_delta(
        self,
        action: str,
        recipient_reputation: float = None  # Not used in image scoring
    ) -> float:
        """
        Calculate reputation change based on action only.

        Args:
            action: "donate" or "not_donate"
            recipient_reputation: Ignored for image scoring

        Returns:
            Reputation delta
        """
        if action == "donate":
            return self.delta_cooperate
        else:
            return self.delta_defect

    @property
    def name(self) -> str:
        return "image_scoring"

    def __repr__(self):
        return f"ImageScoring(cooperate={self.delta_cooperate}, defect={self.delta_defect})"


class Standing(ReputationMechanism):
    """
    Standing (Kandori-style) reputation mechanism.

    Context-aware: distinguishes justified vs unjustified defection.
    - Defecting against low-reputation recipient is justified (no reputation loss)
    - Defecting against high-reputation recipient is unjustified (reputation loss)
    """

    def __init__(
        self,
        delta_cooperate: float = 1.0,
        delta_unjustified_defect: float = -1.0,
        delta_justified_defect: float = 0.0,
        threshold: float = 0.0
    ):
        """
        Initialize Standing mechanism.

        Args:
            delta_cooperate: Reputation change for donating
            delta_unjustified_defect: Reputation change for not donating to high-reputation recipient
            delta_justified_defect: Reputation change for not donating to low-reputation recipient
            threshold: Reputation threshold to distinguish "good" vs "bad" recipients
        """
        self.delta_cooperate = delta_cooperate
        self.delta_unjustified_defect = delta_unjustified_defect
        self.delta_justified_defect = delta_justified_defect
        self.threshold = threshold

    def calculate_reputation_delta(
        self,
        action: str,
        recipient_reputation: float
    ) -> float:
        """
        Calculate reputation change based on action and recipient reputation.

        Args:
            action: "donate" or "not_donate"
            recipient_reputation: Reputation score of the recipient

        Returns:
            Reputation delta
        """
        if action == "donate":
            return self.delta_cooperate
        else:
            # Defecting: is it justified?
            if recipient_reputation < self.threshold:
                # Justified: recipient has low reputation
                return self.delta_justified_defect
            else:
                # Unjustified: recipient has good reputation
                return self.delta_unjustified_defect

    @property
    def name(self) -> str:
        return "standing"

    def __repr__(self):
        return (
            f"Standing(cooperate={self.delta_cooperate}, "
            f"unjustified={self.delta_unjustified_defect}, "
            f"justified={self.delta_justified_defect}, "
            f"threshold={self.threshold})"
        )


class DiscriminatorStrategy:
    """
    Discriminator strategy: cooperate if recipient has positive reputation.

    This is a common strategy in indirect reciprocity games.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Initialize Discriminator.

        Args:
            threshold: Reputation threshold for cooperation
        """
        self.threshold = threshold

    def decide(self, recipient_reputation: float) -> bool:
        """
        Decide whether to cooperate based on recipient's reputation.

        Args:
            recipient_reputation: Recipient's reputation score

        Returns:
            True if should cooperate, False otherwise
        """
        return recipient_reputation >= self.threshold


class ScoringStrategy:
    """
    Scoring strategy: always cooperate to build reputation.

    This strategy always donates to maximize reputation.
    """

    def decide(self, recipient_reputation: float = None) -> bool:
        """Always cooperate."""
        return True


class DefectorStrategy:
    """
    Defector strategy: never cooperate.

    This strategy never donates.
    """

    def decide(self, recipient_reputation: float = None) -> bool:
        """Never cooperate."""
        return False
