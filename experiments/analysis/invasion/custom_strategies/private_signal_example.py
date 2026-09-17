"""Small bounded-memory example, not a claim of improvement over Leading Eight."""
from dataclasses import dataclass


@dataclass
class Signal:
    stance: str = "unknown"
    evidence: int = 0
    recent_actions: tuple = ()


class LLMAgent:
    def __init__(self):
        pass

    def observe(self, target_signal, target_action, partner_signal, partner_action, self_signal):
        history = (target_signal.recent_actions + (target_action,))[-4:]
        if target_action == "cooperate":
            stance = "helpful"
        elif partner_signal.stance == "unhelpful":
            stance = "justified"
        else:
            stance = "unhelpful"
        return Signal(stance, min(100, target_signal.evidence + 1), history)

    def decide(self, opponent_signal):
        return opponent_signal.stance != "unhelpful"
