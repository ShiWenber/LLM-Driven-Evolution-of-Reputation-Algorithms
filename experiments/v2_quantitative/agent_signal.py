"""Framework-owned private Signal memory; compatible with existing game hooks."""
from .signal_executor import SIGNAL_INTERFACE_VERSION, SignalStrategyExecutor


ALLC_SIGNAL_SOURCE = '''
from dataclasses import dataclass
@dataclass
class Signal:
    observations: int = 0
class LLMAgent:
    def __init__(self):
        pass
    def observe(self, target_signal, target_action, partner_signal, partner_action, self_signal):
        return Signal(target_signal.observations + 1)
    def decide(self, opponent_signal):
        return True
'''
ALLD_SIGNAL_SOURCE = ALLC_SIGNAL_SOURCE.replace("return True", "return False")


class SignalAgent:
    def __init__(self, agent_id: int, executor: SignalStrategyExecutor, code: str = ""):
        self.agent_id = agent_id
        self._executor = executor
        self.code = code
        self.fitness = 0.0
        self.reset_for_generation()

    def get_signal(self, other_id):
        if other_id not in self.signals:
            self.signals[other_id] = self._executor.new_signal()
        return self.signals[other_id]

    def get_self_reputation(self):
        # Explicitly absent: a structured signal is not a scalar reputation.
        return None

    @property
    def cooperation_rate(self):
        return self.cooperations / self.total_decisions if self.total_decisions else 0.0

    def choose(self, opponent_id, round_num=0):
        return self._executor.decide(self.get_signal(opponent_id))

    def observe_and_judge(self, donor_id, donor_action, recipient_id, recipient_action):
        target = self.get_signal(donor_id)
        partner = self.get_signal(recipient_id)
        own = self.get_signal(self.agent_id)
        new_target = self._executor.observe(target, donor_action, partner, recipient_action, own)
        new_partner = self._executor.observe(partner, recipient_action, target, donor_action, own)
        # Neither update is committed if either assessment fails.
        self.signals[donor_id] = new_target
        self.signals[recipient_id] = new_partner

    def self_judge(self, donor_action, recipient_id, recipient_action):
        self.observe_and_judge(self.agent_id, donor_action, recipient_id, recipient_action)

    def reset_for_generation(self):
        self.signals = {self.agent_id: self._executor.new_signal()}
        self.total_donations = self.total_decisions = self.cooperations = 0

    def record_donation(self, partner_id, donated, round_num):
        self.total_decisions += 1
        self.cooperations += int(donated)

    def handle_agents_replaced(self, old_ids, new_ids):
        for removed in old_ids:
            self.signals.pop(removed, None)

    def signal_metadata(self):
        return {
            "signal_interface_version": SIGNAL_INTERFACE_VERSION,
            "signal_schema": self._executor.schema_record(),
            "self_signal": self._executor.signal_record(self.get_signal(self.agent_id)),
            "signal_runtime_errors": dict(self._executor.errors),
            "signal_rng_seed": self._executor.seed,
        }
