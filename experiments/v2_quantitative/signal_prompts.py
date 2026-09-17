"""Prompts exclusive to agent-type2-signal; historical prompts stay unchanged."""

SIGNAL_INTERFACE_PROMPT = '''
Define a private memory signal and a policy with this interface:

from dataclasses import dataclass, field

@dataclass
class Signal:
    # Design your own fields, meanings, and defaults here.
    # All fields need a default or field(default_factory=...).
    ...

class LLMAgent:
    def __init__(self):
        pass

    def observe(self, target_signal: Signal, target_action: str,
                partner_signal: Signal, partner_action: str,
                self_signal: Signal) -> Signal:
        ...

    def decide(self, opponent_signal: Signal) -> bool:
        ...

Signal() must work with no arguments. Signal fields and their meanings are
yours to design: finite numbers, strings, booleans, None, and nested lists,
tuples, dictionaries or sets are allowed. Do not assume a scalar score or a
predefined reputation range. Mutable defaults need field(default_factory=...).
Every observer uses its own private Signal format; agents never exchange
Signal objects. self_signal is this observer's current assessment of itself.

observe returns only the target's new Signal. The framework calls it twice,
with target and partner swapped, using the SAME pre-interaction self_signal
and pre-interaction participant signals. This includes the observer's own
interactions: the assessment targeting itself updates its own signal too.
Actions are 'cooperate' or 'defect'. decide receives only the opponent signal
and must return a bool (True=cooperate). All arguments are independent copies.

The framework owns identity-to-signal lookup. IDs, object identity tests,
reflection, external state, files and network are unavailable. Do not create
identity fields or remember/compare identities. Ordinary value comparisons
and evidence counts are allowed. Keep all persistent memory in Signal;
LLMAgent has no mutable instance/class state. Use ordinary instance methods,
not staticmethod/classmethod. Public self.helper(...) methods are allowed.
Do not use initial_signal or an opponent-context attribute.

The restricted runtime supports arithmetic, conditionals, bounded loops and
comprehensions, common container methods, and imports from dataclasses
(dataclass, field, replace), math, or random. Random exposes only random(),
uniform(), randint(), randrange(), choice(), with a framework-seeded private
stream independent of agent IDs; seed/state access and reflection are absent.
Use plain @dataclass and no inheritance or special methods. Assign a field
as signal.field = expression rather than using += on fields. Method defaults
and variadic parameters are not supported. Return only data in Signal, never
functions, classes, or resources. Limits: 12000 source characters, 4096 data
nodes per signal, depth 32, and 20000 execution steps per call. Bound or compress
histories so signals remain within these limits throughout long simulations.

Output only Python code. Define Signal followed by LLMAgent.
'''

INIT_PROMPT_SIGNAL = (
    "Design a strategy intended to maximize its average payoff per action.\n"
    "Evolve the private signal representation, assessment, and decision together.\n"
    + SIGNAL_INTERFACE_PROMPT
)

MUTATION_PROMPT_SIGNAL = (
    "Create a child of the following strategy. Preserve useful behavior and "
    "make at least one change. You may change the Signal schema and both methods "
    "together; new agents start with fresh Signal() values.\n"
    "PARENT FITNESS: {fitness:.3f}\nPARENT:\n{parent_code}\n"
    + SIGNAL_INTERFACE_PROMPT
)

SMALL_MUTATION_PROMPT_SIGNAL = (
    "Create a related variation of the following strategy. Keep the schema "
    "and methods mutually compatible.\n"
    "PARENT FITNESS: {fitness:.3f}\nPARENT:\n{parent_code}\n"
    + SIGNAL_INTERFACE_PROMPT
)

DELIBERATE_MUTATION_PROMPT_SIGNAL = (
    "Infer a plausible weakness of the parent and create a child intended to "
    "obtain higher fitness. Selection evaluates whether it succeeds. Keep the "
    "Signal schema and both methods mutually compatible.\n"
    "PARENT FITNESS: {fitness:.3f}\nPARENT:\n{parent_code}\n"
    + SIGNAL_INTERFACE_PROMPT
)
