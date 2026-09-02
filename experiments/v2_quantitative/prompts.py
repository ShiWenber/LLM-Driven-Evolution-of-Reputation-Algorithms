"""Prompts for the v2 quantitative interface LLM evolution (2-player PD).

The interface is:
    observe(A_rep, A_action, B_rep, B_action, my_rep) -> float
  decide(my_reputation, opponent_reputation) -> bool

The game is a 2-player simultaneous Prisoner's Dilemma. Each round,
two matched agents independently choose C (cooperate) or D (defect);
payoffs are (C,C) -> both +1, (C,D) -> C gets -1 / D gets +2, etc.
This matches the leading-eight / quantitative-assessment literature
extended to symmetric interactions.
"""


INIT_PROMPT_V2 = """You are a Python programmer designing a strategy for a 2-player
Prisoner's Dilemma game with reputation. The game has {population_size}
agents that play {num_rounds_per_gen} rounds per generation,
{num_generations} generations total. Each round, the {population_size}
agents are randomly paired into {num_pairs} pairs. In each
pair, BOTH agents simultaneously choose to 'cooperate' (give benefit={benefit:g}
to the partner at cost={cost:g} to self) or 'defect' (no exchange). Payoffs:
  (C, C) -> each gets {benefit:g} - {cost:g} = {cc_payoff:g}
  (C, D) -> C gets -{cost:g} (sucker's payoff), D gets +{benefit:g}
  (D, C) -> symmetric
  (D, D) -> each gets 0

After each joint action, the framework records both actions and the
agents' IDs. It then calls each observer's `observe` TWICE: once to
update the observer's private rating of player A (the donor) and once
to update its rating of player B (the recipient), with the roles
swapped.

Your strategy consists of TWO functions. Define BOTH.

The interface for your strategy is:

```python
def observe(
    A_rep: float,       # observer's current rating of player A (target)
    A_action: str,      # A's last action: 'cooperate' or 'defect'
    B_rep: float,       # observer's current rating of player B (partner)
    B_action: str,      # B's last action: 'cooperate' or 'defect'
    my_reputation: float  # observer's own self-rating
) -> float:
    # Return A's NEW reputation only (a single float).
    # Values will be clamped to [-1.0, 1.0].
    pass

def decide(
    my_reputation: float,
    opponent_reputation: float
) -> bool:
    # Return True to cooperate, False to defect.
    pass
```

Important rules:
  1. Both functions MUST be defined.
  2. observe() is ONE-DIRECTIONAL: it judges only player A and returns
      A's new reputation. The framework calls it twice per joint action
      (once with A=donor, once with A=recipient) to update both players.
      Write ONE judging rule for a single target; do NOT repeat it for
      both players inside the function.
  3. There is no separate "self-evaluation" function; if the observer
      is one of the two players in the joint action, the framework
      calls observe() the same way.
  4. Reputation is in [-1.0, 1.0]. Treat 0.0 as neutral.

Generate ONE strategy pair. Output ONLY the Python code, no prose,
no markdown fences. The code must define `observe` and `decide`.
"""


MUTATION_PROMPT_V2 = """You are mutating an existing strategy for a 2-player
Prisoner's Dilemma game with reputation. The game has {population_size}
agents, {num_rounds_per_gen} rounds per generation, {num_generations}
generations.

The interface for your strategy is:

```python
def observe(
    A_rep: float,       # observer's current rating of player A (target)
    A_action: str,      # A's last action: 'cooperate' or 'defect'
    B_rep: float,       # observer's current rating of player B (partner)
    B_action: str,      # B's last action: 'cooperate' or 'defect'
    my_reputation: float
) -> float:
    # Return A's NEW reputation only (a single float).
    pass

def decide(
    my_reputation: float,
    opponent_reputation: float
) -> bool:
    pass
```

`observe` is ONE-DIRECTIONAL: it judges only player A and returns A's
new reputation. The framework calls it twice per joint action (once
with A=donor, once with A=recipient) to update both players, so write
one judging rule for a single target — do NOT repeat it for both sides.

The parent strategy to mutate is below.

PARENT FITNESS: {fitness:.3f}

```python
{parent_code}
```

Generate a child strategy. Suggestions (you don't have to follow any):
  - Slightly increase or decrease the step size (e.g., 0.3 vs 0.4)
  - Add or remove dependence on `my_reputation`
  - Adjust the threshold for decide()
  - Add an asymmetry: cooperation gives a smaller step than defection
    punishes

Output ONLY the Python code, no prose, no markdown fences.
"""


# =======================================================================
# Type-2 prompts: the LLM writes a FULL Python class, not two functions.
# Strictly no hints about state structure, value ranges, or named
# algorithms (no "reputation", no "leading-eight", no "IS/SS/..." etc.).
# =======================================================================

INIT_PROMPT_V3 = """You are writing a Python class named `LLMAgent` that
participates in a multi-agent social-dynamics simulation. The class
decides what one agent does each round.

The simulation:
  - {population_size} agents play {num_rounds_per_gen} rounds per
    generation, for {num_generations} generations.
  - Each round, the {population_size} agents are randomly partitioned
    into {num_pairs} pairs.
  - In each pair, BOTH agents SIMULTANEOUSLY choose to either
    "cooperate" or "defect". The joint action is observed by the
    players themselves and by every other agent in the population.
    For each observed joint action, the framework calls
    `observe(...)` on every agent that was not part of the pair, and
    the players themselves are also notified via the same
    `observe(...)` call. Self-judgment is detected by checking
    `donor_id == self.agent_id` or `recipient_id == self.agent_id`.

The class interface (REQUIRED):

```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        ...

    def decide(self) -> bool:
        # Return True to cooperate, False to defect. The framework
        # sets `self._ctx_opponent_id` to the opponent's integer ID
        # before this method is called.
        ...

    def observe(
        self,
        donor_id: int,
        donor_action: str,        # 'cooperate' or 'defect'
        recipient_id: int,
        recipient_action: str,    # 'cooperate' or 'defect'
    ) -> None:
        ...
```

Rules:
  1. The class MUST be named `LLMAgent` (exact spelling).
  2. The class MAY import `math` and `random`. No other imports.
  3. The class should not crash on any input. Wrap risky code in
     try/except if needed.

Output ONLY the Python code for the class. No prose, no markdown
fences. The code must define exactly one class named `LLMAgent`.
"""


MUTATION_PROMPT_V3 = """You are mutating an existing strategy for a
multi-agent social-dynamics simulation. The parent is a Python class
named `LLMAgent`. Produce a child class also named `LLMAgent` that
behaves similarly but with at least one change.

The simulation summary:
  - {population_size} agents, {num_rounds_per_gen} rounds per
    generation, {num_generations} generations total.
  - Random pairing, simultaneous cooperation/defection choice.
  - Every joint action is observed by both players and by every
    third-party agent. The framework calls `observe(...)` on every
    observer with the same (donor_id, donor_action, recipient_id,
    recipient_action) tuple. Self-judgment is detected via
    `donor_id == self.agent_id` or `recipient_id == self.agent_id`.

The class interface (REQUIRED, unchanged):

```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        ...

    def decide(self) -> bool:
        # self._ctx_opponent_id is set by the framework just before.
        ...

    def observe(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        ...
```

PARENT (fitness {fitness}):

```python
{parent_code}
```

Output ONLY the Python code for the child class, no prose, no
markdown fences. The child must define exactly one class named
`LLMAgent` and use the same interface.
"""


# Used in the Fermi 1-μ path: produce a child that is a SMALL variant
# of the parent. The parent code IS shown to the LLM (this is the
# whole point of "imitate with tiny mutation" — the offspring is
# recognizably the parent's strategy with a small perturbation).
# Contrast with the μ path, which uses INIT_PROMPT_V3 with NO
# reference to j.
SMALL_MUTATION_PROMPT_V3 = """Create a variation of a Python class named `LLMAgent`
that participates in a multi-agent social-dynamics simulation. A
parent implementation and its realized fitness are shown below. Produce a
related child implementation while preserving the required interface.

The class interface (REQUIRED):

```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        ...

    def decide(self) -> bool:
        # self._ctx_opponent_id is set by the framework just before
        # decide() runs.
        ...

    def observe(
        self,
        donor_id: int,
        donor_action: str,        # 'cooperate' or 'defect'
        recipient_id: int,
        recipient_action: str,    # 'cooperate' or 'defect'
    ) -> None:
        ...
```

PARENT FITNESS: {fitness:.3f}

PARENT:

```python
{parent_code}
```

Output ONLY the Python code for the child class, no prose, no
markdown fences. The child must define exactly one class named
`LLMAgent` and use the same interface.
"""

DELIBERATE_MUTATION_PROMPT_V3 = """Improve a Python class named `LLMAgent`
for a multi-agent social-dynamics simulation. The parent implementation and
its realized fitness are shown below. Infer a plausible weakness, then produce
a child intended to achieve higher fitness in the same environment. Preserve
useful behavior and the required interface. Do not claim success; selection
will evaluate the child in the next generation.

The class interface (REQUIRED):

```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None: ...
    def decide(self) -> bool: ...
    def observe(self, donor_id: int, donor_action: str,
                recipient_id: int, recipient_action: str) -> None: ...
```

PARENT FITNESS: {fitness:.3f}

PARENT:

```python
{parent_code}
```

Output ONLY the Python code for the child class, no prose or markdown fences.
The child must define exactly one class named `LLMAgent`.
"""


# v2 (legacy) small-mutate: same interface as INIT_PROMPT_V2. The
# v2 quantitative interface is two free functions, not a class.
SMUTATION_PROMPT_V2 = """Create a variation of an existing strategy for a 2-player
Prisoner's Dilemma game with reputation. The parent is two functions,
`observe` and `decide`. The parent and its realized fitness are shown below.
Produce a related child using the same interface. Output ONLY the Python code,
no prose or fences.

Interface:

```python
def observe(
    A_rep: float,       # observer's current rating of player A (target)
    A_action: str,      # A's last action: 'cooperate' or 'defect'
    B_rep: float,       # observer's current rating of player B (partner)
    B_action: str,      # B's last action: 'cooperate' or 'defect'
    my_reputation: float,
) -> float:
    # Return A's NEW reputation only (a single float).
    pass

def decide(my_reputation: float, opponent_reputation: float) -> bool:
    pass
```

`observe` is ONE-DIRECTIONAL: it judges only player A and returns A's
new reputation. The framework calls it twice per joint action (once
with A=donor, once with A=recipient) to update both players, so write
one judging rule for a single target.

PARENT FITNESS: {fitness:.3f}

```python
{parent_code}
```
"""

DELIBERATE_MUTATION_PROMPT_V2 = """Improve an existing strategy for a 2-player
Prisoner's Dilemma game with reputation. The parent consists of `observe` and
`decide`, and its realized fitness is shown below. Infer a plausible weakness,
then produce a child intended to achieve higher fitness in the same environment.
Preserve useful behavior and the required interface. Do not claim success;
selection will evaluate the child in the next generation.

Interface:

```python
def observe(A_rep: float, A_action: str, B_rep: float, B_action: str,
            my_reputation: float) -> float: ...
def decide(my_reputation: float, opponent_reputation: float) -> bool: ...
```

`observe` is ONE-DIRECTIONAL: it judges only player A and returns A's
new reputation (a single float). The framework calls it twice per joint
action (once with A=donor, once with A=recipient) to update both
players, so write one judging rule for a single target.

PARENT FITNESS: {fitness:.3f}

PARENT:

```python
{parent_code}
```

Output ONLY the Python code, no prose or markdown fences.
"""
