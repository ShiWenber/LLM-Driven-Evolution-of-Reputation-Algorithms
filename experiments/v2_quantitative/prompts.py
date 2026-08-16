"""Prompts for the v2 quantitative interface LLM evolution (2-player PD).

The interface is:
    observe(donor_reputation, donor_action,
                    recipient_reputation, recipient_action,
                    my_reputation) -> tuple[float, float] | dict
  decide(my_reputation, opponent_reputation) -> bool

The game is a 2-player simultaneous Prisoner's Dilemma. Each round,
two matched agents independently choose C (cooperate) or D (defect);
payoffs are (C,C) -> both +1, (C,D) -> C gets -1 / D gets +2, etc.
This matches the leading-eight / quantitative-assessment literature
extended to symmetric interactions.
"""


INIT_PROMPT_V2 = """You are a Python programmer designing a strategy for a 2-player
Prisoner's Dilemma game with reputation. The game has 15 agents that play
30 rounds per generation, 30 generations total. Each round, the 15
agents are randomly paired (7 pairs + 1 unpaired per round). In each
pair, BOTH agents simultaneously choose to 'cooperate' (give benefit=2
to the partner at cost=1 to self) or 'defect' (no exchange). Payoffs:
  (C, C) -> each gets 2 - 1 = 1
  (C, D) -> C gets -1 (sucker's payoff), D gets +2
  (D, C) -> symmetric
  (D, D) -> each gets 0

After each joint action, the framework records both actions and the
agents' IDs. It then calls each observer's `observe` ONCE to update the
observer's private rating of BOTH players in that joint action.

Your strategy consists of TWO functions. Define BOTH.

The interface is the leading-eight / quantitative-assessment style
extended to symmetric 2-player PD:

```python
def observe(
    donor_reputation: float,      # observer's current rating of donor
    donor_action: str,            # donor's last action: 'cooperate' or 'defect'
    recipient_reputation: float,  # observer's current rating of recipient
    recipient_action: str,        # recipient's last action: 'cooperate' or 'defect'
    my_reputation: float          # observer's own self-rating
) -> tuple[float, float]:
    # Return (new_donor_reputation, new_recipient_reputation).
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
  2. observe() is called once per observed joint action with BOTH
      players. There is no separate "self-evaluation" function; if the
      observer is one of the two players in the joint action, the
      framework calls observe() the same way.
  3. Reputation is in [-1.0, 1.0]. Treat 0.0 as neutral.
  4. The functions will be called many times; keep them deterministic or
     near-deterministic (no random.random unless you really want it).
  5. Use only Python builtins; no imports other than `math` (already
     available) and `random` (NOT recommended).

Generate ONE strategy pair. Output ONLY the Python code, no prose,
no markdown fences. The code must define `observe` and `decide`.
"""


MUTATION_PROMPT_V2 = """You are mutating an existing strategy for a 2-player
Prisoner's Dilemma game with reputation. The game has 15 agents, 30
rounds per generation, 30 generations.

The interface is leading-eight / quantitative-assessment style
extended to symmetric 2-player PD:

```python
def observe(
    donor_reputation: float,
    donor_action: str,         # 'cooperate' or 'defect'
    recipient_reputation: float,
    recipient_action: str,     # 'cooperate' or 'defect'
    my_reputation: float
) -> tuple[float, float]:
    pass

def decide(
    my_reputation: float,
    opponent_reputation: float
) -> bool:
    pass
```

`observe` is called once per observed joint action with donor and
recipient together. Use one joint judging rule for both sides.

The parent strategy to mutate is below.

PARENT (fitness {fitness}):

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
  - 15 agents play 30 rounds per generation, for 30 generations.
  - Each round, the 15 agents are randomly partitioned into pairs (7
    pairs; one agent sits out if 15 is odd).
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

The simulation summary (do not propose structural changes to the
simulation itself):
  - 15 agents, 30 rounds per generation, 30 generations total.
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
SMALL_MUTATION_PROMPT_V3 = """You are rewriting a Python class named `LLMAgent`
that participates in a multi-agent social-dynamics simulation. A
parent implementation is shown below. Produce a new implementation of
the same class.

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

PARENT:

```python
{parent_code}
```

Output ONLY the Python code for the child class, no prose, no
markdown fences. The child must define exactly one class named
`LLMAgent` and use the same interface.
"""


# v2 (legacy) small-mutate: same interface as INIT_PROMPT_V2. The
# v2 quantitative interface is two free functions, not a class.
SMUTATION_PROMPT_V2 = """You are mutating an existing strategy for a 2-player
Prisoner's Dilemma game with reputation. The parent is two functions,
`observe` and `decide`. Produce a child that is a SMALL variant of
the parent — adjust a single number or threshold, do NOT rewrite the
logic. Output ONLY the Python code, no prose, no fences.

Interface:

```python
def observe(
    donor_reputation: float,
    donor_action: str,
    recipient_reputation: float,
    recipient_action: str,
    my_reputation: float,
) -> tuple[float, float]:
    pass

def decide(my_reputation: float, opponent_reputation: float) -> bool:
    pass
```

PARENT (fitness {fitness}):

```python
{parent_code}
```
"""
