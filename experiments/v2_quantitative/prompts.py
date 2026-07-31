"""Prompts for the v2 quantitative interface LLM evolution (2-player PD).

The interface is:
  evaluate(target_reputation, target_action, my_reputation) -> float
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
agents' IDs. It then calls each observer's `evaluate` to update the
observer's private rating of EACH player in the joint action (twice per
joint action per observer: once for the donor, once for the
recipient, using each player's own action as `target_action`).

Your strategy consists of TWO functions. Define BOTH.

The interface is the leading-eight / quantitative-assessment style
extended to symmetric 2-player PD:

```python
def evaluate(
    target_reputation: float,    # observer's current rating of the target being judged
    target_action: str,           # target's last action: 'cooperate' or 'defect'
    my_reputation: float          # observer's own self-rating
) -> float:
    # Return the NEW target_reputation after observing target's action.
    # Will be clamped to [-1.0, 1.0].
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
  2. evaluate() is called twice per observed joint action: once for the
     donor (with target_action=donor's action) and once for the
     recipient (with target_action=recipient's action). The same
     function handles both. There is no separate "self-evaluation"
     function; if the observer is one of the two players in the joint
     action, the framework calls evaluate() for the observer's own
     rating update just like for any other target.
  3. Reputation is in [-1.0, 1.0]. Treat 0.0 as neutral.
  4. The functions will be called many times; keep them deterministic or
     near-deterministic (no random.random unless you really want it).
  5. Use only Python builtins; no imports other than `math` (already
     available) and `random` (NOT recommended).

Generate ONE strategy pair. Output ONLY the Python code, no prose,
no markdown fences. The code must define `evaluate` and `decide`.
"""


MUTATION_PROMPT_V2 = """You are mutating an existing strategy for a 2-player
Prisoner's Dilemma game with reputation. The game has 15 agents, 30
rounds per generation, 30 generations.

The interface is leading-eight / quantitative-assessment style
extended to symmetric 2-player PD:

```python
def evaluate(
    target_reputation: float,
    target_action: str,        # 'cooperate' or 'defect'
    my_reputation: float
) -> float:
    pass

def decide(
    my_reputation: float,
    opponent_reputation: float
) -> bool:
    pass
```

`evaluate` is called twice per observed joint action: once for the
donor (target_action=donor's action) and once for the recipient
(target_action=recipient's action). Use the same judging rule for
both — there is no separate function for self vs others.

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

INIT_PROMPT_V3 = """You are a Python programmer designing a strategy for a
multi-agent social-dynamics simulation. Your task is to write a single
Python class named `LLMAgent` that decides how one agent behaves.

The simulation:
  - 15 agents play 30 rounds per generation, for 30 generations.
  - Each round, the 15 agents are randomly partitioned into pairs (7
    pairs; one agent sits out if 15 is odd).
  - In each pair, BOTH agents SIMULTANEOUSLY choose to either
    "cooperate" or "defect". The joint action is observed by the
    players themselves and by every other agent in the population
    (third-party observers). For each observed joint action, the
    framework calls `observe(...)` on every agent that was not part
    of the pair, and the players themselves are also notified via
    the same `observe(...)` call (i.e. self-judgment is NOT a
    separate event — you detect it by checking
    `donor_id == self.agent_id` or `recipient_id == self.agent_id`).

The class interface (REQUIRED):

```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        # `agent_id` is this agent's stable integer ID (0..N-1, never
        # reused). Set any internal state you want here.
        ...

    def decide(self) -> bool:
        # Return True to cooperate, False to defect. Called once per
        # round for this agent. The framework sets
        # `self._ctx_opponent_id` to the integer ID of the opponent
        # you're paired with BEFORE this method is called; you can
        # read it but you don't have to use it.
        ...

    def observe(
        self,
        donor_id: int,
        donor_action: str,        # 'cooperate' or 'defect'
        recipient_id: int,
        recipient_action: str,    # 'cooperate' or 'defect'
    ) -> None:
        # Called for every joint action this agent witnesses (both
        # third-party observations and self-judgments). Update any
        # internal state here. Detecting self-judgment: check
        # `donor_id == self.agent_id` or `recipient_id == self.agent_id`.
        ...
```

Important rules:
  1. The class MUST be named `LLMAgent` (exact spelling).
  2. The class MAY import `math` (already available) and `random`. No
     other imports. Avoid `random` unless you have a good reason; keep
     the strategy deterministic when possible.
  3. The class MAY use any data structure you want for state:
     counters, dicts, lists, sets, deques, etc. There is no required
     state structure.
  4. Do NOT assume any range or scale for your internal state — pick
     numbers that work for your strategy.
  5. The class should not crash on any input. Wrap risky code in
     try/except if needed.
  6. Methods may be called many times per generation; keep them
     fast (no expensive I/O).

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
