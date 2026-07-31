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
