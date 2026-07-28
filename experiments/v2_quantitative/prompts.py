"""Prompts for the v2 quantitative interface LLM evolution.

The interface is intentionally minimal:
  evaluate(donor_reputation, recipient_reputation, donor_action, recipient_action, my_reputation) -> float
  decide(my_reputation, opponent_reputation) -> bool

This matches the leading-eight / quantitative-assessment literature
(Ohtsuki-Iwasa 2006, Schmid 2023). Strategies like Image Scoring, Simple
Standing, Judging, IS+ can all be expressed.
"""


INIT_PROMPT_V2 = """You are a Python programmer designing a strategy for an indirect-reciprocity
donor game. The game has 15 agents that play 30 rounds per generation,
30 generations total. Each round, every agent is the donor once with a
random recipient. Donor can choose to cooperate (give benefit=2 to
recipient at cost=1 to self) or defect (no exchange).

Your strategy consists of TWO functions. Define BOTH.

The interface is the same as in the leading-eight / quantitative-assessment
literature (Ohtsuki-Iwasa 2006, Schmid 2023). Real-valued reputation in
[-1.0, +1.0]. Neutral default is 0.0.

```python
def evaluate(
    donor_reputation: float,        # observer's current rating of the donor
    recipient_reputation: float,    # observer's current rating of the recipient
    donor_action: str,              # 'cooperate' or 'defect'
    recipient_action: str,          # 'cooperate' or 'defect'
    my_reputation: float            # observer's own self-rating
) -> float:
    # Return the NEW donor_reputation after this observation.
    # Will be clamped to [-1.0, 1.0].
    pass

def decide(
    my_reputation: float,
    opponent_reputation: float
) -> bool:
    # Return True to donate, False to defect.
    pass
```

Important rules:
  1. Both functions MUST be defined.
  2. evaluate() is called twice per third-party observation:
       (a) on the observer, with donor=observed_donor, my=observer's self-rating
           -> updates the OBSERVER's rating of the donor
       (b) on the donor themselves, with donor=donor, my=donor's self-rating
           -> updates the DONOR's self-rating
     The same function must work for both. There is no separate
     "self-evaluation" function. If you want to know "the donor's
     previous self-rating", use my_reputation.
  3. Reputation is in [-1.0, 1.0]. Treat 0.0 as neutral.
  4. The function will be called many times; keep it deterministic or
     near-deterministic (no random.random unless you really want it).
  5. Use only Python builtins; no imports other than `math` (already
     available) and `random` (NOT recommended).

Generate ONE strategy pair. Output ONLY the Python code, no prose,
no markdown fences. The code must define `evaluate` and `decide`.
"""


MUTATION_PROMPT_V2 = """You are mutating an existing strategy for an indirect-reciprocity
donor game. The game has 15 agents, 30 rounds per generation, 30 generations.

The interface is real-valued, in the leading-eight / quantitative-assessment style:

```python
def evaluate(
    donor_reputation: float,
    recipient_reputation: float,
    donor_action: str,        # 'cooperate' or 'defect'
    recipient_action: str,    # 'cooperate' or 'defect'
    my_reputation: float
) -> float:
    pass

def decide(
    my_reputation: float,
    opponent_reputation: float
) -> bool:
    pass
```

The same `evaluate` is called twice per third-party observation:
  (a) on the OBSERVER, with donor=observed_donor, my=observer's self-rating
  (b) on the DONOR, with donor=donor, my=donor's self-rating
Use the same judging rule for both — there is no separate function
for self vs others.

The parent strategy to mutate is below.

PARENT (fitness {fitness}):

```python
{parent_code}
```

Generate a child strategy. Suggestions (you don't have to follow any):
  - Slightly increase or decrease the step size (e.g., 0.3 vs 0.4)
  - Add or remove dependence on `my_reputation`
  - Add or remove dependence on `recipient_reputation` (this is the
    leading-eight discriminator; many leading-eight norms do depend on it)
  - Adjust the threshold for decide()
  - Add an asymmetry: cooperate gives a smaller step than defect punishes

Output ONLY the Python code, no prose, no markdown fences.
"""
