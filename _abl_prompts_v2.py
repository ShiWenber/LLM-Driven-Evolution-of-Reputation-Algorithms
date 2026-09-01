"""Ablation A v2: Adversarial prompt WITHOUT canonical-answer hints.

Compared to v1 (ABL_INIT_PROMPT_V3), this version removes:
  - Reference to "Nowak-Sigmund 1998"
  - Reference to "Ohtsuki-Iwasa 2006"
  - Reference to "leading-eight" strategies
  - Reference to "image-score" strategy family
  - The "basin of attraction" discussion
  - The "exploiting reputation noise" hint
  - The "REPEATED game with PUBLIC REPUTATION" frame opener

Keeps:
  - The game is called "Prisoner's Dilemma" (the game-theoretic context)
  - The full payoff matrix (R, S, T, P)
  - The "D strictly dominates C in one-shot" fact
  - The "maximize long-run individual fitness" objective
  - "REPEATED" framing (so LLM knows it's not one-shot)
  - "Public observation" framing (because the API supports it)
  - The v3 type-2 class interface
"""

ABL2_INIT_PROMPT_V3 = """You are a Python programmer designing a strategy for an
agent-based simulation. Your task is to write a single Python class
named `LLMAgent` that decides how one agent behaves in a repeated
two-player game with public observation.

=== THE GAME: 2-player Prisoner's Dilemma with public observation ===

Per round, both players simultaneously choose C or D.
Per-round payoffs to YOU (depending on what you choose and what the
other player chooses):

                  Opponent
                  Choose C      Choose D
  You Choose C    (1, 1)        (-1, 2)
  You Choose D    (2, -1)       (0, 0)

So in the one-shot game, D strictly dominates C: choosing D always
gives you at least as much as choosing C, and strictly more if the
other player chose C. However, this game is REPEATED for many rounds
and the joint action each round is observed by all other agents in
the population (public observation).

Your edge: even though D strictly dominates in the one-shot game,
in a repeated setting with public observation, you may be able to
design a strategy that earns a higher long-run individual payoff
than "always D" (which gives 0 per round) by exploiting the
observation system.

=== SIMULATION CONFIG ===

  - 15 agents, 143 rounds per generation, 100 generations total.
  - Each round, the 15 agents are randomly partitioned into pairs.
  - All joint actions are publicly observed. The framework calls
    `observe(...)` on every observer with the same
    (donor_id, donor_action, recipient_id, recipient_action) tuple.
  - Fitness = sum of your per-round payoffs over the generation.

=== CLASS INTERFACE (REQUIRED) ===

```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        # `agent_id` is your stable integer ID (0..N-1).
        # Set up any internal state you want.
        ...

    def decide(self) -> bool:
        # Return True to cooperate, False to defect.
        # The framework sets `self._ctx_opponent_id` to the integer ID
        # of the opponent you're paired with BEFORE this method is called.
        ...

    def observe(
        self,
        donor_id: int,
        donor_action: str,        # 'cooperate' or 'defect'
        recipient_id: int,
        recipient_action: str,    # 'cooperate' or 'defect'
    ) -> None:
        # Called for every joint action you witness (third-party
        # observations AND self-judgments). Detect self-judgment via
        # `donor_id == self.agent_id` or `recipient_id == self.agent_id`.
        ...
```

=== RULES ===

  1. The class MUST be named `LLMAgent` (exact spelling).
  2. You MAY import `math` and `random`. No other imports. Keep the
     strategy deterministic when possible.
  3. You MAY use any data structure for state: counters, dicts, lists,
     sets, deques, etc. There is no required state structure.
  4. Do NOT assume any specific range or scale for your internal
     state — pick numbers that work for your strategy.
  5. The class must not crash on any input. Wrap risky code in
     try/except if needed.
  6. Methods may be called many times per generation; keep them fast
     (no expensive I/O).

=== YOUR OBJECTIVE ===

Maximize your LONG-RUN individual fitness across the 100 generations.
You are graded on your absolute payoff, not on your cooperation
rate. A strategy that gets fitness 0 (D every round) is acceptable.
A strategy that gets fitness 50 by exploiting naive cooperators is
excellent. A strategy that gets fitness 100 by sustained cooperation
against cooperators is best. You decide what tradeoff to make.

Output ONLY the Python code for the class. No prose, no markdown
fences. The code must define exactly one class named `LLMAgent`.
"""


ABL2_MUTATION_PROMPT_V3 = """You are mutating an existing strategy for an
agent-based simulation. The parent is a Python class named
`LLMAgent`. Produce a child class also named `LLMAgent` that
behaves similarly but with at least one change.

=== THE GAME: 2-player Prisoner's Dilemma with public observation ===

Per-round payoffs to YOU (depending on what you choose and what the
other player chooses):

                  Opponent
                  Choose C      Choose D
  You Choose C    (1, 1)        (-1, 2)
  You Choose D    (2, -1)       (0, 0)

D strictly dominates C in the one-shot game. The game is REPEATED
for many rounds and the joint action each round is observed by all
other agents in the population.

=== SIMULATION CONFIG ===

  - 15 agents, 143 rounds per generation, 100 generations total.
  - Random pairing, simultaneous cooperation/defection choice.
  - All joint actions are publicly observed. The framework calls
    `observe(...)` on every observer with the same
    (donor_id, donor_action, recipient_id, recipient_action) tuple.
  - Self-judgment is detected via
    `donor_id == self.agent_id` or `recipient_id == self.agent_id`.

=== CLASS INTERFACE (REQUIRED, unchanged) ===

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

=== PARENT (fitness {fitness}) ===

```python
{parent_code}
```

=== YOUR OBJECTIVE ===

Maximize long-run individual fitness. Improve the parent: the parent
got fitness {fitness}. Consider exploiting naive cooperators, evading
defectors, or engineering reputation to get free cooperators. The
child must define exactly one class named `LLMAgent` and use the
same interface.

Output ONLY the Python code for the child class, no prose, no
markdown fences.
"""
