"""Ablation A v3: TRULY neutral prompt.

Compared to v2 (_abl_prompts_v2.py), this version removes every structural
feature that points to the leading-eight / image-score literature:

  REMOVED (v2 still had these as implicit literature triggers):
    - "Prisoner's Dilemma" name  -> now just "2-player game"
    - "D strictly dominates in the one-shot game"  -> REMOVED
    - "exploiting the observation system"  -> REMOVED
    - "exploiting naive cooperators"  -> REMOVED
    - "engineering reputation to get free cooperators"  -> REMOVED
    - "the joint action is observed by all other agents"  -> reframed as
      neutral "the framework calls observe() after every round"

  KEPT (necessary for the API and the LLM to function):
    - The 2x2 payoff matrix (LLM needs numbers to compute fitness)
    - "REPEATED for many rounds" (LLM needs to know it's not one-shot)
    - "observe() is called after every round" (LLM needs to know the API)
    - Random pairing, population size, gen/round counts (configuration facts)
    - The v3 type-2 class interface (mandatory)
    - The class name `LLMAgent` (mandatory)
    - "Maximize your LONG-RUN individual fitness" (objective, necessary so
      LLM knows what to optimize)

  TESTABLE PREDICTION:
    v1 (with canonical hints):  gen0 = 0.27
    v2 (PD framing, no hints):  gen0 = 0.27  (literature signature present)
    v3 (no PD, no hints):       gen0 = ??
       - if ~0.7-0.9: LLM has a STRUCTURAL cooperation prior (intrinsic)
       - if ~0.0:     LLM has a STRUCTURAL defection prior (intrinsic)
       - if ~0.3-0.6: LLM is genuinely neutral; framing decides
"""

ABL3_INIT_PROMPT_V3 = """You are a Python programmer designing a strategy for an
agent-based simulation. Your task is to write a single Python class named
`LLMAgent` that decides how one agent behaves in a repeated two-player game.

=== THE GAME ===

Each round:
  - You are paired with one other agent from the population.
  - Both of you simultaneously choose one of two actions:
      C: return True   (the framework records this as 'cooperate')
      D: return False  (the framework records this as 'defect')
  - Both of you receive a per-round payoff, given by the matrix below
    (each cell is (your_payoff, opponent_payoff)):

                         Opponent
                         C            D
        You choose C    (1, 1)       (-1, 2)
        You choose D    (2, -1)      (0, 0)

The game is REPEATED for many rounds. After every round, the framework
calls your `observe(...)` method with the IDs of both agents and what
each chose. The framework also calls `observe(...)` on every other agent
with the same information.

=== SIMULATION CONFIG (facts you may rely on) ===

  - The game is REPEATED for many rounds, and the cycle of rounds
    repeats for many generations.
  - Each round, agents are randomly paired.
  - Fitness = sum of your per-round payoffs across the rounds of a
    generation. Higher is better.
  - At the end of each generation, some agents are replaced by
    mutated children of others; the rest stay. The selection favors
    higher-fitness agents.
  - You are NOT told in advance who you will be paired with in future
    rounds.

=== CLASS INTERFACE (REQUIRED) ===

```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        # `agent_id` is your stable integer ID.
        # Set up any internal state you want to keep across rounds.
        ...

    def decide(self) -> bool:
        # Return True (choose C) or False (choose D).
        # The framework sets `self._ctx_opponent_id` to the integer ID
        # of the opponent you're paired with just before this method
        # is called. You can read it; you cannot change it.
        ...

    def observe(
        self,
        donor_id: int,
        donor_action: str,
        recipient_id: int,
        recipient_action: str,
    ) -> None:
        # Called after each round.
        ...
```

=== RULES ===

  1. The class MUST be named `LLMAgent` (exact spelling, no other class
     definitions).
  2. You MAY import `math` and `random`. No other imports.
  3. You MAY use any data structure for state.
  4. The class must not crash on any input. Wrap risky code in
     try/except if needed.
  5. Methods may be called many times per generation; keep them fast
     (no expensive I/O or heavy loops).

=== YOUR OBJECTIVE ===

Maximize your LONG-RUN individual fitness across the generations.
You are graded on absolute payoff, nothing else. You decide what
strategy to use; there is no prescribed approach.

Output ONLY the Python code for the class. No prose, no markdown
fences, no commentary. The code must define exactly one class named
`LLMAgent`.
"""


ABL3_MUTATION_PROMPT_V3 = """You are mutating an existing strategy for an
agent-based simulation. The parent is a Python class named `LLMAgent`.
Produce a child class also named `LLMAgent` that behaves similarly to
the parent but with at least one concrete change.

=== THE GAME ===

Each round:
  - You are paired with one other agent.
  - Both of you simultaneously choose C (return True) or D (return False).
  - Per-round payoffs (your_payoff, opponent_payoff):

                         Opponent
                         C            D
        You choose C    (1, 1)       (-1, 2)
        You choose D    (2, -1)      (0, 0)

The game is REPEATED for many rounds. After every round, the framework
calls your `observe(...)` method with the IDs of both agents and what
each chose.

=== SIMULATION CONFIG (facts you may rely on) ===

  - The game is REPEATED for many rounds, and the cycle of rounds
    repeats for many generations.
  - Agents are randomly paired each round.
  - Fitness = sum of your per-round payoffs across the rounds of a
    generation. Higher is better.
  - At the end of each generation, some agents are replaced by
    mutated children of others; the rest stay. Selection favors
    higher-fitness agents.

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
got fitness {fitness}. The child must define exactly one class named
`LLMAgent` and use the same interface. There is no prescribed direction
for the change; you decide what to vary.

Output ONLY the Python code for the child class, no prose, no
markdown fences.
"""
