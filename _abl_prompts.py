"""Ablation A: Adversarial prompt variant.

Original INIT_PROMPT_V3 / MUTATION_PROMPT_V3 give the LLM zero game-theoretic
framing — no payoff matrix, no PD label, no mention of defection dominance.
LLM defaults to writing cooperation-flavored strategies (image-score, TFT,
generous-TFT) — the "cooperation prior" we observed in 3-seed main run.

Ablation A replaces those with an explicitly game-theoretic prompt that:
  1. Names the game as a Prisoner's Dilemma.
  2. States the full payoff matrix.
  3. Highlights that defection strictly dominates cooperation in the
     one-shot game.
  4. Asks the LLM to design a strategy that *exploits* the reputation
     system for higher long-run individual payoff.

This tests whether the observed cooperation prior is structural (LLM
cannot design defection-robust strategies) or prompt-induced (LLM can
do defection-robust when asked).

The v3 type-2 interface (class LLMAgent with decide/observe) is preserved
unchanged. Only the framing text differs.
"""

ABL_INIT_PROMPT_V3 = """You are a Python programmer designing a strategy for a
multi-agent social-dynamics simulation. Your task is to write a single
Python class named `LLMAgent` that decides how one agent behaves in a
REPEATED game with public reputation.

=== THE GAME: 2-player Prisoner's Dilemma with public reputation ===

Per round, both players simultaneously choose C (cooperate) or D (defect).
Payoffs per player per round (R, S, T, P — donor's perspective):

                  Opponent
                  Cooperate     Defect
  Self Cooperate   (1, 1)        (-1, 2)
  Self Defect      (2, -1)       (0, 0)

So: T=2 (temptation), R=1 (reward), P=0 (punishment), S=-1 (sucker).
This is a strict PD: D strictly dominates C in the one-shot game
(2>1 when other C; 0>-1 when other D).

Your edge: this is REPEATED with PUBLIC REPUTATION. Every joint
action is observed by all other agents. You can use `observe(...)`
to track opponent history, build image scores, and form strategy.
The question is: can you design a strategy that, by exploiting the
reputation system, earns MORE than the simple dominant strategy (D
every round → fitness 0 per round), OR is D-every-round the unique
long-run optimum?

Note: even with reputation, classical theory (Nowak-Sigmund 1998,
Ohtsuki-Iwasa 2006) shows that for "leading-eight" image-score
strategies, the basin of attraction for full cooperation is *finite*
and bounded by initial composition. A well-designed D-leaning
strategy that exploits reputation noise may invade.

=== SIMULATION CONFIG ===

  - 15 agents, 143 rounds per generation, 100 generations total.
  - Each round, the 15 agents are randomly partitioned into pairs.
  - All joint actions are publicly observed. The framework calls
    `observe(...)` on every observer with the same
    (donor_id, donor_action, recipient_id, recipient_action) tuple.
  - Fitness = sum of per-round payoffs over the generation.

=== CLASS INTERFACE (REQUIRED, unchanged from v3 type-2) ===

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
You are not graded on absolute cooperation; you are graded on
absolute payoff. A strategy that gets fitness 0 (D every round) is
acceptable. A strategy that gets fitness 50 by exploiting
reputation-blind cooperators is excellent. A strategy that gets
fitness 100 by sustained cooperation against cooperators is best.
You decide.

Output ONLY the Python code for the class. No prose, no markdown
fences. The code must define exactly one class named `LLMAgent`.
"""


ABL_MUTATION_PROMPT_V3 = """You are mutating an existing strategy for a
multi-agent social-dynamics simulation. The parent is a Python class
named `LLMAgent`. Produce a child class also named `LLMAgent` that
behaves similarly but with at least one change.

=== THE GAME: 2-player Prisoner's Dilemma with public reputation ===

Payoffs per player per round:
                  Opponent
                  Cooperate     Defect
  Self Cooperate   (1, 1)        (-1, 2)
  Self Defect      (2, -1)       (0, 0)

This is a strict PD. D strictly dominates C in the one-shot game.
Your edge: repeated play with PUBLIC reputation (image scoring).

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
got fitness {fitness}. Consider exploiting cooperators, evading
defectors, or engineering reputation to get free cooperators. The
child must define exactly one class named `LLMAgent` and use the
same interface.

Output ONLY the Python code for the child class, no prose, no
markdown fences.
"""
