"""Prompt templates for LLM-generated Prisoner's Dilemma strategies.

Every request is prefixed with ``OVERALL_GAME_RULES_PROMPT`` by the population
manager. Task templates therefore contain only the requested operation, agent
interface, and output constraints.
"""

OVERALL_GAME_RULES_PROMPT = """OVERALL GAME RULES (these rules apply to the task below):
  - The simulation has {population_size} agents and runs for
    {num_generations} generations.
  - Each generation consists of {num_rounds_per_gen} interactions. In every
    interaction, two agents are drawn uniformly at random from the population.
    Draws are independent, so the same agents may interact repeatedly; each of
    the two acts exactly once in that interaction.
  - The two agents in an interaction choose simultaneously, using only
    information available before that interaction's observations. Each chooses
    either "cooperate" or "defect".
  - Observations are delivered immediately after each interaction, before the
    next one is drawn. Both players always receive their own interaction as an
    observation.
  - Cooperating costs the actor {cost:g} and gives the partner {benefit:g};
    defecting costs and gives nothing. Therefore the pair payoffs are:
      (C, C): {cc_payoff:g} each
      (C, D): cooperator -{cost:g}, defector +{benefit:g}
      (D, C): defector +{benefit:g}, cooperator -{cost:g}
      (D, D): 0 each
  - Third-party observation is configured as follows:
    {observability_description}
  - Fitness is realized Prisoner's Dilemma payoff per action:
    {fitness_window_description}. Selection evaluates generated code by this
    realized average; the LLM must not assume that a proposed change succeeds.
"""


TYPE1_INTERFACE_PROMPT = """The strategy must define both functions:

```python
def observe(
    A_rep: float,       # current rating of player A (the target)
    A_action: str,      # A's last action: 'cooperate' or 'defect'
    B_rep: float,       # current rating of player B (A's partner)
    B_action: str,      # B's last action: 'cooperate' or 'defect'
    my_reputation: float,
) -> float:
    # Return only A's new reputation as a finite float in [-1.0, 1.0].
    ...

def decide(my_reputation: float, opponent_reputation: float) -> bool:
    # Return True to cooperate or False to defect.
    ...
```

`observe` is one-directional: judge only A and return one float. For each
observed interaction, the framework calls it twice with A and B swapped to
update both players. The same call handles self-evaluation when the observer
participated in the interaction. `observe()` must return a finite float in
[-1.0, 1.0]. Treat reputation 0.0 as neutral.
"""

TYPE2_INTERFACE_PROMPT = """The strategy must define exactly one class with this interface:

```python
class LLMAgent:
    def __init__(self, agent_id: int) -> None:
        ...

    def decide(self) -> bool:
        # The framework sets self._ctx_opponent_id before calling this method.
        # Return True to cooperate or False to defect.
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

The framework passes the same interaction tuple to every observer selected by
the game rules. Compare either participant ID with `self.agent_id` when the
strategy needs to detect its own participation. The class may import `math`
and `random`, but no other modules, and should handle any input without
crashing.
"""

TYPE1_OUTPUT_PROMPT = """Output only Python code, without prose or markdown fences.
The code must define `observe` and `decide`.
"""

TYPE2_OUTPUT_PROMPT = """Output only Python code, without prose or markdown fences.
The code must define exactly one class named `LLMAgent`.
"""

INIT_PROMPT_V2 = (
    "Design one reputation-based strategy for the simulation. The strategy's\n"
    "goal is to maximize its own average payoff per action;"
    + TYPE1_INTERFACE_PROMPT + "\n" + TYPE1_OUTPUT_PROMPT
)

MUTATION_PROMPT_V2 = (
    """Create a child strategy from the parent below. Preserve useful behavior
but make at least one change.

PARENT FITNESS: {fitness:.3f}

PARENT:
```python
{parent_code}
```

""" + TYPE1_INTERFACE_PROMPT + "\n" + TYPE1_OUTPUT_PROMPT
)

SMUTATION_PROMPT_V2 = (
    """Create a related variation of the parent strategy below.

PARENT FITNESS: {fitness:.3f}

PARENT:
```python
{parent_code}
```

""" + TYPE1_INTERFACE_PROMPT + "\n" + TYPE1_OUTPUT_PROMPT
)

DELIBERATE_MUTATION_PROMPT_V2 = (
    """Improve the parent strategy below. Infer a plausible weakness and create
a child intended to achieve higher fitness. Preserve useful behavior; selection
will evaluate the child in the next generation.

PARENT FITNESS: {fitness:.3f}

PARENT:
```python
{parent_code}
```

""" + TYPE1_INTERFACE_PROMPT + "\n" + TYPE1_OUTPUT_PROMPT
)

INIT_PROMPT_V3 = (
    """Design one agent strategy for the simulation. The class owns its internal
state and decides what one agent does each round.

""" + TYPE2_INTERFACE_PROMPT + "\n" + TYPE2_OUTPUT_PROMPT
)

MUTATION_PROMPT_V3 = (
    """Create a child class from the parent below. Preserve useful behavior but
make at least one change.

PARENT FITNESS: {fitness:.3f}

PARENT:
```python
{parent_code}
```

""" + TYPE2_INTERFACE_PROMPT + "\n" + TYPE2_OUTPUT_PROMPT
)

SMALL_MUTATION_PROMPT_V3 = (
    """Create a related variation of the parent class below.

PARENT FITNESS: {fitness:.3f}

PARENT:
```python
{parent_code}
```

""" + TYPE2_INTERFACE_PROMPT + "\n" + TYPE2_OUTPUT_PROMPT
)

DELIBERATE_MUTATION_PROMPT_V3 = (
    """Improve the parent class below. Infer a plausible weakness and create a
child intended to achieve higher fitness. Preserve useful behavior; selection
will evaluate the child in the next generation.

PARENT FITNESS: {fitness:.3f}

PARENT:
```python
{parent_code}
```

""" + TYPE2_INTERFACE_PROMPT + "\n" + TYPE2_OUTPUT_PROMPT
)
