"""Prompt templates for LLM code generation and mutation.

CRITICAL DESIGN PRINCIPLE: These prompts deliberately omit any mention of
"reputation", "image scoring", "standing", "trust", or strategic advice.
The LLM is told ONLY the payoff structure and the function interfaces.
It must discover through evolution that tracking others' behavior is valuable.

Each agent runs TWO functions:
1. evaluate(): update private reputation score based on observations
2. decide(): use reputation to make donation decisions

Only three LLM call types exist (no per-round calls):
1. INITIALIZATION: Generate N diverse (evaluate, decide) function pairs
2. MUTATION: Vary a successful pair to create offspring
"""


# ============================================================================
# Strategy Function Template (shown to LLM as interface specification)
# ============================================================================

STRATEGY_INTERFACE = '''def evaluate(
    current_reputation: float,
    observation: dict,
    my_history: list[dict],
    round_num: int
) -> float:
    """
    Update your private assessment of an agent after observing their action.

    Called whenever you witness an interaction involving another agent.
    You receive the agent's CURRENT reputation score and the observation,
    and return an UPDATED reputation score.

    observation dict keys:
        "round": int
        "donor": int (the agent whose action you are evaluating)
        "recipient": int (the agent receiving the action)
        "donor_reputation": float (the donor's reputation in YOUR private store;
            starts at a small positive value if you have never observed them)
        "recipient_reputation": float (the recipient's reputation in YOUR private
            store; starts at a small positive value if you have never observed them)
        "action": "cooperate" | "defect"
        (the action label is pre-named: "cooperate" is the option that costs the
         donor 1 and gives the recipient 2; "defect" is the option that costs 0
         and gives the recipient 0. These names describe the game's underlying
         semantics, not the LLM-facing vocabulary.)

    Note: the relative effects on the donor's and recipient's scores are
    deterministic from the action label; "cooperate" gives the recipient 2
    and the donor pays 1; "defect" gives nothing to either party.

    CRITICAL: "action" is a string, NOT a boolean.
    Example: if obs["action"] == "cooperate": ...

    my_history: interactions you personally participated in.
      Each entry: {"round": int, "role": "donor"|"recipient",
                   "partner": int, "action": "cooperate"|"defect",
                   "partner_action": "cooperate"|"defect"|None}

    Returns: updated reputation score (float between -1.0 and 1.0).
    """
    pass


def decide(
    recipient_reputation: float,
    round_num: int,
    my_history: list[dict]
) -> bool:
    """
    Choose which action to take against the recipient.

    "cooperate" = DONATE (the agent pays 1 and the recipient gains 2).
    "defect" = NOT DONATE (no cost, no benefit).
    These names describe the game's underlying semantics directly.

    recipient_reputation: YOUR private assessment of the recipient.
      This is the score you have built up by observing their behavior.
      If you have never observed this recipient, this value is near 0.0
      (specifically 0.01 — a small positive value to enable cold start).

    my_history: interactions you personally participated in.
      Same format as in evaluate().

    Return True to cooperate, False to defect.
    """
    pass'''

# Compact version without docstrings for batch generation
STRATEGY_INTERFACE_COMPACT = (
    "def evaluate(current_reputation: float, observation: dict, "
    "my_history: list[dict], round_num: int) -> float:\n"
    "    # Update private reputation based on observed action.\n"
    "    # observation dict keys:\n"
    "    #   'donor' (int), 'recipient' (int), 'action' ('cooperate'|'defect'),\n"
    "    #   'donor_reputation' (float, donor's standing in YOUR private store),\n"
    "    #   'recipient_reputation' (float, recipient's standing in YOUR private store),\n"
    "    #   'round' (int)\n"
    "    ...\n\n"
    "def decide(recipient_reputation: float, round_num: int, "
    "my_history: list[dict]) -> bool:\n"
    "    # Choose which action to take based on recipient's reputation\n"
    "    ...\n"
)

# Additional interface context for the prompt
STRATEGY_INTERFACE_EXPLANATION = """
Each agent runs TWO functions:

1. evaluate(current_reputation, observation, my_history, round_num) -> float
   - Called when you OBSERVE another agent's action
   - You receive the observed agent's current reputation and the observation
   - You return an UPDATED reputation score
   - The observation tells you: who donated to whom, and what action they took
   - The observation also includes "donor_reputation" and "recipient_reputation"
     — the standing each agent has in YOUR private store (useful for designing
     strategies that condition on the recipient's prior standing, e.g. whether
     defection-against-a-good-recipient should be punished differently from
     defection-against-a-bad-recipient). Both default to a small positive
     warm-start value (~0.01) for agents you have never observed.
   - Use observation["action"] == "cooperate" (NOT observation["picked"]) to check.
     The "action" field is a string with two possible values, "cooperate"
     and "defect"; these names describe the underlying game semantics directly.
     "cooperate" is the cost-imposing positive-sum option (donor pays 1,
     recipient gains 2); "defect" is the no-cost no-benefit option.

2. decide(recipient_reputation, round_num, my_history) -> bool
   - Called when YOU must decide which action to take against a recipient
   - You receive YOUR private reputation assessment of the recipient
   - If you have never observed this recipient, reputation starts near 0 (0.01)
   - Return True to cooperate, False to defect
   - The action labels in the prompt are pre-named with the game's
     underlying semantics ("cooperate" = donate, "defect" = not donate)

Both functions are compiled together as one code block.
"""


# ============================================================================
# Initialization prompt: generate N diverse (evaluate, decide) pairs
# ============================================================================

INIT_PROMPT_TEMPLATE = """You are designing agent strategies for a repeated economic game.

GAME RULES:
- {population_size} agents interact over {num_rounds} rounds.
- Each round, every agent is paired with a random recipient.
- Each round, the agent must choose one of TWO actions: "cooperate" or
  "defect". "cooperate" means the agent pays a cost of 1 and the recipient
  gains 2 (a positive-sum move). "defect" means no cost and no benefit.
  The action labels are pre-named with the game's underlying semantics.
- Agents interact with the same population repeatedly.
- Each agent keeps a private history of interactions it participated in.
- Each agent may also observe some fraction of other agents' interactions.

DIVERSITY REQUIREMENTS:
- Write {num_strategies} DIVERSE strategy pairs using different approaches
- Some pairs might have simple evaluate (always return the same score) and simple decide (always cooperate / always defect)
- Some pairs might have evaluate that treats the two actions differently (e.g. one is rewarded, one is punished)
- Some pairs might have evaluate that weights recent actions more heavily
- Some pairs might have decide that uses different reputation thresholds
- Some pairs might have decide that also considers personal history
- Each pair must define BOTH "evaluate" AND "decide"
- Write ONLY the Python code, nothing else

Separate each pair with "# ---" on its own line.

Pair {start_idx}:
"""


# ============================================================================
# Mutation prompt: vary a successful strategy pair
# ============================================================================

MUTATION_PROMPT_TEMPLATE = """You are improving agent strategies for a repeated economic game.

GAME RULES:
- {population_size} agents interact over many rounds.
- Each round, every agent is paired with a random recipient.
- The agent must choose one of two actions, referred to in the observation
  dict as "cooperate" and "defect". "cooperate" costs the agent 1 and
  gives the recipient 2 (a positive-sum cooperative move). "defect" costs
  nothing and gives the recipient nothing (a non-cooperative move).
  The action labels are pre-named with the game's underlying semantics.

Below is a strategy pair that performed well (score: {fitness:.1f}).
It has TWO functions: evaluate (updates reputation from observations) and decide (chooses which action to take).

ORIGINAL CODE:
```python
{parent_code}
```

Your task: Create a VARIANT of this strategy pair.
The variant MUST contain both "evaluate" and "decide" functions.

MODIFICATION GUIDELINES:
- Change how evaluate updates reputation: modify scoring rules, weight recent vs old actions, add/remove conditions
- Change how decide uses reputation: modify thresholds, add conditions based on personal history
- The variant should be recognizably related to the original but make DIFFERENT choices
- You may add new tracking variables, different counting methods, or alternative approaches
- Keep the EXACT same function signatures (parameter names and order must match)
- Use observation["action"] (a string equal to "cooperate" or "defect") in evaluate
- Return True from decide() to cooperate, False to defect. The action
  labels directly correspond to the game's underlying semantics.

Return ONLY the modified Python code (both functions), nothing else.
"""


# ============================================================================
# Exploration-mode mutation: prompt variant that does NOT name specific algorithms
# but encourages consideration of strategies that aggregate information over
# multiple observations rather than reacting to single events. Used by the
# algorithmic-complexity ceiling probes (v16+).
# ============================================================================

EXPLORATION_MUTATION_PROMPT_TEMPLATE = """You are improving agent strategies for a repeated economic game.

GAME RULES:
- {population_size} agents interact over many rounds.
- Each round, every agent is paired with a random recipient.
- The agent must choose one of two actions, referred to in the observation
  dict as "cooperate" and "defect". "cooperate" costs the agent 1 and
  gives the recipient 2 (a positive-sum cooperative move). "defect" costs
  nothing and gives the recipient nothing (a non-cooperative move).
  The action labels are pre-named with the game's underlying semantics.

Below is a strategy pair that performed well (score: {fitness:.1f}).
It has TWO functions: evaluate (updates reputation from observations) and decide (chooses which action to take).

ORIGINAL CODE:
```python
{parent_code}
```

Your task: Create a VARIANT of this strategy pair.
The variant MUST contain both "evaluate" and "decide" functions.

DIVERSITY GUIDELINES (this is an exploration round — be creative):
- Consider strategies that score an agent based on its PATTERN of actions
  over multiple observations, not just the most recent single event.
  For example: a sliding window of the last several observations; counting
  cooperations vs defections over time; weighting recent observations
  differently from older ones; tracking how OFTEN an agent cooperated
  relative to the population average.
- Consider decide() rules that depend on multiple inputs at once
  (recipient reputation AND own past behavior AND round number),
  not only single thresholds.
- The variant should be recognizably related to the original but should
  make choices in situations where the original strategy is INDECISIVE
  (e.g. when recipient_reputation is near 0, or when round_num is mid-game).
- You may add new tracking variables, different counting methods, or
  alternative approaches.
- Keep the EXACT same function signatures (parameter names and order must match).
- Use observation["action"] (a string equal to "cooperate" or "defect") in evaluate
- Return True from decide() to cooperate, False to defect. The action
  labels directly correspond to the game's underlying semantics.

Return ONLY the modified Python code (both functions), nothing else.
"""


# ============================================================================
# Diversity check: ensure strategies are actually different
# ============================================================================

DIVERSITY_PROMPT_TEMPLATE = """Here are two strategy pairs for a donation game.
Each has an evaluate function and a decide function.

Strategy A:
```python
{code_a}
```

Strategy B:
```python
{code_b}
```

Are these strategy pairs meaningfully different in their logic?
Answer YES or NO, then briefly explain.
"""


# ============================================================================
# Strategy analysis: post-hoc qualitative analysis of evolved strategies
# ============================================================================

ANALYSIS_PROMPT_TEMPLATE = """Analyze this strategy pair from a repeated cooperation game.

GAME RULES:
- Each round, an agent picks one of two actions: "cooperate" (pays 1, gives
  recipient 2) or "defect" (pays nothing, gives recipient nothing).
  The action labels describe the game's underlying semantics directly.
- Agents run evaluate() to update private reputation scores from observations.
- Agents run decide() to choose which action to take based on reputation.

STRATEGY CODE:
```python
{code}
```

FITNESS: {fitness:.1f}

Answer these questions:
1. How does evaluate() update reputation? What information does it use? (1-2 sentences)
2. How does decide() use the reputation score? What threshold or rule? (1-2 sentences)
3. What behavioral archetype does this resemble? (e.g., image-scoring, standing, always-cooperate, always-defect, tit-for-tat, discriminator, novel)
4. Does this strategy implement direct reciprocity (personal history), indirect reciprocity (observation-based reputation), or both?
5. What would you name this strategy?
"""


# ============================================================================
# Helper functions
# ============================================================================

def build_init_prompt(
    num_strategies: int = 20,
    population_size: int = 20,
    num_rounds: int = 30,
    start_idx: int = 0,
    compact: bool = False,
    recent_window: int = 0,
) -> str:
    """Build the initialization prompt to generate diverse strategy pairs."""
    prompt = INIT_PROMPT_TEMPLATE
    prompt = prompt.replace("{num_strategies}", str(num_strategies))
    prompt = prompt.replace("{population_size}", str(population_size))
    prompt = prompt.replace("{num_rounds}", str(num_rounds))
    prompt = prompt.replace("{interface_explanation}", STRATEGY_INTERFACE_EXPLANATION)
    if recent_window > 0:
        # Inject the recent_window field description into the interface
        # (only when this feature is enabled — keeps the default prompt
        # identical to v15 for backward compatibility)
        from_str = '"action": "cooperate" | "defect"\n        (the action label is pre-named: "cooperate" is the option that costs the\n         donor 1 and gives the recipient 2; "defect" is the option that costs 0\n         and gives the recipient 0. These names describe the game\'s underlying\n         semantics, not the LLM-facing vocabulary.)'
        to_str = from_str + '\n\n        "recent_window": list[dict] (optional; only present if the game was\n            configured with a recent-actions window. Each entry: {"donor": int,\n            "action": "cooperate"|"defect", "round": int} — the most recent\n            observed interactions involving OTHER agents, in reverse\n            chronological order. Empty list if no observations yet. You may\n            ignore this key entirely if you don\'t need it; it is only present\n            to enable strategies that aggregate over multiple observations.)'
        interface = STRATEGY_INTERFACE.replace(from_str, to_str)
        interface_compact = STRATEGY_INTERFACE_COMPACT.replace(
            "'round' (int), 'recent_window' (optional list of recent observed actions)",
            "'round' (int), 'recent_window' (optional list of recent observed actions)",
        )
        prompt = prompt.replace("{strategy_interface}",
                                interface_compact if compact else interface)
    else:
        prompt = prompt.replace("{strategy_interface}",
                                STRATEGY_INTERFACE_COMPACT if compact else STRATEGY_INTERFACE)
    prompt = prompt.replace("{start_idx}", str(start_idx))
    return prompt


def build_mutation_prompt(
    parent_code: str,
    fitness: float,
    population_size: int = 20,
    recent_window: int = 0,
) -> str:
    """Build the mutation prompt to vary a successful strategy pair."""
    prompt = MUTATION_PROMPT_TEMPLATE
    prompt = prompt.replace("{parent_code}", parent_code)
    prompt = prompt.replace("{fitness}", f"{fitness:.1f}")
    prompt = prompt.replace("{population_size}", str(population_size))
    if recent_window > 0:
        # Append a note about the recent_window observation field
        prompt += (
            "\nNOTE: observation dict now contains a 'recent_window' field — a list "
            "of recent (donor, action, round) dicts for OTHER agents. You may use it "
            "to aggregate over multiple observations (e.g. count cooperations, "
            "compute a sliding-window majority) if you wish; or you may ignore it."
        )
    return prompt


def build_exploration_mutation_prompt(
    parent_code: str,
    fitness: float,
    population_size: int = 20,
    recent_window: int = 0,
) -> str:
    """Build the exploration-mode mutation prompt. Does NOT name specific
    algorithms (leading-eight, RL, MCTS, etc.) but encourages the LLM to
    consider strategies that aggregate information over multiple observations
    rather than reacting to single events. Used by --exploration-mutation flag."""
    prompt = EXPLORATION_MUTATION_PROMPT_TEMPLATE
    prompt = prompt.replace("{parent_code}", parent_code)
    prompt = prompt.replace("{fitness}", f"{fitness:.1f}")
    prompt = prompt.replace("{population_size}", str(population_size))
    if recent_window > 0:
        prompt += (
            "\nNOTE: observation dict now contains a 'recent_window' field — a list "
            "of recent (donor, action, round) dicts for OTHER agents. You may use it "
            "to aggregate over multiple observations (e.g. count cooperations, "
            "compute a sliding-window majority) if you wish; or you may ignore it."
        )
    return prompt


def build_analysis_prompt(code: str, fitness: float) -> str:
    """Build the analysis prompt for post-hoc strategy classification."""
    return ANALYSIS_PROMPT_TEMPLATE.format(code=code, fitness=fitness)


def build_diversity_prompt(code_a: str, code_b: str) -> str:
    """Build the diversity check prompt."""
    return DIVERSITY_PROMPT_TEMPLATE.format(code_a=code_a, code_b=code_b)
