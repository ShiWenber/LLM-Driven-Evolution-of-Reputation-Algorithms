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
        "action": "donate" | "not_donate"

    Note: evaluating a defection AGAINST a good-reputation recipient
    may be different from defection against a bad-reputation recipient.
    Similarly, donating TO a bad-reputation recipient may be interpreted
    differently. Use recipient identity to inform your evaluation.

    CRITICAL: "action" is a string, NOT "donated".
    Example: if obs["action"] == "donate": ...

    my_history: interactions you personally participated in.
      Each entry: {"round": int, "role": "donor"|"recipient",
                   "partner": int, "action": "donate"|"not_donate",
                   "partner_action": "donate"|"not_donate"|None}

    Returns: updated reputation score (float between -1.0 and 1.0).
    """
    pass


def decide(
    recipient_reputation: float,
    round_num: int,
    my_history: list[dict]
) -> bool:
    """
    Decide whether to DONATE to the recipient.

    DONATE: you pay 1, recipient gains 2.
    NOT DONATE: you pay 0, recipient gains 0.

    recipient_reputation: YOUR private assessment of the recipient.
      This is the score you have built up by observing their behavior.
      If you have never observed this recipient, this value is near 0.0
      (specifically 0.01 — a small positive value to enable cold start).

    my_history: interactions you personally participated in.
      Same format as in evaluate().

    Returns True to DONATE, False to NOT DONATE.
    """
    pass'''

# Compact version without docstrings for batch generation
STRATEGY_INTERFACE_COMPACT = (
    "def evaluate(current_reputation: float, observation: dict, "
    "my_history: list[dict], round_num: int) -> float:\n"
    "    # Update private reputation based on observed action\n"
    "    ...\n\n"
    "def decide(recipient_reputation: float, round_num: int, "
    "my_history: list[dict]) -> bool:\n"
    "    # Decide whether to donate based on recipient's reputation\n"
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
   - Use observation["action"] == "donate" (NOT observation["donated"]) to check

2. decide(recipient_reputation, round_num, my_history) -> bool
   - Called when YOU must decide whether to donate to a recipient
   - You receive YOUR private reputation assessment of the recipient
   - If you have never observed this recipient, reputation starts near 0 (0.01)
   - Return True to DONATE, False to NOT DONATE

Both functions are compiled together as one code block.
"""


# ============================================================================
# Initialization prompt: generate N diverse (evaluate, decide) pairs
# ============================================================================

INIT_PROMPT_TEMPLATE = """You are designing agent strategies for a repeated economic game.

GAME RULES:
- {population_size} agents interact over {num_rounds} rounds.
- Each round, every agent is paired with a random recipient.
- The agent can DONATE (pay 1, recipient gains 2) or NOT DONATE (pay 0, recipient gains 0).
- Agents interact with the same population repeatedly.
- Each agent keeps a private history of interactions it participated in.
- Each agent may also observe some fraction of other agents' interactions.

Each agent runs TWO functions:

{interface_explanation}

Your task: Write {num_strategies} DIFFERENT strategy pairs.
Each pair MUST contain exactly TWO functions named "evaluate" and "decide".

Here is the exact interface you must follow for BOTH functions:

```python
{strategy_interface}
```

DIVERSITY REQUIREMENTS:
- Write {num_strategies} DIVERSE strategy pairs using different approaches
- Some pairs might have simple evaluate (always return same score) and simple decide (always/never donate)
- Some pairs might have evaluate that counts good/bad actions differently
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
- The agent can DONATE (pay 1, recipient gains 2) or NOT DONATE (pay 0, recipient gains 0).

Below is a strategy pair that performed well (score: {fitness:.1f}).
It has TWO functions: evaluate (updates reputation from observations) and decide (makes donation decisions).

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
- Use observation["action"] == "donate" (NOT observation["donated"]) in evaluate

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

ANALYSIS_PROMPT_TEMPLATE = """Analyze this strategy pair from a repeated donation game.

GAME RULES:
- Agents can DONATE (pay 1, recipient gains 2) or NOT DONATE (pay 0, recipient gains 0).
- Agents run evaluate() to update private reputation scores from observations.
- Agents run decide() to make donation decisions based on reputation.

STRATEGY CODE:
```python
{code}
```

FITNESS: {fitness:.1f}

Answer these questions:
1. How does evaluate() update reputation? What information does it use? (1-2 sentences)
2. How does decide() use the reputation score? What threshold or rule? (1-2 sentences)
3. What behavioral archetype does this resemble? (e.g., image-scoring, standing, always-cooperate, tit-for-tat, discriminator, defector, novel)
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
    compact: bool = False
) -> str:
    """Build the initialization prompt to generate diverse strategy pairs."""
    prompt = INIT_PROMPT_TEMPLATE
    prompt = prompt.replace("{num_strategies}", str(num_strategies))
    prompt = prompt.replace("{population_size}", str(population_size))
    prompt = prompt.replace("{num_rounds}", str(num_rounds))
    prompt = prompt.replace("{interface_explanation}", STRATEGY_INTERFACE_EXPLANATION)
    prompt = prompt.replace("{strategy_interface}",
                            STRATEGY_INTERFACE_COMPACT if compact else STRATEGY_INTERFACE)
    prompt = prompt.replace("{start_idx}", str(start_idx))
    return prompt


def build_mutation_prompt(
    parent_code: str,
    fitness: float,
    population_size: int = 20,
) -> str:
    """Build the mutation prompt to vary a successful strategy pair."""
    prompt = MUTATION_PROMPT_TEMPLATE
    prompt = prompt.replace("{parent_code}", parent_code)
    prompt = prompt.replace("{fitness}", f"{fitness:.1f}")
    prompt = prompt.replace("{population_size}", str(population_size))
    return prompt


def build_analysis_prompt(code: str, fitness: float) -> str:
    """Build the analysis prompt for post-hoc strategy classification."""
    return ANALYSIS_PROMPT_TEMPLATE.format(code=code, fitness=fitness)


def build_diversity_prompt(code_a: str, code_b: str) -> str:
    """Build the diversity check prompt."""
    return DIVERSITY_PROMPT_TEMPLATE.format(code_a=code_a, code_b=code_b)
