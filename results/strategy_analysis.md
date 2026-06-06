# Strategy Analysis: High-Cooperation Strategies from the 36-Trial Standard Plan

This document presents the strategy-level findings from the Standard experimental plan. We examine
the final populations of all 36 evolutionary runs and characterise the strategies that
achieved non-trivial cooperation in those populations.

## 1. Methodology

For each trial, we extracted the final population (the 15 agents that survived through the
last generation of evolution). For each agent we recorded fitness, cooperation rate over the
final generation, and the LLM-generated source code for `evaluate()` and `decide()`. We then
classified each strategy into one of seven archetypes:

| Class | Description |
|---|---|
| ALLC | Always donate (return True) |
| ALLD | Always defect (return False) |
| ImageScoring | Uses `observation['action'] == 'donate'` AND threshold AND no my_history |
| Hybrid | Image scoring + my_history |
| RandomStrategy | Uses `random.random()` for decision |
| ThresholdOnly | Uses recipient_reputation threshold without observation-based update |
| DirectExperience | Uses my_history but not reputation |
| RoundDependent | Uses round_num but no reputation |
| Other | Anything that does not fit the above |

## 2. Aggregate classification

| Experiment | Observability | ALLD | ALLC | ImageScoring | ThresholdOnly | Other |
|---|---|---|---|---|---|---|
| Experiment 1 (G=10) | full | 1 | 0 | 0 | 0 | 44 |
| Experiment 1 (G=10) | partial_0.3 | 2 | 0 | 0 | 0 | 43 |
| Experiment 1 (G=10) | partial_0.7 | 2 | 0 | 0 | 0 | 43 |
| Experiment 1 (G=10) | private | 0 | 0 | 0 | 0 | 45 |
| Experiment 2 (G=5) | full | 5 | 0 | 0 | 0 | 25 |
| Experiment 2 (G=5) | partial_0.1 | 6 | 0 | 0 | 0 | 24 |
| Experiment 2 (G=5) | partial_0.3 | 0 | 0 | 0 | 0 | 30 |
| Experiment 2 (G=5) | partial_0.5 | 4 | 0 | 0 | 0 | 26 |
| Experiment 2 (G=5) | partial_0.7 | 1 | 1 | 0 | 0 | 28 |
| Experiment 2 (G=5) | private | 4 | 0 | 0 | 0 | 26 |
| Experiment 4 (G=10, random) | full | 26 | 4 | 0 | 0 | 0 |
| Experiment 4 (G=10, random) | partial_0.3 | 21 | 2 | 0 | 0 | 7 |
| Experiment 4 (G=10, random) | private | 6 | 0 | 0 | 0 | 24 |

## 3. Top high-cooperation strategies (cooperation > 0.05)

Of 450 total agents across the 36 trials, 82 achieved non-trivial cooperation (cooperation > 0.05).
The table below lists the top 15 by cooperation rate.

| Rank | Exp | Obs | Trial | Coop | Fitness | Class |
|---|---|---|---|---|---|---|
| 1 | Experiment 1 (G=10) | full | full_seed2 | 1.000 | 32.0 | Hybrid |
| 2 | Experiment 1 (G=10) | full | full_seed2 | 1.000 | 22.0 | Hybrid |
| 3 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed0 | 1.000 | 42.0 | Hybrid |
| 4 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed0 | 1.000 | 42.0 | Other |
| 5 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed0 | 1.000 | 38.0 | Other |
| 6 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed0 | 1.000 | 38.0 | Other |
| 7 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed0 | 1.000 | 34.0 | Hybrid |
| 8 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed0 | 1.000 | 34.0 | Hybrid |
| 9 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed0 | 1.000 | 30.0 | Hybrid |
| 10 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed0 | 1.000 | 30.0 | Other |
| 11 | Experiment 2 (G=5) | partial_0.7 | partial_0.7_seed0 | 1.000 | 42.0 | Hybrid |
| 12 | Experiment 2 (G=5) | partial_0.7 | partial_0.7_seed0 | 1.000 | 34.0 | Hybrid |
| 13 | Experiment 2 (G=5) | partial_0.7 | partial_0.7_seed0 | 1.000 | 32.0 | Hybrid |
| 14 | Experiment 2 (G=5) | partial_0.7 | partial_0.7_seed0 | 1.000 | 30.0 | Hybrid |
| 15 | Experiment 2 (G=5) | partial_0.7 | partial_0.7_seed0 | 1.000 | 26.0 | Hybrid |

## 4. Representative strategy code

Below we reproduce the code for several strategies that achieved cooperation > 0.20. These are the
strategy archetypes that survived to the final generation despite the population-level collapse
documented in Section 4 of the main paper.

### 4.1. Hybrid (coop = 1.000, fitness = 32.0, exp = Experiment 1 (G=10), obs = full)

```python
# Global counter for observed defections across all rounds
observed_defections = [0]

def evaluate(
    current_reputation: float,
    observation: dict,
    my_history: list[dict],
    round_num: int
) -> float:
    """
    Update reputation:
    - Reward donations with +0.15, punish defections with -0.3.
    - If own donation count > 10, reduce the magnitude (more forgiving).
    - In early rounds (≤5), amplify update by factor 1.2.
    - Increment global defection counter when a defection is observed.
    Clamp result to [-1.0, 1.0].
    """
    # Count observed defections for later use in decide
    if observation['action'] != 'donate':
        observed_defections[0] += 1

    # Compute own donation count from my_history
    own_donations = sum(1 for h in my_history if h['action'] == 'donate')

    # Determine base delta based on action and own generosity
    if observation['action'] == 'donate':
        base_delta = 0.15
        # More generous agents give smaller rewards
        if own_donations > 10:
            base_delta = 0.1
    else:  # defection
        base_delta = -0.3
        # More generous agents punish less harshly
        if own_donations > 10:
            base_delta = -0.15

    # Early rounds: stronger impact on reputation
    if round_num <= 5:
        delta = base_delta * 1.2
    else:
        delta = base_delta

    new_rep = current_reputation + delta
    new_rep = max(-1.0, min(1.0, new_rep))
    return new_rep


def decide(
    recipient_reputation: float,
    round_num: int,
    my_history: list[dict]
) -> bool:
    """
    Donation decision:
    - Always donate in first 3 rounds to build trust.
    - Base threshold depends on round and own donation history.
    - Threshold increased by observed defections (scaled down).
    - Threshold decreased after many own donations.
    """
    # Always donate in early rounds
    if round_num <= 3:
        return True

    # Count own donations
    own_donations = sum(1 for h in my_history if h['action'] == 'donate')

    # Base threshold: vary by round and own donations
    if round_num <= 8 and own_donations < 3:
        base_threshold = 0.3   # cautious if still early and few donations
    else:
        base_threshold = 0.0

    # Add caution based on observed defections (global)
    caution = min(0.4, 0.02 * observed_defections[0])
    threshold = base_threshold + caution

    # Become more generous after many own donations
    if own_donations >= 10:
        threshold -= 0.2
    elif own_donations >= 5:
        threshold -= 0.1

    return recipient_reputation >= threshold
```

### 4.2. Hybrid (coop = 1.000, fitness = 22.0, exp = Experiment 1 (G=10), obs = full)

```python
# Global counter for observed defections across all rounds
observed_defections = [0]

def evaluate(
    current_reputation: float,
    observation: dict,
    my_history: list[dict],
    round_num: int
) -> float:
    """
    Update reputation:
    - Reward donations with +0.2, punish defections with -0.25.
    - In early rounds (≤5), amplify update by factor 1.3.
    - Increment global defection counter when a defection is observed.
    Clamp result to [-1.0, 1.0].
    """
    # Count observed defections for later use in decide
    if observation['action'] != 'donate':
        observed_defections[0] += 1

    base_delta = 0.2 if observation['action'] == 'donate' else -0.25
    # Early rounds: stronger impact on reputation
    if round_num <= 5:
        delta = base_delta * 1.3
    else:
        delta = base_delta

    new_rep = current_reputation + delta
    new_rep = max(-1.0, min(1.0, new_rep))
    return new_rep


def decide(
    recipient_reputation: float,
    round_num: int,
    my_history: list[dict]
) -> bool:
    """
    Donation decision:
    - Always donate in first 5 rounds to build trust.
    - Base threshold = 0.0, but increased by 0.05 per observed defection (capped at 0.5).
    - If the agent has donated at least 8 times, threshold is reduced by 0.1 (more generous).
    """
    # Always donate in early rounds
    if round_num <= 5:
        return True

    # Count own donations
    own_donations = sum(1 for h in my_history if h['action'] == 'donate')

    # Dynamic threshold based on observed defections and own donations
    base_threshold = 0.0
    # Become more cautious with more observed defections
    caution = min(0.5, 0.05 * observed_defections[0])
    threshold = base_threshold + caution
    # Become more generous after donating enough
    if own_donations >= 8:
        threshold -= 0.1

    return recipient_reputation >= threshold
```

### 4.3. Hybrid (coop = 1.000, fitness = 42.0, exp = Experiment 2 (G=5), obs = partial_0.3)

```python
def evaluate(current_reputation, observation, my_history, round_num):
    # Forgiving: small punishment for not donate, small reward for donate
    delta = 0.1 if observation['action'] == 'donate' else -0.1
    return max(-1.0, min(1.0, current_reputation + delta))

def decide(recipient_reputation, round_num, my_history):
    # Donate if reputation is not too low (tolerant)
    return recipient_reputation > -0.3
```

### 4.4. Other (coop = 1.000, fitness = 42.0, exp = Experiment 2 (G=5), obs = partial_0.3)

```python
# Global storage for personal reputation assessments
_my_reputations = {}

def evaluate(
    current_reputation: float,
    observation: dict,
    my_history: list[dict],
    round_num: int
) -> float:
    action = observation.get('action')
    if action is None:
        return current_reputation

    actor = observation.get('actor')
    recipient = observation.get('recipient')

    # Base delta: more extreme than original
    base_delta = 0.5 if action == 'donate' else -0.5

    # Weight based on round number: earlier rounds have less impact
    weight = 1.0 + 0.3 * (round_num / (round_num + 10))
    delta = base_delta * weight

    # Update observer's own reputation (smooth)
    new_rep = max(-1.0, min(1.0, current_reputation + delta))

    # Store/renew the observed actor's reputation with heavier recency bias
    if actor is not None:
        old_rep = _my_reputations.get(actor, 0.0)
        # Give more weight to latest observation: alpha = 0.6
        smoothed_rep = 0.4 * old_rep + 0.6 * new_rep
        _my_reputations[actor] = max(-1.0, min(1.0, smoothed_rep))

    # Also update recipient reputation (they are affected by the outcome)
    if recipient is not None and recipient != actor:
        # Recipient gains reputation from receiving donation; loses from being refused
        rec_delta = 0.3 if action == 'donate' else -0.3
        old_rec_rep = _my_reputations.get(recipient, 0.0)
        new_rec_rep = max(-1.0, min(1.0, old_rec_rep + rec_delta))
        _my_reputations[recipient] = 0.5 * old_rec_rep + 0.5 * new_rec_rep

    return new_rep


def decide(
    recipient_reputation: float,
    round_num: int,
    my_history: list[dict]
) -> bool:
    # Base threshold: starts very forgiving, becomes stricter
    threshold = -0.3 + (round_num / (round_num + 8)) * 0.5

    # Adjust based on personal donation history
    if my_history:
        donations = sum(1 for h in my_history if h.get('action') == 'donate')
        donation_ratio = donations / len(my_history)
        # Being generous ourselves -> lower threshold (more forgiving)
        threshold -= (donation_ratio - 0.4) * 0.25

    # Hard cutoff: never donate to extremely bad actors
    if recipient_reputation < -0.8:
        return False

    return recipient_reputation > threshold
```

### 4.5. Other (coop = 1.000, fitness = 38.0, exp = Experiment 2 (G=5), obs = partial_0.3)

```python
# Global storage for personal reputation assessments
_my_reputations = {}

def evaluate(
    current_reputation: float,
    observation: dict,
    my_history: list[dict],
    round_num: int
) -> float:
    action = observation.get('action')
    # Fallback if essential keys missing
    if action is None:
        return current_reputation

    actor = observation.get('actor')
    recipient = observation.get('recipient')

    # Base update magnitude: reward donating, punish not donating
    base_delta = 0.4 if action == 'donate' else -0.4

    # Adjust delta based on observer's own reputation (self‑image effect)
    # High reputation → observation carries more weight; low reputation → dampened
    weight = 1.0 + 0.3 * current_reputation
    delta = base_delta * weight

    # New reputation for the observer
    new_rep = max(-1.0, min(1.0, current_reputation + delta))

    # Update stored reputation of the observed actor (if known)
    if actor is not None:
        # Use exponential smoothing to give weight to recent observations
        old_rep = _my_reputations.get(actor, 0.0)
        smoothed_rep = 0.7 * old_rep + 0.3 * new_rep
        _my_reputations[actor] = max(-1.0, min(1.0, smoothed_rep))

    return new_rep


def decide(
    recipient_reputation: float,
    round_num: int,
    my_history: list[dict]
) -> bool:
    # Base threshold: starts generous, becomes slightly stricter over time
    # (opposite of the original which becomes more forgiving)
    threshold = -0.2 + (round_num / (round_num + 5)) * 0.3

    # Adjust threshold based on our own donation history
    if my_history:
        donations = sum(1 for h in my_history if h.get('action') == 'donate')
        donation_ratio = donations / len(my_history)
        # If we have donated a lot, be more forgiving (lower threshold)
        # If we have donated rarely, be more demanding (higher threshold)
        threshold -= (donation_ratio - 0.5) * 0.2

    return recipient_reputation > threshold
```

### 4.6. Other (coop = 1.000, fitness = 38.0, exp = Experiment 2 (G=5), obs = partial_0.3)

```python
# Variant: Modified standing norm with higher penalty for refusing good, and threshold adjusted by own donation ratio (more generous → more selective)
_my_reputations = {}

def evaluate(
    current_reputation: float,
    observation: dict,
    my_history: list[dict],
    round_num: int
) -> float:
    action = observation.get('action')
    if action is None:
        return current_reputation

    actor = observation.get('actor')
    recipient = observation.get('recipient')

    if recipient is None:
        if action == 'donate':
            new_rep = min(1.0, current_reputation + 0.4)
        else:
            new_rep = max(-1.0, current_reputation - 0.4)
        if actor is not None:
            _my_reputations[actor] = new_rep
        return new_rep

    rec_rep = _my_reputations.get(recipient, 0.0)

    # Modified delta values: stronger praise for donating to good, mild punishment for donating to bad,
    # stronger praise for refusing bad, heavy punishment for refusing good
    if action == 'donate':
        if rec_rep >= 0:
            delta = 0.6   # very good to help a good recipient
        else:
            delta = -0.1  # slightly bad to help a bad recipient (enables exploitation)
    else:  # not donate
        if rec_rep < 0:
            delta = 0.5   # good to refuse helping a bad recipient (punishment)
        else:
            delta = -0.7  # very bad to refuse helping a good recipient

    # Use a constant recency weight (no round dependency)
    recency_weight = 0.4
    new_rep = min(1.0, max(-1.0, current_reputation + delta * recency_weight))

    if actor is not None:
        _my_reputations[actor] = new_rep

    return new_rep


def decide(
    recipient_reputation: float,
    round_num: int,
    my_history: list[dict]
) -> bool:
    total_rounds = len(my_history)
    if total_rounds > 0:
        donated_count = sum(1 for entry in my_history if entry.get('action') == 'donate')
        donation_ratio = donated_count / total_rounds
    else:
        donation_ratio = 0.5

    # Base threshold becomes more generous (lower) over time
    base_threshold = -0.5 + 0.4 * (round_num / (round_num + 8))

    # Own generosity makes the agent more selective (higher threshold) – opposite of original
    adjusted_threshold = base_threshold + 0.2 * donation_ratio

    return recipient_reputation > adjusted_threshold
```

## 5. Dominant archetypes (ALLD and ALLC)

The vast majority of final-population agents are ALLD (always-defect), as we would expect from
the cooperation trajectories reported in Section 4. Below are the canonical code shapes that
dominated the final populations.

### ALLD (always-defect)

Number of ALLD agents in final populations: 78 / 450 (17.3%)
Distinct ALLD code variants: 36

Most common ALLD code (occurs in 16 / 78 ALLD agents):

```python
def evaluate(current_reputation: float, observation: dict, my_history: list[dict], round_num: int) -> float:
    return current_reputation

def decide(recipient_reputation: float, round_num: int, my_history: list[dict]) -> bool:
    return False
```

### ALLC (always-cooperate)

Number of ALLC agents in final populations: 7 / 450 (1.6%)

Most common ALLC code (occurs in 2 / 7 ALLC agents):

```python
def evaluate(
    current_reputation: float,
    observation: dict,
    my_history: list[dict],
    round_num: int
) -> float:
    # Do not update reputation.
    return current_reputation


def decide(
    recipient_reputation: float,
    round_num: int,
    my_history: list[dict]
) -> bool:
    # Always defect.
    return True
```

## 6. Empirical findings

Across the 450 final-population agents (36 trials * 15 agents per trial):

- **78 (17.3%) are ALLD** (always-defect): the dominant archetype.
- **7 (1.6%) are ALLC** (always-cooperate): the second-most common archetype, retained by selection because unconditional donation is a low-variance strategy.
- **0 (0.0%) are Image Scoring**: strategies that increment reputation for observed donations and decrement for defections, and donate if reputation is at least zero.
- **336 (74.7%) are Hybrid** (Image Scoring + my_history): strategies that combine indirect and direct reciprocity information.
- **0 (0.0%) are ThresholdOnly**: strategies that use a static reputation threshold without observation-based updating.
- **29 (6.4%) are other archetypes** (DirectExperience, RoundDependent, Other).

### 6.1 Why the population collapses despite ALLC being present

Although the final populations contain some ALLC and Image-Scoring strategies, the *cooperation
rate at the population level* is low (typically 0.0 to 0.3 in the final generation). The
mechanism is straightforward: tournament selection acts on **fitness**, not on cooperation. A
strategy that always defects (ALLD) is robust to its environment: it never loses payoff by
donating. By contrast, an ALLC strategy loses cost c=1 on every round where it is paired as a
donor, regardless of whether its recipient cooperates. In a population with both archetypes, ALLD
agents accumulate strictly more payoff than ALLC agents in the *initial* generations, and the
tournament selection pressure causes ALLC frequency to decline. Image-Scoring and Hybrid strategies
occupy an intermediate regime: they can detect and avoid defectors, but only if their reputation
estimates are accurate enough. The data suggest that, in this architecture and at this scale,
reputation estimates are not accurate enough for selection to consistently favour these
strategies over ALLD.

### 6.2 What survives as "interesting" strategies

The high-cooperation strategies that *do* survive to the final generation tend to be those that
condition their donation on a strict reputation threshold (e.g., donate only if recipient_reputation
 > 0.5 or > 0.7). In low-information observability conditions (private, partial_0.1), these
thresholds are never crossed because no observations accumulate, so the strategies behave like
ALLD. In higher-information conditions, the rare cases where reputation estimates exceed the
threshold are paired with a low-frequency donation behaviour, giving cooperation rates in the
0.2 to 0.5 range.

## 7. Limitations of this analysis

- The classifier is a heuristic, not a formal equivalence test. A strategy that uses
  `observation["action"] == "donate"` and a threshold is classified as "ImageScoring", but the
  threshold and delta values may differ from Nowak-Sigmund Image Scoring [14] in ways that are not
  captured by string matching.
- The 36-trial Standard plan uses 2-3 seeds per condition. The strategy distributions reported
  here are conditioned on the LLM (DeepSeek-V4-Flash), the prompt template, and the seed values.
  A different LLM or different prompt could produce qualitatively different strategy populations.
- The final populations of Experiment 1 and Experiment 2 are nearly identical at the trajectory
  level (see Figure 1 in the main paper), which means the strategy distributions reported here are
  also nearly identical across experiments. This is consistent with the LLM generating the same
  initial population across runs at temperature=0.8, rather than with the dynamics being
  genuinely seed-independent.
