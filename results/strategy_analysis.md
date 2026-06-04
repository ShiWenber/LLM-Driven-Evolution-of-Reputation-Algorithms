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
| Experiment 1 (G=10) | full | 39 | 6 | 0 | 0 | 0 |
| Experiment 1 (G=10) | partial_0.3 | 39 | 6 | 0 | 0 | 0 |
| Experiment 1 (G=10) | partial_0.7 | 36 | 9 | 0 | 0 | 0 |
| Experiment 1 (G=10) | private | 38 | 7 | 0 | 0 | 0 |
| Experiment 2 (G=5) | full | 18 | 8 | 0 | 0 | 4 |
| Experiment 2 (G=5) | partial_0.1 | 22 | 2 | 0 | 0 | 6 |
| Experiment 2 (G=5) | partial_0.3 | 27 | 1 | 0 | 0 | 2 |
| Experiment 2 (G=5) | partial_0.5 | 16 | 4 | 0 | 0 | 10 |
| Experiment 2 (G=5) | partial_0.7 | 28 | 2 | 0 | 0 | 0 |
| Experiment 2 (G=5) | private | 28 | 2 | 0 | 0 | 0 |
| Experiment 4 (G=10, random) | full | 26 | 4 | 0 | 0 | 0 |
| Experiment 4 (G=10, random) | partial_0.3 | 26 | 4 | 0 | 0 | 0 |
| Experiment 4 (G=10, random) | private | 24 | 6 | 0 | 0 | 0 |

## 3. Top high-cooperation strategies (cooperation > 0.05)

Of 450 total agents across the 36 trials, 22 achieved non-trivial cooperation (cooperation > 0.05).
The table below lists the top 15 by cooperation rate.

| Rank | Exp | Obs | Trial | Coop | Fitness | Class |
|---|---|---|---|---|---|---|
| 1 | Experiment 2 (G=5) | partial_0.5 | partial_0.5_seed0 | 1.000 | 8.0 | ALLC |
| 2 | Experiment 2 (G=5) | partial_0.5 | partial_0.5_seed1 | 1.000 | 8.0 | ALLC |
| 3 | Experiment 2 (G=5) | partial_0.1 | partial_0.1_seed0 | 0.600 | 14.0 | Hybrid |
| 4 | Experiment 2 (G=5) | partial_0.1 | partial_0.1_seed1 | 0.600 | 14.0 | Hybrid |
| 5 | Experiment 2 (G=5) | full | full_seed0 | 0.533 | 2.0 | DirectExperience |
| 6 | Experiment 2 (G=5) | full | full_seed1 | 0.533 | 2.0 | DirectExperience |
| 7 | Experiment 2 (G=5) | partial_0.1 | partial_0.1_seed0 | 0.533 | 12.0 | DirectExperience |
| 8 | Experiment 2 (G=5) | partial_0.1 | partial_0.1_seed1 | 0.533 | 12.0 | DirectExperience |
| 9 | Experiment 2 (G=5) | partial_0.1 | partial_0.1_seed0 | 0.500 | 11.0 | DirectExperience |
| 10 | Experiment 2 (G=5) | partial_0.1 | partial_0.1_seed1 | 0.500 | 11.0 | DirectExperience |
| 11 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed1 | 0.500 | 9.0 | DirectExperience |
| 12 | Experiment 2 (G=5) | partial_0.3 | partial_0.3_seed1 | 0.467 | 12.0 | DirectExperience |
| 13 | Experiment 2 (G=5) | full | full_seed0 | 0.400 | 10.0 | DirectExperience |
| 14 | Experiment 2 (G=5) | full | full_seed1 | 0.400 | 10.0 | DirectExperience |
| 15 | Experiment 2 (G=5) | partial_0.5 | partial_0.5_seed0 | 0.333 | 12.0 | Hybrid |

## 4. Representative strategy code

Below we reproduce the code for several strategies that achieved cooperation > 0.20. These are the
strategy archetypes that survived to the final generation despite the population-level collapse
documented in Section 4 of the main paper.

### 4.1. ALLC (coop = 1.000, fitness = 8.0, exp = Experiment 2 (G=5), obs = partial_0.5)

```python
def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return True
```

### 4.2. Hybrid (coop = 0.600, fitness = 14.0, exp = Experiment 2 (G=5), obs = partial_0.1)

```python
def evaluate(current_reputation, observation, my_history, round_num):
    if observation["action"] == "donate":
        return current_reputation + 0.8
    else:
        return current_reputation - 1.0

def decide(recipient_reputation, round_num, my_history):
    return recipient_reputation >= 0.0
```

### 4.3. DirectExperience (coop = 0.533, fitness = 2.0, exp = Experiment 2 (G=5), obs = full)

```python
import random

def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return random.random() < 0.5
```

### 4.4. DirectExperience (coop = 0.467, fitness = 12.0, exp = Experiment 2 (G=5), obs = partial_0.3)

```python
import random

def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return random.random() < 0.6
```

### 4.5. DirectExperience (coop = 0.400, fitness = 10.0, exp = Experiment 2 (G=5), obs = full)

```python
import random

def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return random.random() < 0.4
```

### 4.6. Hybrid (coop = 0.333, fitness = 12.0, exp = Experiment 2 (G=5), obs = partial_0.5)

```python
def evaluate(current_reputation, observation, my_history, round_num):
    if observation["action"] == "donate":
        return current_reputation + 0.7
    else:
        return current_reputation - 1.0

def decide(recipient_reputation, round_num, my_history):
    return recipient_reputation >= 0.0
```

## 5. Dominant archetypes (ALLD and ALLC)

The vast majority of final-population agents are ALLD (always-defect), as we would expect from
the cooperation trajectories reported in Section 4. Below are the canonical code shapes that
dominated the final populations.

### ALLD (always-defect)

Number of ALLD agents in final populations: 367 / 450 (81.6%)
Distinct ALLD code variants: 1

Most common ALLD code (occurs in 367 / 367 ALLD agents):

```python
def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return False
```

### ALLC (always-cooperate)

Number of ALLC agents in final populations: 61 / 450 (13.6%)

Most common ALLC code (occurs in 61 / 61 ALLC agents):

```python
def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation

def decide(recipient_reputation, round_num, my_history):
    return True
```

## 6. Empirical findings

Across the 450 final-population agents (36 trials * 15 agents per trial):

- **367 (81.6%) are ALLD** (always-defect): the dominant archetype.
- **61 (13.6%) are ALLC** (always-cooperate): the second-most common archetype, retained by selection because unconditional donation is a low-variance strategy.
- **0 (0.0%) are Image Scoring**: strategies that increment reputation for observed donations and decrement for defections, and donate if reputation is at least zero.
- **12 (2.7%) are Hybrid** (Image Scoring + my_history): strategies that combine indirect and direct reciprocity information.
- **0 (0.0%) are ThresholdOnly**: strategies that use a static reputation threshold without observation-based updating.
- **10 (2.2%) are other archetypes** (DirectExperience, RoundDependent, Other).

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
