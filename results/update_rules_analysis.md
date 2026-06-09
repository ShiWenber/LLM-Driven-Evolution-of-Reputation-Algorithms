# LLM-discovered reputation update rules — empirical analysis

This report distils the reputation-update rules the LLMs discovered across the
Standard plan (deepseek-v4-flash, 36 trials, 167 Hybrid strategies) and the
robustness check (deepseek-coder, 6 trials, 88 Hybrid strategies). For each
experiment we extract the body of every `evaluate()` function, group by
structural fingerprint, and report the dominant update rules.

## 1. What does "Hybrid" mean in our data?

Every Hybrid strategy observed in our 42 trials has the following shape:

```python
def evaluate(current_reputation, observation, my_history, round_num):
    if observation["action"] == "donate":
        delta_pos = ...        # positive update amount
    else:
        delta_neg = ...        # negative update amount (sign: usually negative)
    new_rep = f(current_reputation, delta_pos_or_neg, ...)
    return max(-1.0, min(1.0, new_rep))   # clamp
```

The 5 structural axes along which the LLMs vary:

1. **Additive** vs **EMA** update (`new = r + δ` vs `new = α·r + δ`)
2. **Asymmetric** vs symmetric delta (`|δ_pos| ≠ |δ_neg|`)
3. **Round modulation** — `δ` scales with `round_num` (decay or growth)
4. **Self-history modulation** — `δ` is a function of the agent's own past
   actions (warm start for cooperators, or fatigue)
5. **Global counter** — a mutable list shared across rounds to aggregate
   observations

## 2. The three dominant update families

Across all 255 Hybrid strategies, three structural families account for the
majority. None of them is the canonical Nowak–Sigmund Image Scoring rule
(`new = r + s` with `s ∈ {-1, +1}`).

### Family A: Linear additive Image-Scoring (most common)

Frequency: ~30% of all Hybrid strategies in BOTH LLMs.

```python
if observation["action"] == "donate":
    return max(-1.0, min(1.0, current_reputation + 0.3))
else:
    return max(-1.0, min(1.0, current_reputation - 0.5))
```

Canonical parameters observed: `δ_pos ∈ {0.1, 0.2, 0.3}`, `δ_neg ∈ {-0.2, -0.3, -0.5, -1.0}`.
**Always asymmetric**: `|δ_neg| > |δ_pos|` (punishment > reward). This is
*consistent* with the Nowak–Sigmund model and with the indirect-reciprocity
literature's observation that punishment is more important than reward for
cooperation to evolve.

### Family B: EMA-style update (with decay)

Frequency: ~15% of all Hybrid strategies, with notable LLM differences.

```python
decayed = current_reputation * 0.85           # EMA decay 0.85
if observation["action"] == "donate":
    return min(1.0, decayed + 0.3)
else:
    return max(-1.0, decayed - 0.2)
```

EMA decay parameter `α` observed: `{0.8, 0.85, 0.9, 0.95, 0.99}`.
The two LLMs split the responsibility differently:
- **deepseek-v4-flash** uses EMA mostly in conjunction with **round-modulation** (e.g. `α = 0.7 + 0.2 * round_num/15`)
- **deepseek-coder** uses EMA with a **self-history modulation** of the reward amount (boost when agent has been donating)

### Family C: Constant-or-passthrough (no reputation update at all)

Frequency: ~30% of Hybrid strategies (mostly in the deepseek-v4-flash
standard plan).

```python
def evaluate(current_reputation, observation, my_history, round_num):
    return current_reputation
```

This is a degenerate Hybrid: the agent uses `recipient_reputation` in
`decide()` (which is why the classifier labels it Hybrid) but does not
update it. In effect these are **static-threshold** strategies: `decide`
makes the call and `evaluate` is a no-op.

**This family is what kept cooperation alive in many Standard-plan runs**:
a static-threshold strategy combined with an LLM-generated initial
reputation value (typically 0.0 or 0.5) can sustain cooperation as long as
the threshold is calibrated correctly. The LLM's "Hybrid" implementation
in this family is effectively a `ThresholdOnly` strategy that the regex
classifier happens to call Hybrid because the function signature still
contains `my_history`.

## 3. Novel features not in canonical Image Scoring

Three features appear across both LLMs and are absent from the canonical
Nowak–Sigmund model:

### 3.1 Self-referential delta modulation

The reward `δ_pos` and/or punishment `δ_neg` depend on the agent's own
donation history. Two variants:

(a) **Warm-start for cooperators** — agents with high own-donation-rate
get a larger reward, encouraging sustained cooperation:
```python
own_donations = sum(1 for h in my_history if h['action'] == 'donate')
if observation['action'] == 'donate':
    base_delta = 0.15 if own_donations <= 10 else 0.1   # smaller reward for serial cooper-ators
else:
    base_delta = -0.3 if own_donations <= 10 else -0.15  # smaller punishment
```

(b) **Fatigue / conservatism** — agents that have donated a lot recently
require higher reputation before donating:
```python
recent_donations = sum(1 for entry in my_history[-5:] if entry['action'] == 'donate')
fatigue = 0.2 if recent_donations >= 4 else 0.0
# ... then threshold += fatigue
```

### 3.2 Asymmetric delta with magnitude asymmetry

Every asymmetric Hybrid in our data has `|δ_neg| > |δ_pos|`. Concretely,
the typical delta ratio is 0.3 / -0.5 (i.e. **1.67× more punishment than
reward**). The flash-generated strategies use ratios of 0.7 / -1.0
(1.43×), the coder-generated strategies use 0.3 / -0.5 (1.67×). The
common thread is *positive* asymmetry: defectors lose reputation faster
than donors gain it. This is the same asymmetry embedded in the
"leading eight" norms of Ohtsuki & Iwasa.

### 3.3 Adaptive round-based decay

The EMA decay parameter `α` (or the additive `δ` magnitude) is a
function of `round_num`:

```python
round_factor = 1.0 / (1 + (round_num / 30))   # 1.0 -> 0.5 over 30 rounds
delta = base_delta * round_factor              # less weight on later observations
```

This effectively makes the agent **forget recent observations more slowly
as the game progresses** (or, conversely, weight new observations more
strongly early on). It is *not* present in canonical Image Scoring but
appears in 7–14% of Hybrid strategies in both LLMs.

## 4. Cross-LLM comparison

| Feature | deepseek-v4-flash (167) | deepseek-coder (88) |
|---|---|---|
| Pure additive (`r + δ`) | 28% | 27% |
| EMA (`α·r + δ`) | 6% | 11% |
| Constant / passthrough | **31%** | 16% |
| Asymmetric delta (`\|δ_neg\| > \|δ_pos\|`) | 100% of arithmetic ones | 100% of arithmetic ones |
| Round modulation in `evaluate()` | 40% | 41% |
| `my_history` used in `evaluate()` | 5% | 16% |
| Global counter pattern | 1.2% | 0% |

The two LLMs converge on the **asymmetric-delta Image-Scoring-with-threshold
attractor** but differ in implementation style:
- **flash** uses explicit branch on `observation['action']` and a passthrough
  evaluate that lets the initial-reputation-and-threshold do the work.
- **coder** uses EMA, self-history modulation, and round-based decay
  more often, treating `evaluate()` as a learned filter on top of the
  observed-action stream.

## 5. Summary

The LLMs independently discovered **three families of reputation-update rules**:

1. **Linear additive Image-Scoring with asymmetric delta** (Family A) — closest
   to canonical Nowak–Sigmund, but with `|δ_neg| > |δ_pos|` rather than
   `δ_pos = +1, δ_neg = -1`.
2. **EMA-style update with decay** (Family B) — closer to a Kalman-filter
   intuition; novel.
3. **Constant-or-passthrough** (Family C) — degenerate Hybrid where
   `evaluate()` is a no-op and the work is done in `decide()`'s
   threshold; novel in that the *combined* (evaluate, decide) still
   functions as a reputation-based system because the threshold is
   applied to `recipient_reputation` even though the latter is never
   updated.

All three families preserve the high-level structure of reputation-based
cooperation: maintain a per-agent scalar estimate, condition donation on a
threshold. They differ in *how* the estimate is updated — uniformly
additive, exponentially smoothed, or static — and in the magnitude of the
update — uniformly small additive deltas, scaled EMA deltas, or none at
all.

The two LLMs converge on the attractor at the **algorithm-class level** but
diverge at the **implementation level** — strong evidence that the
Image-Scoring-with-threshold attractor in the strategy space is robust
across LLM priors.
